# Chapter 4: GPU Programming — Thousands of Chapter 1's Workers, at Once

> "A `SIMD[DType.float32, 4]` register gave you four workers stapling in lockstep. A GPU gives you thousands of workers, organized into crews, each one running the exact same instructions as the others — but no longer four, and no longer required to stay in perfect lockstep with every other crew. The programming model this chapter introduces is what makes 'thousands' as manageable to reason about as 'four' was in Chapter 1."

**What you will understand by the end of this chapter:**

- The GPU **thread hierarchy** — grid, block, thread — and how to compute a thread's unique **global index** by hand from its position in that hierarchy, the number every kernel actually needs to know which data element it owns
- The **host/device execution model**: which code runs on the CPU (allocating memory, launching work) and which runs on the GPU (the actual parallel computation), and why they are, in a real sense, two separate computers cooperating on one program
- The three GPU **memory scopes** — global, shared, local — and the speed/scope tradeoff each one makes
- **Memory coalescing**, which is Chapter 3's bus-utilization argument again, now at a scale where the penalty for getting it wrong is far larger, because a GPU services dozens of threads per memory request instead of the 4–16 lanes a CPU's SIMD register handles
- How **broadcasting** (one thread per output element, no coordination needed) and **matrix multiplication** (one thread per output element, each running its own dot product) map onto this thread model, traced by hand on small, complete examples

**What you need to know first:**

- Chapter 1.4 (SIMD as a first-class type) — a GPU thread grid is best understood as SIMD's "many workers doing the same thing at once" idea, scaled up by several orders of magnitude
- Chapter 3 in full, especially Section 3.1's bus-utilization argument and Section 3.3's SoA-enables-SIMD-loads argument — both reappear directly in Section 4.4

## 4.1 The Thread Hierarchy: Grid, Block, Thread `[FOUNDATIONAL]`

### Intuition

An army is organized into divisions, and each division is made up of platoons, and each platoon is made up of individual soldiers. Every soldier has a seat number *within their own platoon* — but that number alone doesn't identify them across the whole army, since platoon 0's seat 5 and platoon 3's seat 5 are two completely different people. What uniquely identifies a soldier across the entire army is the *combination* of which platoon they're in and their seat within it.

A GPU's **thread hierarchy** is exactly this structure. A **grid** is the whole army; a **block** is a platoon (Mojo and CUDA cap a block at 1024 threads); a **thread** is one soldier. Every thread knows its own seat within its block (`thread_idx`) and its block's position within the grid (`block_idx`) — but neither number alone says which data element that thread owns. Only the combination does.

### Background

For the common case of one-dimensional work (processing a flat array), a thread's **global index** — the number that actually corresponds to "which array element is mine" — is computed as:

```
global_id = block_idx * block_dim + thread_idx
```

where `block_dim` is the number of threads per block (a fixed configuration value, not something the thread queries about itself).

| | Grid | Block | Thread |
|---|---|---|---|
| Analogy | The whole army | One platoon | One soldier |
| Typical size | As many blocks as needed to cover the data | Up to 1024 threads | 1 execution unit |
| Local identity | — | `block_idx` (this block's position in the grid) | `thread_idx` (this thread's seat within its block) |
| Identity that maps to data | — | — | `global_id = block_idx * block_dim + thread_idx` |

### Worked Example 4.1.1 — Computing every global ID by hand

```mojo
alias THREADS_PER_BLOCK = 8
alias BLOCKS_PER_GRID = 4
alias TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_PER_GRID   # 32
```

Trace block 2, thread 5: `global_id = 2 × 8 + 5 = 21`. Trace all of block 0 (`block_idx = 0`): thread 0 → global ID `0×8+0=0`, thread 1 → `1`, ..., thread 7 → `7` — block 0 owns global IDs `0` through `7`. Trace all of block 3 (`block_idx = 3`): thread 0 → `3×8+0=24`, ..., thread 7 → `3×8+7=31` — block 3 owns global IDs `24` through `31`. Across all four blocks, every global ID from `0` to `31` is produced by exactly one `(block_idx, thread_idx)` pair — `TOTAL_THREADS = 32` distinct workers, each with a distinct, unambiguous piece of the data.

### ASCII Diagram — 4 blocks, 8 threads each, 32 unique global IDs

```
Block 0: thread_idx 0 1 2 3 4 5 6 7   -> global_id  0  1  2  3  4  5  6  7
Block 1: thread_idx 0 1 2 3 4 5 6 7   -> global_id  8  9 10 11 12 13 14 15
Block 2: thread_idx 0 1 2 3 4 5 6 7   -> global_id 16 17 18 19 20 21 22 23
                                                              ^^ (block 2, thread 5) -> 21
Block 3: thread_idx 0 1 2 3 4 5 6 7   -> global_id 24 25 26 27 28 29 30 31
```

> `[COMMON TRAP]` `thread_idx` resets to `0` at the start of *every* block, so "thread 5" is not a unique name anywhere in the grid — block 0's thread 5 (global ID `5`) and block 2's thread 5 (global ID `21`) are two completely different workers touching two completely different array elements. Any kernel that indexes memory using `thread_idx` alone, instead of the combined `global_id`, will have every block stepping on the same handful of elements near the start of the array while leaving the rest of the array completely untouched.

## 4.2 Host and Device: Two Machines, One Program `[FOUNDATIONAL]`

### Intuition

A construction foreman standing at street level doesn't personally lift a single brick. Their job is entirely coordination: draw up the plan, hand out materials, tell the crew on the scaffolding "go," and collect the finished wall once the crew reports it's done. The wall itself is built entirely by the crew, working simultaneously across the whole face of the building — the foreman never touches a brick, and the crew never decides what to build next on their own.

The **host** (the CPU) is the foreman. The **device** (the GPU) is the crew. A GPU program is really two programs cooperating: ordinary sequential Mojo code that runs on the host — allocating memory, copying data over, deciding on a grid/block configuration, launching the work, and later reading the result back — and a single function, the **kernel**, that the device runs simultaneously as thousands of independent copies, one per thread, each one seeing its own `thread_idx` and `block_idx`.

### Background

| | Host (CPU) | Device (GPU) |
|---|---|---|
| Runs | Ordinary sequential code | One kernel function, many simultaneous copies |
| Responsible for | Allocating memory, copying data, launching kernels, reading results | Executing the kernel body once per thread |
| A loop over `N` elements looks like | One loop, `N` iterations, one after another | One kernel body, launched with enough threads to cover all `N` elements at once (bounds-checked) |

### Worked Example 4.2.1 — The same computation, host-only, traced by hand

```mojo
fn simple_vector_add_cpu(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                        result: UnsafePointer[Float32], size: Int):
    for i in range(size):
        result[i] = a[i] + b[i]
```

With `a[i] = Float32(i)` and `b[i] = Float32(i * 2)`, trace the first three iterations of this purely host-side loop: `i=0`: `a=0.0, b=0.0, result=0.0+0.0=0.0`; `i=1`: `a=1.0, b=2.0, result=1.0+2.0=3.0`; `i=2`: `a=2.0, b=4.0, result=2.0+4.0=6.0`. For `size = 1000`, this loop runs its body 1000 times, strictly one after another — there is no parallelism here at all, and none is claimed; this is the baseline the device version below is measured against.

### Worked Example 4.2.2 — The same math, reframed for the device

The device version of this exact computation launches enough threads to cover all 1000 elements — say, 4 blocks of 256 threads each (`⌈1000/256⌉ = 4`) — and every one of those `4 × 256 = 1024` threads runs the *same* kernel body simultaneously:

```mojo
# conceptual device kernel body, one copy per thread:
var global_id = block_idx.x * block_dim.x + thread_idx.x    # Section 4.1's formula
if global_id < size:                                         # bounds check -- 1024 threads launched, only 1000 needed
    result[global_id] = a[global_id] + b[global_id]
```

Thread with `global_id = 2` computes exactly what host iteration `i=2` computed above — `2.0 + 4.0 = 6.0` — but it does so *at the same moment* as the thread computing `global_id = 999`, not after 997 other iterations have already run. The bounds check `global_id < size` matters because `1024` threads were launched to comfortably cover `1000` elements in whole blocks of `256`; the `24` threads with `global_id` from `1000` to `1023` simply do nothing, protected by that one `if`.

### ASCII Diagram — host timeline vs. device timeline

```
Host (CPU) timeline for one kernel launch:
 [ allocate memory ] -> [ copy data to GPU ] -> [ launch kernel ] -> [ copy result back ]
                                                        |
                                                        v
Device (GPU), during "launch kernel":
 Block 0: threads 0..255   \
 Block 1: threads 0..255    \  all four blocks' threads run
 Block 2: threads 0..255    /  the SAME kernel body simultaneously
 Block 3: threads 0..255   /
```

## 4.3 The Memory Hierarchy On a GPU `[FOUNDATIONAL]`

### Intuition

Extend Chapter 1.5's hotel-room picture across an entire city. **Global memory** is a public warehouse on the other side of town: any soldier from any platoon can drive there, but the drive is slow, and everyone shares the same access road. **Shared memory** is a supply closet built directly into one platoon's own barracks: any soldier in that specific platoon can reach it almost instantly, but soldiers from other platoons can't use it at all, and its contents disappear the moment that platoon is dismissed. **Local memory** is the single backpack strapped to one soldier's own back: the fastest thing to reach by far, but it belongs to that one soldier alone, and it's gone the instant their assignment ends.

### Background

| | Global memory | Shared memory | Local (per-thread) memory |
|---|---|---|---|
| Visible to | Every thread, every block | Every thread *in one block* | One single thread |
| Speed | Slowest | Fast | Fastest |
| Typical size | Large (GBs) | Small (tens of KB per block) | Tiny, per-thread |
| Lifetime | Whole kernel launch (and beyond) | One block's execution | One thread's execution |

### Worked Example 4.3.1 — Where the vector-add kernel's data actually lives

In Section 4.2's kernel, `a`, `b`, and `result` are all **global memory** — every thread across every block needs to be able to read its own `a[global_id]` and `b[global_id]` and write its own `result[global_id]`, and threads in *different* blocks have no other way to see each other's data at all. Contrast this with a kernel that reuses the *same* small chunk of data across many threads *within one block* — Part 5's tiled matrix multiplication is exactly this case — where staging that chunk once into that block's shared memory lets every thread in the block read it at shared-memory speed instead of each thread separately paying global memory's slow round trip for identical bytes.

> `[COMMON TRAP]` "Shared memory" is easy to mistake for "the GPU's general-purpose RAM," as though it were just a faster version of global memory available to everyone. It is not general-purpose at all — it is a small, per-block scratchpad (typically tens of kilobytes), explicitly staged into and out of by the kernel's own code, and its entire contents vanish the moment that specific block finishes executing. A value written to shared memory by one block is completely invisible to every other block, always.

## 4.4 Memory Coalescing: Chapter 3's Bandwidth Argument, at GPU Scale `[FOUNDATIONAL]`

### Intuition

Recall Chapter 3.1's movers, forced to carry a whole mixed box just to deliver the one item you wanted. Now send 32 movers to a warehouse at the same instant — a **warp**, the group of threads a GPU actually schedules and executes together. If the 32 boxes they each need happen to sit on one single shelf, right next to each other, one truck makes one trip and grabs the entire shelf at once. If instead those 32 boxes are scattered across 32 different aisles, the same 32 pickups can now cost up to 32 separate trips — for the exact same 32 boxes of useful cargo.

### Background

Real GPU hardware groups threads into **warps** of 32, and when every thread in a warp requests an address that falls inside the same aligned memory block, the hardware services the entire warp with a single **coalesced** transaction. When the 32 addresses are scattered, the hardware may need up to 32 separate transactions to service the same warp — up to a 32× bandwidth penalty for delivering the identical amount of useful data. This is Section 3.1's bus-utilization argument again, just at a scale where a CPU's 4-to-16-lane SIMD register becomes a 32-thread warp, and the gap between "coalesced" and "not" grows accordingly.

### Worked Example 4.4.1 — Three access patterns, one small example scaled down for hand-tracing

To keep the arithmetic tractable, use a block of 4 threads (instead of a full 32-thread warp) and treat every 4 consecutive `Float32`s (16 bytes) as one memory transaction — the same principle real hardware applies to 32 floats (128 bytes) per warp, just scaled down. The data array is `data[i] = Float32(i)` for `i = 0..15`, which splits into four 16-byte chunks: `chunk0 = [0,1,2,3]`, `chunk1 = [4,5,6,7]`, `chunk2 = [8,9,10,11]`, `chunk3 = [12,13,14,15]`.

**Coalesced** — block 0's four threads access `data[0], data[1], data[2], data[3]`: all four fall inside `chunk0` alone.

```
transactions needed = 1   (chunk0)
bytes moved         = 16
bytes used          = 16
utilization         = 16/16 = 100%
```

**Strided**, stride of 4 elements — thread 0 → `data[0]` (`chunk0`), thread 1 → `data[4]` (`chunk1`), thread 2 → `data[8]` (`chunk2`), thread 3 → `data[12]` (`chunk3`): four distinct chunks.

```
transactions needed = 4   (chunk0, chunk1, chunk2, chunk3)
bytes moved         = 4 x 16 = 64
bytes used          = 4 x 4  = 16   (one float actually needed per chunk fetched)
utilization         = 16/64 = 25%
```

**Random** — threads access `data[7], data[2], data[11], data[5]`: `7` and `5` both fall in `chunk1`, `2` falls in `chunk0`, `11` falls in `chunk2` — three distinct chunks, not four, purely by chance.

```
transactions needed = 3   (chunk0, chunk1, chunk2)
bytes moved         = 3 x 16 = 48
bytes used          = 16
utilization         = 16/48 ≈ 33.3%
```

Random access happened to land a hair better than fully strided access here — but only by luck (two of its four indices happened to share a chunk); nothing about the access pattern *guarantees* that, the way coalesced access guarantees exactly one transaction every time.

### ASCII Diagram — one chunk vs. four scattered chunks

```
Coalesced (1 transaction):        Strided (4 transactions):
 chunk0: [d0 d1 d2 d3] <- all 4    chunk0:[d0 ..] <-t0   chunk1:[.. d4 ..] <-t1
                                   chunk2:[.. d8 ..] <-t2  chunk3:[.. d12] <-t3
```

> `[COMMON TRAP]` The fix for strided/random access is the same fix Chapter 3 already argued for on entirely separate grounds: lay data out as **Struct-of-Arrays**, so that the values a warp needs for one operation are contiguous with each other rather than interleaved with fields it doesn't need. "Use SoA layout for better coalescing" isn't a new GPU-specific rule — it's Chapter 3.3's SIMD-load argument, restated for 32-wide warps instead of 4-wide SIMD registers, because the underlying requirement (the needed values must be neighbors in memory) is identical at both scales.

## 4.5 Broadcasting: One Thread, One Output Element `[FOUNDATIONAL]`

### Intuition

Picture a banquet's seating chart, where every seat is labeled with a row and a column, and the guest in that seat has exactly one job: read the placard for their row, read the placard for their column, add the two together, and set that sum as their own place card. No guest needs to speak to any other guest to do this — every seat's answer depends only on its own row and column labels.

### Background

Broadcasting a `(1, N)` row vector against an `(N, 1)` column vector into an `(N, N)` output is precisely this pattern: thread `(row, col)` reads `a[col]` (the row vector's `col`-th entry) and `b[row]` (the column vector's `row`-th entry), adds them, and writes `output[row][col]` — entirely independent of every other thread, with no synchronization needed anywhere. This is exactly why broadcasting is singled out in this book's own list of GPU-friendly operations: every output element's computation is already, by construction, a completely separate unit of work.

### Worked Example 4.5.1 — The full `2×2` case, every thread traced

With `a = [0, 1]` and `b = [0, 1]` (`SIZE = 2`):

```
Thread(row=0, col=0): a[0] + b[0] = 0 + 0 = 0
Thread(row=0, col=1): a[1] + b[0] = 1 + 0 = 1
Thread(row=1, col=0): a[0] + b[1] = 0 + 1 = 1
Thread(row=1, col=1): a[1] + b[1] = 1 + 1 = 2
```

```
output = [ [0, 1],
           [1, 2] ]
```

Verification compares this against `expected[row][col] = row + col` directly: `expected[0][0]=0, expected[0][1]=1, expected[1][0]=1, expected[1][1]=2` — an exact match with every entry computed above, which is what this chapter's code reports as `Verification: PASSED`.

### ASCII Diagram — 4 independent threads, 4 independent answers

```
        col=0        col=1
row=0  [a0+b0=0]    [a1+b0=1]
row=1  [a0+b1=1]    [a1+b1=2]

Each cell computed by its own thread, with no dependency on
any other cell -- all four could run in any order, or all at
the exact same instant, and the result is identical either way.
```

## 4.6 Matrix Multiplication: One Thread, One Dot Product `[FOUNDATIONAL]`

### Intuition

An assembly line where each worker is assigned exactly one finished product — one cell of the result matrix — and is handed the one full row of ingredients from `A` and the one full column of ingredients from `B` needed to assemble it. Unlike the broadcast case, this worker has real work to do: a small loop, multiplying and accumulating pairs of numbers. But just like the broadcast case, no worker ever needs to see another worker's row, column, or partial result.

### Background

Thread `(row, col)` computes `output[row][col] = Σ_k A[row][k] × B[k][col]` — its own independent dot product between one row of `A` and one column of `B`. This is the "thread grid: (m, n)" configuration this chapter's `simulate_gpu_matrix_multiply` builds: an `m × n` output means `m × n` threads, one per output cell, each running its own inner loop over `k`.

### Worked Example 4.6.1 — A complete `2×2 @ 2×2`, every thread's dot product traced

```
A = [ 1  2 ]        B = [ 5  6 ]
    [ 3  4 ]            [ 7  8 ]
```

```
Thread(0,0): A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 =  5+14 = 19
Thread(0,1): A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 =  6+16 = 22
Thread(1,0): A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 15+28 = 43
Thread(1,1): A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 18+32 = 50
```

```
C = A @ B = [ 19  22 ]
            [ 43  50 ]
```

Four threads, four independent 2-term dot products, one shared final matrix — the classic `2×2` matrix product, arrived at exactly as `simulate_gpu_matrix_multiply`'s nested loops describe: one thread per `(row, col)`, each summing over `k`.

### ASCII Diagram — 4 threads, each running its own 2-term loop

```
Thread(0,0): k=0: 1*5=5   k=1: 2*7=14   sum=19
Thread(0,1): k=0: 1*6=6   k=1: 2*8=16   sum=22
Thread(1,0): k=0: 3*5=15  k=1: 4*7=28   sum=43
Thread(1,1): k=0: 3*6=18  k=1: 4*8=32   sum=50
```

> `[COMMON TRAP]` This simulation has every thread read straight from global memory for every single multiply — and `Thread(0,0)` and `Thread(0,1)` both read the *entire same row* of `A` independently, paying global memory's slow round trip twice for identical bytes. A production GPU matrix multiply avoids exactly this by having threads *within one block* cooperatively stage a shared tile of `A` and `B` into shared memory (Section 4.3) once, then have every thread in that block read from the fast shared copy instead — the specific optimization Chapter 9 (GPU Kernel Implementation) builds out in full. Nothing about this section's version is wrong; it is deliberately the simplest correct version, with the performance work still ahead of it.

## 4.7 Complete Runnable Code

### File: `15_gpu_basics.mojo` — Sections 4.1–4.3

**Execution:** `pixi run mojo 15_gpu_basics.mojo`

```mojo
from memory import UnsafePointer

fn simple_vector_add_cpu(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                        result: UnsafePointer[Float32], size: Int):
    """CPU version of vector addition for comparison."""
    for i in range(size):
        result[i] = a[i] + b[i]

fn gpu_basics_demo():
    """Demonstrate basic GPU concepts without GPU execution."""
    print("=== GPU Programming Basics ===")

    print("GPU Programming Concepts:")
    print("1. Thread Hierarchy:")
    print("   - Grid: Collection of thread blocks")
    print("   - Block: Collection of threads (up to 1024 threads)")
    print("   - Thread: Individual execution unit")

    print("\n2. Memory Types:")
    print("   - Global Memory: Large, slow, accessible by all threads")
    print("   - Shared Memory: Fast, shared within a block")
    print("   - Local Memory: Per-thread private memory")

    print("\n3. Execution Model:")
    print("   - Kernel: Function that runs on GPU")
    print("   - Host: CPU code that launches kernels")
    print("   - Device: GPU that executes kernels")

    # Demonstrate CPU vector addition for comparison
    var size = 1000
    var a = UnsafePointer[Float32].alloc(size)
    var b = UnsafePointer[Float32].alloc(size)
    var result = UnsafePointer[Float32].alloc(size)

    # Initialize test data
    for i in range(size):
        a[i] = Float32(i)
        b[i] = Float32(i * 2)

    # CPU computation
    simple_vector_add_cpu(a, b, result, size)

    print("\nCPU Vector Addition Results:")
    print("a[0:5] =", a[0], a[1], a[2], a[3], a[4])
    print("b[0:5] =", b[0], b[1], b[2], b[3], b[4])
    print("result[0:5] =", result[0], result[1], result[2], result[3], result[4])

    a.free()
    b.free()
    result.free()

fn main():
    """Main function for GPU basics demonstration."""
    gpu_basics_demo()
```

### File: `16_thread_indexing.mojo` — Section 4.1

**Execution:** `pixi run mojo 16_thread_indexing.mojo`

```mojo
from memory import UnsafePointer

fn simulate_gpu_thread_indexing():
    """Simulate GPU thread indexing patterns."""
    print("=== GPU Thread Indexing Simulation ===")

    # Simulate GPU grid configuration
    alias THREADS_PER_BLOCK = 8
    alias BLOCKS_PER_GRID = 4
    alias TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_PER_GRID

    print("Grid Configuration:")
    print("  Threads per block:", THREADS_PER_BLOCK)
    print("  Blocks per grid:", BLOCKS_PER_GRID)
    print("  Total threads:", TOTAL_THREADS)

    print("\nThread Index Calculation:")
    print("  thread_idx = block_idx * block_dim + thread_idx_within_block")
    print()

    # Simulate what each thread would compute
    for block_id in range(BLOCKS_PER_GRID):
        print("Block", block_id, ":")
        for thread_in_block in range(THREADS_PER_BLOCK):
            var global_thread_id = block_id * THREADS_PER_BLOCK + thread_in_block
            print("  Thread", thread_in_block, "-> Global ID:", global_thread_id)

    print("\nData Parallel Processing:")
    var data_size = 32
    var data = UnsafePointer[Float32].alloc(data_size)
    var result = UnsafePointer[Float32].alloc(data_size)

    # Initialize data
    for i in range(data_size):
        data[i] = Float32(i)

    # Simulate parallel processing (each "thread" processes one element)
    print("Simulating GPU kernel: result[i] = data[i] * 2 + 1")
    for block_id in range(BLOCKS_PER_GRID):
        for thread_in_block in range(THREADS_PER_BLOCK):
            var global_id = block_id * THREADS_PER_BLOCK + thread_in_block
            if global_id < data_size:
                result[global_id] = data[global_id] * 2 + 1

    print("Results (first 16 elements):")
    for i in range(16):
        print("  data[" + String(i) + "] =", data[i], "-> result[" + String(i) + "] =", result[i])

    data.free()
    result.free()

fn main():
    """Main function for thread indexing demonstration."""
    simulate_gpu_thread_indexing()
```

### File: `17_memory_patterns.mojo` — Section 4.4

**Execution:** `pixi run mojo 17_memory_patterns.mojo`

```mojo
from memory import UnsafePointer

fn demonstrate_memory_coalescing():
    """Demonstrate memory access patterns for GPU optimization."""
    print("=== GPU Memory Access Patterns ===")

    var size = 16
    var threads_per_block = 4
    var data = UnsafePointer[Float32].alloc(size)

    # Initialize data
    for i in range(size):
        data[i] = Float32(i)

    print("Data array:", end=" ")
    for i in range(size):
        print(data[i], end=" ")
    print()

    print("\n1. Coalesced Access Pattern (GOOD):")
    print("   Adjacent threads access adjacent memory locations")
    print("   Thread 0 -> data[0], Thread 1 -> data[1], etc.")

    # Simulate coalesced access
    for block in range(size // threads_per_block):
        print("   Block", block, "threads access:", end=" ")
        for thread in range(threads_per_block):
            var index = block * threads_per_block + thread
            print("data[" + String(index) + "]", end=" ")
        print()

    print("\n2. Strided Access Pattern (BAD):")
    print("   Threads access memory with large strides")
    print("   Thread 0 -> data[0], Thread 1 -> data[4], etc.")

    # Simulate strided access
    var stride = 4
    for thread in range(threads_per_block):
        var index = thread * stride
        if index < size:
            print("   Thread", thread, "accesses data[" + String(index) + "]")

    print("\n3. Random Access Pattern (WORST):")
    print("   Threads access memory randomly - no pattern")
    var random_indices = List[Int]()
    random_indices.append(7)
    random_indices.append(2)
    random_indices.append(11)
    random_indices.append(5)

    for i in range(len(random_indices)):
        print("   Thread", i, "accesses data[" + String(random_indices[i]) + "]")

    print("\nMemory Coalescing Rules:")
    print("  + Adjacent threads should access adjacent memory")
    print("  + 32-thread warps should access 128-byte aligned blocks")
    print("  + Avoid bank conflicts in shared memory")
    print("  + Use SoA layout for better coalescing")

    data.free()

fn main():
    """Main function for memory patterns demonstration."""
    demonstrate_memory_coalescing()
```

### File: `18_broadcast_kernel_sim.mojo` — Section 4.5

**Execution:** `pixi run mojo 18_broadcast_kernel_sim.mojo`

```mojo
from memory import UnsafePointer

struct Matrix2D:
    """Simple 2D matrix structure for GPU kernel simulation."""
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Float32].alloc(rows * cols)

    fn __del__(owned self):
        self.data.free()

    fn get(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn set(self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn print_matrix(self, name: String):
        print(name + ":")
        for i in range(self.rows):
            print("  [", end="")
            for j in range(self.cols):
                print(self.get(i, j), end="")
                if j < self.cols - 1:
                    print(", ", end="")
            print("]")

fn simulate_broadcast_add_kernel(output: Matrix2D, a: Matrix2D, b: Matrix2D):
    """Simulate GPU broadcasting kernel like the example provided."""
    print("=== Broadcasting Addition Kernel Simulation ===")

    print("Kernel Logic:")
    print("  output[row][col] = a[0][col] + b[row][0]")
    print("  Broadcasting (1,N) + (N,1) -> (N,N)")

    # Simulate GPU threads - each thread handles one output element
    for row in range(output.rows):
        for col in range(output.cols):
            # This simulates what each GPU thread would do:
            # thread_idx.y = row, thread_idx.x = col
            var a_val = a.get(0, col)      # a is (1, cols) - broadcast row
            var b_val = b.get(row, 0)      # b is (rows, 1) - broadcast column
            var result = a_val + b_val
            output.set(row, col, result)

            print("  Thread(" + String(row) + "," + String(col) + "): " +
                  String(a_val) + " + " + String(b_val) + " = " + String(result))

fn demonstrate_gpu_broadcasting():
    """Demonstrate GPU broadcasting patterns."""
    print("=== GPU Broadcasting Demonstration ===")

    alias SIZE = 3

    # Create matrices for broadcasting
    var output = Matrix2D(SIZE, SIZE)     # (3, 3) output
    var a = Matrix2D(1, SIZE)            # (1, 3) - row vector
    var b = Matrix2D(SIZE, 1)            # (3, 1) - column vector

    # Initialize input matrices
    print("Initializing input matrices:")
    for i in range(SIZE):
        a.set(0, i, Float32(i))          # a = [0, 1, 2]
        b.set(i, 0, Float32(i))          # b = [0; 1; 2]

    a.print_matrix("Matrix A (1x3)")
    b.print_matrix("Matrix B (3x1)")

    # Simulate GPU kernel execution
    simulate_broadcast_add_kernel(output, a, b)

    print("\nResult:")
    output.print_matrix("Output (3x3)")

    print("\nGPU Execution Model:")
    print("  Grid dim: (1, 1, 1)")
    print("  Block dim: (3, 3, 1)")
    print("  Total threads: 9")
    print("  Each thread computes one output element")

fn main():
    """Main function for broadcasting demonstration."""
    demonstrate_gpu_broadcasting()
```

### File: `19_tensor_operations_gpu.mojo` — Section 4.6

**Execution:** `pixi run mojo 19_tensor_operations_gpu.mojo`

```mojo
from memory import UnsafePointer

struct GPUTensorSim[dtype: DType]:
    """Simulated GPU tensor for automatic differentiation."""
    var data: UnsafePointer[Scalar[dtype]]
    var gradients: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var size: Int
    var requires_grad: Bool

    fn __init__(out self, shape: List[Int], requires_grad: Bool = False):
        self.shape = shape
        self.requires_grad = requires_grad

        # Calculate total size - fixed List iteration
        self.size = 1
        for i in range(len(shape)):
            self.size *= shape[i]

        # Allocate memory
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        if requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            # Initialize gradients to zero
            for i in range(self.size):
                self.gradients[i] = Scalar[dtype](0)
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor copying."""
        self.requires_grad = existing.requires_grad
        self.size = existing.size

        # Copy shape
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])

        # Allocate and copy data
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

        # Allocate and copy gradients if needed
        if self.requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            for i in range(self.size):
                self.gradients[i] = existing.gradients[i]
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        self.data.free()
        if self.requires_grad:
            self.gradients.free()

    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with constant value."""
        for i in range(self.size):
            self.data[i] = value

    fn simulate_gpu_elementwise_add(self, other: GPUTensorSim[dtype]) -> GPUTensorSim[dtype]:
        """Simulate GPU element-wise addition kernel."""
        print("Simulating GPU kernel: elementwise_add")

        var result = GPUTensorSim[dtype](self.shape, self.requires_grad or other.requires_grad)

        # Simulate GPU execution with thread blocks
        alias THREADS_PER_BLOCK = 256
        var num_blocks = (self.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        print("  Grid config: " + String(num_blocks) + " blocks, " + String(THREADS_PER_BLOCK) + " threads per block")
        print("  Total threads: " + String(num_blocks * THREADS_PER_BLOCK))
        print("  Elements to process: " + String(self.size))

        # Simulate what each GPU thread would do
        for block_id in range(num_blocks):
            for thread_id in range(THREADS_PER_BLOCK):
                var global_id = block_id * THREADS_PER_BLOCK + thread_id

                # Bounds check (important in GPU kernels)
                if global_id < self.size:
                    result.data[global_id] = self.data[global_id] + other.data[global_id]

        return result

    fn simulate_gpu_matrix_multiply(self, other: GPUTensorSim[dtype]) -> GPUTensorSim[dtype]:
        """Simulate GPU matrix multiplication kernel."""
        print("Simulating GPU kernel: matrix_multiply")

        # Assume 2D matrices for simplicity
        var m = self.shape[0]
        var k = self.shape[1]
        var n = other.shape[1]

        var result_shape = List[Int]()
        result_shape.append(m)
        result_shape.append(n)
        var result = GPUTensorSim[dtype](result_shape, self.requires_grad or other.requires_grad)

        print("  Matrix A: " + String(m) + "x" + String(k))
        print("  Matrix B: " + String(k) + "x" + String(n))
        print("  Result C: " + String(m) + "x" + String(n))

        # Simulate GPU thread grid (one thread per output element)
        print("  Thread grid: (" + String(m) + ", " + String(n) + ")")

        for row in range(m):
            for col in range(n):
                var sum: Scalar[dtype] = 0

                # Each thread computes one dot product
                for i in range(k):
                    var a_val = self.data[row * k + i]
                    var b_val = other.data[i * n + col]
                    sum += a_val * b_val

                result.data[row * n + col] = sum

        return result

    fn print_tensor(self, name: String):
        """Print tensor values."""
        print(name + " (shape: [", end="")
        for i in range(len(self.shape)):
            print(self.shape[i], end="")
            if i < len(self.shape) - 1:
                print(", ", end="")
        print("]):")

        if len(self.shape) == 1:
            # Vector
            print("  [", end="")
            for i in range(self.size):
                print(self.data[i], end="")
                if i < self.size - 1:
                    print(", ", end="")
            print("]")
        elif len(self.shape) == 2:
            # Matrix
            var rows = self.shape[0]
            var cols = self.shape[1]
            for i in range(rows):
                print("  [", end="")
                for j in range(cols):
                    print(self.data[i * cols + j], end="")
                    if j < cols - 1:
                        print(", ", end="")
                print("]")

fn demonstrate_gpu_tensor_operations():
    """Demonstrate GPU tensor operations for automatic differentiation."""
    print("=== GPU Tensor Operations for Automatic Differentiation ===")

    # Create test tensors
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)

    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(2)

    var tensor_a = GPUTensorSim[DType.float32](shape1, True)
    var tensor_b = GPUTensorSim[DType.float32](shape2, True)

    # Initialize with test data
    for i in range(tensor_a.size):
        tensor_a.data[i] = Float32(i + 1)

    for i in range(tensor_b.size):
        tensor_b.data[i] = Float32(i + 1) * 0.1

    tensor_a.print_tensor("Tensor A")
    tensor_b.print_tensor("Tensor B")

    # Simulate GPU matrix multiplication
    print("\nGPU Matrix Multiplication:")
    var result = tensor_a.simulate_gpu_matrix_multiply(tensor_b)
    result.print_tensor("Result C = A @ B")

    print("\nGPU Programming Benefits for AD:")
    print("  + Massive parallelization of tensor operations")
    print("  + Efficient gradient computation across thousands of parameters")
    print("  + Memory bandwidth optimization for large tensors")
    print("  + Concurrent forward and backward pass execution")
    print("  + Optimal for deep learning workloads")

fn main():
    """Main function for GPU tensor operations demonstration."""
    demonstrate_gpu_tensor_operations()
```

### File: `20_gpu_programming_complete.mojo` — Section 4.5, fully verified

**Execution:** `pixi run mojo 20_gpu_programming_complete.mojo`

```mojo
from memory import UnsafePointer

# Simulate the GPU programming concepts from the provided example
fn simulate_broadcast_add_complete():
    """Complete simulation of GPU broadcasting addition kernel."""
    print("=== Complete GPU Programming Demonstration ===")

    # Configuration matching the provided example
    alias SIZE = 2
    alias BLOCKS_PER_GRID = 1
    alias THREADS_PER_BLOCK_X = 3
    alias THREADS_PER_BLOCK_Y = 3

    print("GPU Configuration:")
    print("  Grid dimensions: (" + String(BLOCKS_PER_GRID) + ", 1, 1)")
    print("  Block dimensions: (" + String(THREADS_PER_BLOCK_X) + ", " + String(THREADS_PER_BLOCK_Y) + ", 1)")
    print("  Total threads: " + String(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y))

    # Create tensors
    var output = UnsafePointer[Float32].alloc(SIZE * SIZE)
    var a = UnsafePointer[Float32].alloc(SIZE)        # (1, SIZE) broadcasted
    var b = UnsafePointer[Float32].alloc(SIZE)        # (SIZE, 1) broadcasted

    # Initialize input tensors (matching the example)
    for i in range(SIZE):
        a[i] = Float32(i)      # a = [0, 1]
        b[i] = Float32(i)      # b = [0, 1]

    print("\nInput tensors:")
    print("  a (1x" + String(SIZE) + " broadcasted): [", end="")
    for i in range(SIZE):
        print(a[i], end="")
        if i < SIZE - 1:
            print(", ", end="")
    print("]")

    print("  b (" + String(SIZE) + "x1 broadcasted): [", end="")
    for i in range(SIZE):
        print(b[i], end="")
        if i < SIZE - 1:
            print(", ", end="")
    print("]")

    print("\nSimulating GPU kernel execution:")
    print("  Kernel: broadcast_add(output, a, b)")
    print("  Operation: output[row][col] = a[col] + b[row]")

    # Simulate GPU threads executing the kernel
    for thread_y in range(SIZE):  # Simulates thread_idx.y
        for thread_x in range(SIZE):  # Simulates thread_idx.x
            var row = thread_y
            var col = thread_x

            # This is what each GPU thread computes
            var a_val = a[col]        # a[j] where j = col (broadcasting)
            var b_val = b[row]        # b[i] where i = row (broadcasting)
            var result = a_val + b_val

            output[row * SIZE + col] = result

            print("    Thread(" + String(thread_y) + "," + String(thread_x) +
                  "): a[" + String(col) + "] + b[" + String(row) + "] = " +
                  String(a_val) + " + " + String(b_val) + " = " + String(result))

    print("\nOutput tensor (" + String(SIZE) + "x" + String(SIZE) + "):")
    for i in range(SIZE):
        print("  [", end="")
        for j in range(SIZE):
            print(output[i * SIZE + j], end="")
            if j < SIZE - 1:
                print(", ", end="")
        print("]")

    print("\nExpected output:")
    print("  [0, 1]")
    print("  [1, 2]")

    # Verify results
    var expected = UnsafePointer[Float32].alloc(SIZE * SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            expected[i * SIZE + j] = Float32(i + j)

    var correct = True
    for i in range(SIZE * SIZE):
        if abs(output[i] - expected[i]) > 0.001:
            correct = False
            break

    print("\nVerification:", "PASSED" if correct else "FAILED")

    # GPU Programming Concepts Summary
    print("\nGPU Programming Key Concepts:")
    print("1. Thread Indexing:")
    print("   - thread_idx.x, thread_idx.y: Thread position within block")
    print("   - block_idx.x, block_idx.y: Block position within grid")
    print("   - block_dim.x, block_dim.y: Block dimensions")

    print("\n2. Memory Access:")
    print("   - Coalesced access: Adjacent threads access adjacent memory")
    print("   - Broadcasting: Efficient reuse of data across threads")
    print("   - Global memory: Accessible by all threads")

    print("\n3. Execution Model:")
    print("   - SIMT: Single Instruction, Multiple Threads")
    print("   - Warps: Groups of 32 threads execute together")
    print("   - Synchronization: __syncthreads() for block-level sync")

    print("\n4. Automatic Differentiation Applications:")
    print("   - Element-wise operations: Perfect for GPU parallelization")
    print("   - Matrix operations: Utilize shared memory and tiling")
    print("   - Gradient computation: Parallel backward pass")
    print("   - Parameter updates: Vectorized optimizer steps")

    # Cleanup
    output.free()
    a.free()
    b.free()
    expected.free()

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Complete GPU programming demonstration."""
    simulate_broadcast_add_complete()
```

### Expected Output for `20_gpu_programming_complete.mojo`

```
=== Complete GPU Programming Demonstration ===
GPU Configuration:
  Grid dimensions: (1, 1, 1)
  Block dimensions: (3, 3, 1)
  Total threads: 9

Input tensors:
  a (1x2 broadcasted): [0, 1]
  b (2x1 broadcasted): [0, 1]

Simulating GPU kernel execution:
  Kernel: broadcast_add(output, a, b)
  Operation: output[row][col] = a[col] + b[row]
    Thread(0,0): a[0] + b[0] = 0 + 0 = 0
    Thread(0,1): a[1] + b[0] = 1 + 0 = 1
    Thread(1,0): a[0] + b[1] = 0 + 1 = 1
    Thread(1,1): a[1] + b[1] = 1 + 1 = 2

Output tensor (2x2):
  [0, 1]
  [1, 2]

Expected output:
  [0, 1]
  [1, 2]

Verification: PASSED

GPU Programming Key Concepts:
1. Thread Indexing:
   - thread_idx.x, thread_idx.y: Thread position within block
   - block_idx.x, block_idx.y: Block position within grid
   - block_dim.x, block_dim.y: Block dimensions

2. Memory Access:
   - Coalesced access: Adjacent threads access adjacent memory
   - Broadcasting: Efficient reuse of data across threads
   - Global memory: Accessible by all threads

3. Execution Model:
   - SIMT: Single Instruction, Multiple Threads
   - Warps: Groups of 32 threads execute together
   - Synchronization: __syncthreads() for block-level sync

4. Automatic Differentiation Applications:
   - Element-wise operations: Perfect for GPU parallelization
   - Matrix operations: Utilize shared memory and tiling
   - Gradient computation: Parallel backward pass
   - Parameter updates: Vectorized optimizer steps
```

This is the exact `Thread(0,0)…Thread(1,1)` trace worked by hand in Worked Example 4.5.1, run to completion and self-verified against the `expected[row][col] = row + col` formula.

## Chapter Summary

A GPU thread's identity is really two numbers — its position within its block (`thread_idx`) and its block's position within the grid (`block_idx`) — combined into one global index that alone maps to a specific piece of data; nothing about `thread_idx` in isolation is unique across a whole grid. The host (CPU) and device (GPU) split responsibilities cleanly: the host allocates, copies, and launches; the device runs one kernel body as thousands of simultaneous, independent copies, each seeing its own thread and block identity. Three memory scopes — global (slow, universally visible), shared (fast, visible within one block, temporary), and local (fastest, visible to one thread alone) — trade reach for speed, in that order. Memory coalescing is Chapter 3's bus-utilization argument again, now at the scale of a 32-thread warp instead of a 4-lane SIMD register: adjacent threads touching adjacent memory cost one transaction, while scattered access can cost up to 32, for identical useful data — which is exactly why Chapter 3's SoA recommendation reappears here verbatim. Broadcasting and matrix multiplication both map naturally onto "one thread per output element" — broadcasting needing no per-thread loop at all, matrix multiplication needing a short dot-product loop per thread — and both were traced completely by hand on small examples this chapter, landing on the same `[[0,1],[1,2]]` and `[[19,22],[43,50]]` a real kernel launch would produce.

## Self-Check Questions

1. Block 5 contains 16 threads per block. What is the global ID of thread 9 in block 5, and why is "thread 9" alone not enough information to answer that question?
2. Explain, in terms of who runs what, why a GPU program is really "two programs cooperating" rather than one program that happens to mention a kernel.
3. A kernel writes a value to shared memory during block 3's execution. Can block 7, running concurrently, read that value? Why or why not?
4. Four threads in a warp-sized group access `data[0], data[8], data[16], data[24]`, with each 16-byte chunk holding 4 floats (as in Worked Example 4.4.1). How many transactions does this access pattern need, and what is its bus utilization?
5. Both the broadcasting kernel (Section 4.5) and the matrix-multiplication kernel (Section 4.6) assign one thread per output element. What is the key difference in how much work each individual thread does, and why does that difference not change the fact that neither kernel needs any inter-thread synchronization?

## Where We Go Next

Chapter 5 returns to SIMD — the CPU-scale version of "many workers doing the same thing at once" this chapter's GPU threads scaled up from — and looks specifically at vectorization: how a single Mojo loop can be written once and compiled down to the native SIMD width of whatever CPU it eventually runs on, closing the loop back to Chapter 1.4's very first `SIMD[DType.float32, 4]` example.

## Worked Solutions

**1.** `global_id = block_idx * block_dim + thread_idx = 5 × 16 + 9 = 89`. "Thread 9" alone is ambiguous because `thread_idx` resets to `0` at the start of every block — block 0's thread 9, block 5's thread 9, and block 20's thread 9 are three entirely different workers with three entirely different global IDs, and only the combination with `block_idx` disambiguates them.

**2.** The host (CPU) and device (GPU) are, functionally, two separate processors with two separate memory spaces, each running its own instruction stream: the host runs ordinary sequential code to allocate memory, copy data across, and decide on a launch configuration, while the device runs a completely different program — the kernel — simultaneously across thousands of threads. Neither one can do the other's job: the host cannot itself execute the kernel's massively parallel body, and the device's kernel threads have no way to allocate memory, decide on a grid configuration, or copy data on their own initiative — that setup has to happen first, on the host.

**3.** No. Shared memory's scope is exactly one block's execution — every thread within block 3 can see what block 3 writes there, but block 7 is a separate block with its own separate shared-memory allocation, entirely invisible to block 3 and vice versa. The only memory scope visible to *both* block 3 and block 7 is global memory.

**4.** With 16-byte chunks holding 4 floats each, `data[0]` falls in chunk 0, `data[8]` in chunk 2, `data[16]` in chunk 4, `data[24]` in chunk 6 — four distinct chunks, so four separate transactions are needed. Total bytes moved: `4 × 16 = 64`. Bytes actually used: `4 × 4 = 16` (one float per fetched chunk). Utilization: `16/64 = 25%` — structurally identical to Worked Example 4.4.1's strided case, just with a larger stride.

**5.** The broadcasting kernel's per-thread work is a single addition (`a[col] + b[row]`) — no loop at all. The matrix-multiplication kernel's per-thread work is a loop over `k`, accumulating a full dot product. Despite that difference in workload size, neither kernel needs synchronization between threads, because both are still cases where every thread's entire computation depends only on its own inputs (`a[col]`/`b[row]` for broadcasting; one row of `A` and one column of `B` for matrix multiplication) and writes only to its own output cell — no thread ever needs to wait for, or read a result from, any other thread.
