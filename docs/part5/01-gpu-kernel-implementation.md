# Chapter 18: GPU Kernel Implementation — Kernels That Earn What Chapter 4 Promised

> "Chapter 4 built the model: a thread's identity is two numbers, memory has three scopes, and a warp either reads together or pays for reading separately. This chapter is what happens when that model meets a kernel with a job to actually finish quickly — the launch math that doesn't waste a thread, the layout that doesn't waste a transaction, the on-chip memory that doesn't make ten threads pay for the same value ten times over, and the one instruction that skips memory altogether for the last few steps of a reduction."

**What you will understand by the end of this chapter:**

- How to compute a real launch configuration by ceiling division — block count, total threads launched, and exactly how many of them are wasted — and why the in-kernel bounds check isn't defensive boilerplate but the other half of a single design decision, traced on a tensor of a million elements
- Memory coalescing translated from Chapter 4's abstract warp argument into a concrete win on a real struct: reading one field across many bond records, counted in actual transactions and actual bandwidth utilization, for both the Array-of-Structs and Struct-of-Arrays layouts
- Why a naive convolution kernel reads most of its input many times over, counted exactly rather than asserted, and how staging one block's input tile into shared memory — with more threads doing the loading than will ever compute an output — removes that redundancy entirely
- Why padding the input (rather than shrinking the output) is what a "same"-shaped convolution actually requires, traced by hand at a border position and an interior position of the same small example
- Warp-level shuffle instructions as the natural endpoint of a tree reduction once the active thread count drops to one warp — and the one precondition (every lane in the warp must participate) that this chapter's own Section 18.1 bounds-check pattern can silently violate

**What you need to know first:**

- Chapter 4 in full: the thread hierarchy and global-index formula (4.1), the host/device split (4.2), the three memory scopes (4.3), and memory coalescing's warp-transaction argument (4.4). This chapter takes each of those ideas into real launch configurations and real kernel code.
- Chapter 3 (Struct-of-Arrays vs Array-of-Structs) — Section 18.2 reuses its argument directly on a financial data struct.
- Chapter 14.1 (`tensor_sum`'s tree reduction) — Section 18.4 is precisely the optimization that shortens the last few rounds of that same reduction.
- Chapter 13.1 (matrix multiplication's index-summation form) — useful background for how shared-memory tiling generalizes past convolution to any operation where neighboring threads reread the same input.

## 18.1 CUDA-Style Kernel Design `[FOUNDATIONAL]`

### Intuition

A moving company only sends out crews in fixed sizes — say, four movers to a truck — never a lone mover, no matter how small the job. If a job needs eleven boxes moved, the company can't send "two and three quarters" crews; it sends three full crews, twelve movers, and the twelfth mover simply has nothing to do when they arrive. The company's actual obligation isn't to send exactly the right number of movers — it's to send *enough* crews to cover the job, and to make sure the extra mover on the last truck knows to stand aside rather than grab a box that was never there. A GPU kernel launch works exactly the same way: threads only come in fixed-size blocks, so covering `N` elements almost never divides evenly, and the kernel itself has to be the one that tells the leftover threads to stand aside.

### Background

Every kernel in this book follows the same two-part template Chapter 4 introduced in the abstract: compute a thread's global index, then guard it.

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

fn generic_elementwise_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    """The template every kernel in this book follows: compute a global
    thread index, bounds-check it, do one unit of work."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * Float32(2.0)
```

The launch configuration is the other half of the decision, and it's made entirely on the host, before a single thread runs:

```mojo
fn launch_elementwise(ctx: DeviceContext, output: UnsafePointer[Scalar[DType.float32]],
                       input: UnsafePointer[Scalar[DType.float32]], size: Int) raises:
    alias THREADS_PER_BLOCK = 256
    var num_blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    ctx.enqueue_function[generic_elementwise_kernel](
        output, input, size,
        grid_dim=(num_blocks, 1, 1),
        block_dim=(THREADS_PER_BLOCK, 1, 1),
    )
    ctx.synchronize()
```

`num_blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK` is *ceiling* division computed with integer arithmetic — the `+ THREADS_PER_BLOCK - 1` before the floor-dividing `//` is what rounds up instead of down. Rounding up guarantees every element gets covered by some block; the `if idx < size` guard back in the kernel is what stops the inevitable overshoot from writing past the end of the buffer. Neither piece is optional, and each protects against a different failure:

| Ceiling division | Bounds check | Result |
|---|---|---|
| No (`size // N`) | No | Tail elements past the last full block are never processed at all |
| No (`size // N`) | Yes | Still misses the tail — a guard can't rescue elements whose block was never launched |
| Yes (`(size+N-1)//N`) | No | Overshoot threads write past the buffer's end — memory corruption |
| Yes (`(size+N-1)//N`) | Yes | Full coverage, no corruption — the pattern every kernel in this book uses |

### Worked Example 18.1.1 — A million elements, traced to the exact wasted thread

For a tensor of exactly `1,000,000` elements at `THREADS_PER_BLOCK = 256`: `num_blocks = (1,000,000 + 255) // 256 = 1,000,255 // 256 = 3907` (integer division truncates the remainder). Those `3907` blocks launch `3907 × 256 = 1,000,192` threads in total — `192` more than the tensor has elements. The last block, block `3906`, covers global indices `999,936` through `1,000,191`; of its `256` threads, only the first `64` satisfy `idx < 1,000,000` and do real work — indices `999,936` through `999,999`. The remaining `192` threads in that one block, and only that one block, are the ones the bounds check exists to stop.

### Worked Example 18.1.2 — A different size and block width, to check the pattern generalizes

Same formula, different numbers: `size = 10,000`, `THREADS_PER_BLOCK = 128`. `num_blocks = (10,000 + 127) // 128 = 10,127 // 128 = 79`. Total threads launched: `79 × 128 = 10,112` — `112` wasted. The last block is block `78`, covering global indices `78 × 128 = 9,984` through `10,111`. Valid indices are `9,984` through `9,999` — exactly `16` active threads in that block, and the remaining `128 - 16 = 112` idle threads account for the entire wasted count computed above. As in the million-element case, every wasted thread lives in exactly one block — the last one — never scattered across the launch.

```
[COMMON TRAP]  Removing "wasteful" rounding by shrinking block count

It is tempting to compute num_blocks = size // THREADS_PER_BLOCK
instead -- after all, why launch a block that is mostly idle? The
answer is in the table above: floor division doesn't shrink the idle
thread count, it deletes real work. For size=10,000 and
THREADS_PER_BLOCK=128, size // 128 = 78 (not 79) -- exactly one block
short. Elements 9,984 through 9,999, the sixteen elements this
chapter's own worked example just traced as "active," would never be
covered by any launched block at all, and no bounds check inside the
kernel can process an index that no thread was ever assigned. The
"wasted" threads on the last block are not a bug to engineer away --
they are the cost of guaranteeing full coverage with fixed-size
blocks, paid once per launch, and the bounds check is what makes that
cost safe to pay.
```

## 18.2 Memory Coalescing Optimization `[FOUNDATIONAL]`

### Intuition

A records clerk is asked to look up one field — say, the interest rate — for four different customer files. If all four rates happen to be written on one shared summary sheet, one glance retrieves all four. If instead each rate is buried on page three of four separate, fully-detailed customer folders, the clerk pays for four separate trips to the filing cabinet to retrieve the exact same four numbers. Struct-of-Arrays is the summary sheet; Array-of-Structs is the stack of folders — and a GPU warp reading a field across many records is exactly this clerk, dozens of times over, every single kernel launch.

### Background

```mojo
# Array of Structs (AoS) -- poor coalescing
#   Memory: [face,maturity,rate,spread,...][face,maturity,rate,spread,...]...
#   A kernel reading every bond's `rate` jumps past every OTHER field between reads.
#
# Struct of Arrays (SoA) -- optimal coalescing
#   Memory: face:[f0,f1,f2,...]  maturity:[m0,m1,...]  rate:[r0,r1,...]
#   A kernel reading every bond's `rate` reads one contiguous array.

fn compute_from_soa_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    rate: UnsafePointer[Scalar[DType.float32]],      # contiguous: coalesced read
    spread: UnsafePointer[Scalar[DType.float32]],    # contiguous: coalesced read
    num_items: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_items:
        output[idx] = rate[idx] + spread[idx]
```

This is not a hypothetical: it is the exact struct that prices a bond portfolio in Part 7, where the SoA `ZeroCouponBondSystemSoA` holds eight `Float32` fields (`face_value`, `time_to_maturity`, `risk_free_rate`, `credit_spread`, `present_value`, `yield_to_maturity`, `duration`, `portfolio_weight`) as eight separate contiguous arrays instead of one interleaved struct per bond. An Array-of-Structs record holding those same eight fields is `8 × 4 = 32` bytes wide; the stride between the *same* field in two adjacent AoS records is that entire `32`-byte record, while the stride between adjacent elements of one SoA array is just `4` bytes — an `8×` difference, not a rough estimate:

```
Memory layout comparison:
  AoS stride between same field, adjacent records: 32 bytes
  SoA stride between adjacent elements, one field:  4 bytes
  Coalescing improvement:                          8x better
```

### Worked Example 18.2.1 — Counting actual transactions, AoS vs. SoA

Scale a 32-thread warp down to `4` threads (the same scaling trick Chapter 4.4 used), and treat every `16` contiguous bytes (`4` `Float32`s) as one memory transaction. Four threads read `risk_free_rate` for bond records `0` through `3`.

**SoA**: `risk_free_rate[0], risk_free_rate[1], risk_free_rate[2], risk_free_rate[3]` sit contiguously at float-indices `0, 1, 2, 3` — all inside the same `16`-byte chunk.

```
transactions needed = 1
bytes moved         = 16
bytes used           = 16
utilization         = 100%
```

**AoS**, with the field order above (`risk_free_rate` is the third of eight fields, float-index `2` within each `8`-float record): record `0`'s `risk_free_rate` is at float-index `2`, record `1`'s is at `8 + 2 = 10`, record `2`'s is at `16 + 2 = 18`, record `3`'s is at `24 + 2 = 26`. Dividing each by `4` to find its `16`-byte chunk: `2 → chunk 0`, `10 → chunk 2`, `18 → chunk 4`, `26 → chunk 6` — four distinct chunks, one per thread.

```
transactions needed = 4
bytes moved         = 4 x 16 = 64
bytes used           = 4 x 4  = 16
utilization         = 16/64 = 25%
```

Four threads, four numbers, identical useful data — `1` transaction against `4`, precisely the structural shape of Chapter 4.4's own strided-access case, just instantiated on a real eight-field financial record instead of an abstract array.

```
[COMMON TRAP]  Assuming SoA collapses every field access into one transaction

SoA guarantees that reading ONE field across many records is coalesced
-- it says nothing about how many transactions a kernel that needs
SEVERAL fields per thread will issue. compute_bond_prices_kernel
(Part 7) reads risk_free_rate, credit_spread, face_value, and
time_to_maturity for every bond it prices -- four separate SoA arrays,
so four separate (individually coalesced) transactions per warp, not
one. SoA turns each of those four reads from a scattered, 32-byte-
stride mess into a single clean transaction; it does not, and cannot,
merge four logically different fields into one physical read.
```

## 18.3 Shared Memory Utilization `[FOUNDATIONAL]`

### Intuition

Picture four apprentices in one workshop bay, each assigned to finish one tile of a mosaic, where every tile needs paint from a shared can sitting on a shelf across the room. If each apprentice fetches their own paint every time they need a dab, the same can gets carried across the room far more times than necessary — especially since neighboring tiles need overlapping colors. A foreman who instead sends a *few* apprentices to bring the whole can to the workbench once, share it there, and only then start painting saves every trip after the first. Shared memory is that workbench: on-chip, visible to every thread in one block, and loaded exactly once per block instead of once per thread.

### Background

The naive convolution kernel each thread runs independently, re-reading global memory for every multiply:

```mojo
@gpu_kernel
fn mojo_conv2d_kernel(
    input: DTypePointer[DType.float32], input_h: Int, input_w: Int,
    kernel: DTypePointer[DType.float32], kernel_h: Int, kernel_w: Int,
    output: DTypePointer[DType.float32], output_h: Int, output_w: Int,
):
    var out_row = block_idx().y * block_dim().y + thread_idx().y
    var out_col = block_idx().x * block_dim().x + thread_idx().x
    if out_row >= output_h or out_col >= output_w:
        return

    var sum: Float32 = 0.0
    for k_r in range(kernel_h):
        for k_c in range(kernel_w):
            var in_row = out_row + k_r
            var in_col = out_col + k_c
            sum += input[in_row * input_w + in_col] * kernel[k_r * kernel_w + k_c]
    output[out_row * output_w + out_col] = sum
```

Neighboring output threads' input windows overlap heavily — a `3×3` kernel means most interior input positions are read by up to nine different output threads, all pulling from global memory independently. The shared-memory fix stages the block's *entire* input footprint into on-chip memory once, using *more* threads than the block will ever use to compute outputs, because the extra input rows and columns at the tile's edges (the "halo") still need loading even though no thread centered there will ever own an output:

```mojo
alias TILE = 16                          # output elements per block, per side
alias KERNEL_SIZE = 3                    # this kernel's fixed kernel width/height
alias SHARED_DIM = TILE + KERNEL_SIZE - 1  # input footprint one tile of outputs needs

@gpu_kernel
fn mojo_conv2d_tiled_kernel(
    input: DTypePointer[DType.float32], input_h: Int, input_w: Int,
    kernel: DTypePointer[DType.float32], kernel_h: Int, kernel_w: Int,
    output: DTypePointer[DType.float32], output_h: Int, output_w: Int,
):
    var tile = stack_allocation[SHARED_DIM * SHARED_DIM, DType.float32, address_space = AddressSpace.SHARED]()

    # Every thread in the (oversized) block loads exactly one tile cell --
    # including the threads that will never compute an output.
    var in_row = block_idx().y * TILE + thread_idx().y
    var in_col = block_idx().x * TILE + thread_idx().x
    if in_row < input_h and in_col < input_w:
        tile[thread_idx().y * SHARED_DIM + thread_idx().x] = input[in_row * input_w + in_col]
    else:
        tile[thread_idx().y * SHARED_DIM + thread_idx().x] = 0.0

    barrier()   # every thread in the block must finish writing before ANY thread reads

    # Only the TILE x TILE "interior" threads own an output element --
    # the rest existed purely to help fill the tile's halo.
    if thread_idx().y >= TILE or thread_idx().x >= TILE:
        return
    var out_row = block_idx().y * TILE + thread_idx().y
    var out_col = block_idx().x * TILE + thread_idx().x
    if out_row >= output_h or out_col >= output_w:
        return

    var sum: Float32 = 0.0
    for k_r in range(kernel_h):
        for k_c in range(kernel_w):
            sum += tile[(thread_idx().y + k_r) * SHARED_DIM + (thread_idx().x + k_c)] * kernel[k_r * kernel_w + k_c]
    output[out_row * output_w + out_col] = sum
```

Every value this kernel needs is read from global memory exactly once, by whichever thread happens to own that tile position during the load phase; every subsequent read, during the actual convolution loop, comes from `tile` — on-chip, block-scoped, and (per Chapter 4.3) dramatically faster than global memory.

### Worked Example 18.3.1 — The naive kernel's redundant reads, counted exactly

Reuse a small, complete example: a `4×4` input filled sequentially and a `3×3` kernel of alternating `1`s and `0`s, the exact values used in the author's own convolution reference notes.

```
Input (4x4):            Kernel (3x3):
 1  2  3  4               1 0 1
 5  6  7  8               0 1 0
 9 10 11 12               1 0 1
13 14 15 16
```

With no padding, output is `2×2`: `output(0,0)` sums over input rows `0`-`2`, cols `0`-`2`: `1·1 + 2·0 + 3·1 + 5·0 + 6·1 + 7·0 + 9·1 + 10·0 + 11·1 = 1+3+6+9+11 = 30`. The same arithmetic on the other three windows gives `output(0,1)=35`, `output(1,0)=50`, `output(1,1)=55`:

```
output = 30  35
         50  55
```

Now count how many of the naive kernel's four output threads read each input cell. Interior cells `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` are each read by all four output threads — `4` times each; edge cells are read twice; corner cells once. Summing every cell's read count gives `36` total global-memory reads for `16` unique input values — exactly `4` outputs `× 9` kernel taps each, confirming the count independently. A larger image pushes interior cells toward the full `kernel_h × kernel_w = 9` reads this `4×4` example is too small to reach on its own, but the mechanism is identical at any size.

### Worked Example 18.3.2 — The same computation, staged through shared memory

For this small example, one block (`TILE=2`, `KERNEL_SIZE=3`, so `SHARED_DIM = 2+3-1 = 4`) covers the *entire* `4×4` input in a single tile. `SHARED_DIM × SHARED_DIM = 16` threads launch; each loads exactly one input cell into `tile`, so all `16` unique values are read from global memory exactly once, in total, for the whole block — not `36` times. After `barrier()`, only the `TILE × TILE = 4` threads with `thread_idx().y < 2` and `thread_idx().x < 2` proceed to compute an output, each reading its `3×3` window entirely from the now-fully-populated `tile` buffer. The four output values recovered are identical — `30, 35, 50, 55` — because `tile` holds exactly the same numbers the naive kernel read directly from `input`; only the number of *global* memory transactions changed, from `36` down to `16`.

### Worked Example 18.3.3 — Padding trades output size for border zeros

Padding by `1` on every side turns this same `4×4` input into an effective `6×6` buffer with a zero border, producing a `4×4 + 2·1 - 3 + 1 = 4×4` output — matching the input's own size, the "same"-convolution behavior Part 6's neural network layers rely on. At the top-right corner, `output(0,3)`: the kernel's window would need columns `2` through `4`, but column `4` doesn't exist, so that tap contributes `0`. Working through the remaining eight taps: `3·0 + 4·1 + 0·0` (row `0`, entirely off the top so contributing `0` regardless) `+ 7·1 + 8·0 + 0·0 = 4 + 7 = 11`. Compare this to an *interior* padded position: `output(1,1)`'s window, once the padding offset is accounted for, lands on exactly input rows `0`-`2` and columns `0`-`2` — the same window the unpadded `output(0,0)` used — so `output(1,1)` with padding equals `30`, the unpadded chapter's own first answer, shifted into a new position by the padding amount rather than recomputed from different numbers.

## 18.4 Warp-level Primitives `[FOUNDATIONAL]`

### Intuition

A relay team's first several handoffs happen across the length of a stadium, baton carried by runners who can't see each other. But the last few exchanges, once the race narrows to the final few runners bunched together at the finish line, don't need the stadium's length at all — the runners are close enough to pass the baton hand to hand, no track required. A GPU's tree reduction is the same shape: the first several rounds genuinely need shared or global memory to exchange partial sums across widely separated threads, but the final rounds, once the number of live threads drops to one warp (`32`), are all happening among threads close enough to exchange values directly through registers.

### Background

The reduction kernels from Chapter 14.1 are written generically over thread count — correct on any GPU generation, at the cost of routing every round's exchange through memory, even the last few rounds where only a handful of threads are still participating. Warp-level shuffle instructions let those final rounds skip memory entirely:

```mojo
from gpu.warp import shuffle_down

alias WARP_SIZE = 32

fn warp_reduce_sum(value: Float32) -> Float32:
    """Collapses the last WARP_SIZE elements of a tree reduction into
    log2(WARP_SIZE) register-to-register exchanges -- no shared or
    global memory involved for any of these rounds."""
    var v = value
    var offset = WARP_SIZE // 2
    while offset > 0:
        v += shuffle_down(v, offset)
        offset //= 2
    return v
```

`shuffle_down(v, offset)` reads the value a *different* lane in the same warp is holding in its own register — specifically, the lane `offset` positions ahead — without either lane ever writing to memory. Halving `offset` each round is exactly the tree-reduction pattern Chapter 14.1 already uses, just operating on registers instead of a shared array: `log2(32) = 5` rounds fully reduce a warp, versus `5` rounds of shared-memory reads, writes, and `barrier()` calls the portable version pays for.

### Worked Example 18.4.1 — An 8-lane warp, reduced by hand

Scale a 32-lane warp down to `8` lanes (the same scaling convention Chapter 4.4 used for 32-wide warps), holding values `[1, 2, 3, 4, 5, 6, 7, 8]` (sum `= 36`). `log2(8) = 3` rounds:

```
lane:    0    1    2    3    4    5    6    7
value:   1    2    3    4    5    6    7    8

round 1 (offset=4): lane_i += shuffle_down(lane_i, 4), for i=0..3
value:   6    8   10   12    .    .    .    .

round 2 (offset=2): lane_i += shuffle_down(lane_i, 2), for i=0..1
value:  16   20    .    .    .    .    .    .

round 3 (offset=1): lane_0 += shuffle_down(lane_0, 1)
value:  36    .    .    .    .    .    .    .
```

Lane `0` ends up holding `36` — the full sum — after exactly `3` register-to-register exchanges and zero memory operations, matching `log2(8)=3` exactly the way a real `32`-lane warp resolves in `log2(32)=5`.

```
[COMMON TRAP]  Section 18.1's bounds check meets Section 18.4's shuffle

shuffle_down requires every lane it names to actually be executing --
it reads a value from another lane's live register, not from some
persistent buffer. Section 18.1 established that a kernel's bounds
check (`if idx < size: return`) is exactly the mechanism that keeps an
overshoot launch safe. Combine the two carelessly and the bounds check
becomes the bug: if the last block in a launch has some lanes with
idx < size and others with idx >= size, and the idx >= size lanes
return early -- BEFORE the warp reaches a shuffle_down call the
remaining lanes still need to execute -- those returned lanes are no
longer live participants in the warp, and shuffling with them reads
undefined register content instead of the zero or identity value the
reduction needs. The fix is to let every lane in the warp reach the
shuffle (padding inactive lanes' local value with the reduction's
identity element -- 0 for a sum -- instead of returning early), not to
skip the bounds check that Section 18.1 established is otherwise
required.
```

## 18.5 Reference Implementations

None of this chapter's Mojo has been compiled or run — consistent with the rest of this book's newly-composed material, it is presented as source, not as a captured session. The one exception to "purely newly-composed" is Worked Examples 18.3.1 through 18.3.3: their `4×4` input and `3×3` kernel values are drawn directly from a real Mojo GPU convolution reference the author had on hand, though that reference's own code uses an older, incompatible Mojo dialect (`let`, `inout self`, a built-in `Tensor[DType.float32](rows, cols)` constructor) that doesn't match this book's `UnsafePointer`-based structs — so the numbers are real and independently reverified above, but the Mojo code presented is written fresh, in this book's own established idiom, rather than copied from that source. What follows is every kernel this chapter derived, assembled in one place:

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu.warp import shuffle_down
from memory import UnsafePointer

# ---- 18.1: launch configuration ----

fn generic_elementwise_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * Float32(2.0)

fn launch_elementwise(ctx: DeviceContext, output: UnsafePointer[Scalar[DType.float32]],
                       input: UnsafePointer[Scalar[DType.float32]], size: Int) raises:
    alias THREADS_PER_BLOCK = 256
    var num_blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    ctx.enqueue_function[generic_elementwise_kernel](
        output, input, size,
        grid_dim=(num_blocks, 1, 1),
        block_dim=(THREADS_PER_BLOCK, 1, 1),
    )
    ctx.synchronize()


# ---- 18.2: memory coalescing ----

fn compute_from_soa_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    rate: UnsafePointer[Scalar[DType.float32]],
    spread: UnsafePointer[Scalar[DType.float32]],
    num_items: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_items:
        output[idx] = rate[idx] + spread[idx]


# ---- 18.3: shared memory tiling ----

@gpu_kernel
fn mojo_conv2d_kernel(
    input: DTypePointer[DType.float32], input_h: Int, input_w: Int,
    kernel: DTypePointer[DType.float32], kernel_h: Int, kernel_w: Int,
    output: DTypePointer[DType.float32], output_h: Int, output_w: Int,
):
    var out_row = block_idx().y * block_dim().y + thread_idx().y
    var out_col = block_idx().x * block_dim().x + thread_idx().x
    if out_row >= output_h or out_col >= output_w:
        return
    var sum: Float32 = 0.0
    for k_r in range(kernel_h):
        for k_c in range(kernel_w):
            var in_row = out_row + k_r
            var in_col = out_col + k_c
            sum += input[in_row * input_w + in_col] * kernel[k_r * kernel_w + k_c]
    output[out_row * output_w + out_col] = sum

alias TILE = 16
alias KERNEL_SIZE = 3
alias SHARED_DIM = TILE + KERNEL_SIZE - 1

@gpu_kernel
fn mojo_conv2d_tiled_kernel(
    input: DTypePointer[DType.float32], input_h: Int, input_w: Int,
    kernel: DTypePointer[DType.float32], kernel_h: Int, kernel_w: Int,
    output: DTypePointer[DType.float32], output_h: Int, output_w: Int,
):
    var tile = stack_allocation[SHARED_DIM * SHARED_DIM, DType.float32, address_space = AddressSpace.SHARED]()
    var in_row = block_idx().y * TILE + thread_idx().y
    var in_col = block_idx().x * TILE + thread_idx().x
    if in_row < input_h and in_col < input_w:
        tile[thread_idx().y * SHARED_DIM + thread_idx().x] = input[in_row * input_w + in_col]
    else:
        tile[thread_idx().y * SHARED_DIM + thread_idx().x] = 0.0
    barrier()
    if thread_idx().y >= TILE or thread_idx().x >= TILE:
        return
    var out_row = block_idx().y * TILE + thread_idx().y
    var out_col = block_idx().x * TILE + thread_idx().x
    if out_row >= output_h or out_col >= output_w:
        return
    var sum: Float32 = 0.0
    for k_r in range(kernel_h):
        for k_c in range(kernel_w):
            sum += tile[(thread_idx().y + k_r) * SHARED_DIM + (thread_idx().x + k_c)] * kernel[k_r * kernel_w + k_c]
    output[out_row * output_w + out_col] = sum

fn mojo_conv2d_padded(input: GPUTensor, kernel: GPUTensor, padding: Int) -> GPUTensor:
    # Pad input on all sides by `padding` zeros, then run the tiled kernel
    # above against the padded buffer -- output_h/w now equal input_h/w
    # instead of shrinking by (kernel_size - 1).
    var padded = pad_tensor(input, padding)
    return mojo_conv2d_tiled(padded, kernel)


# ---- 18.4: warp-level reduction ----

alias WARP_SIZE = 32

fn warp_reduce_sum(value: Float32) -> Float32:
    var v = value
    var offset = WARP_SIZE // 2
    while offset > 0:
        v += shuffle_down(v, offset)
        offset //= 2
    return v
```

### Expected Output

There is no captured run to reproduce here — see the note above on why. Worked Examples 18.1.1 through 18.4.1 remain the source of truth for what each kernel computes on real numbers; this section is an index into them, assembled in one place the way a real project would actually organize the file, not a substitute for the worked-by-hand arithmetic above.

## Chapter Summary

A kernel launch's block count is ceiling division, not floor division — `(size + N - 1) // N` — and the reason is asymmetric: rounding down silently drops real elements no bounds check can rescue, while rounding up and pairing it with `if idx < size` inside the kernel wastes a bounded, known number of idle threads (`192` for this chapter's million-element example, `112` for its ten-thousand-element one) in exchange for guaranteed full coverage. Memory coalescing is Chapter 4.4's warp-bandwidth argument made concrete on a real eight-field financial record: reading one field across four bond records costs one transaction in Struct-of-Arrays layout and four in Array-of-Structs, a `4×` penalty on this chapter's scaled-down example that becomes the same `8×` byte-stride penalty Part 7's real portfolio pricing measures directly — though SoA only fixes *this*, not a kernel that genuinely needs several different fields per thread. A naive convolution kernel reads most of its input several times over — `36` total reads for `16` unique values in this chapter's own `4×4` example — purely because neighboring output threads' windows overlap; staging the block's entire input footprint into shared memory once, using more threads to load than will ever compute an output, cuts that down to exactly one global read per unique value, with a `barrier()` between the loading phase and the computing phase standing between "correct" and "a race condition." Padding trades a shrinking output for a fixed-size one by surrounding the input with zeros rather than narrowing the valid window, verified here at both a border position (`11`) and an interior one that exactly reproduces the unpadded chapter's own first answer (`30`), just shifted. Finally, warp-level shuffle instructions collapse the last `log2(32)=5` rounds of any tree reduction into register-to-register exchanges with no memory traffic at all — but they require every lane in the warp to still be executing, which makes Section 18.1's own bounds-check pattern, applied carelessly to a kernel that also uses shuffles, the exact mechanism that can break one.

## Self-Check Questions

1. For `size = 5,000` and `THREADS_PER_BLOCK = 512`, compute `num_blocks`, the total number of threads launched, the number wasted, and how many threads in the last block are actually active.
2. A warp-scaled group of `4` threads reads the `duration` field (the seventh of the eight `ZeroCouponBondSystemSoA` fields, float-index `6` within each `8`-float AoS record) for bond records `0` through `3`, using the same `16`-byte/`4`-float chunk convention as Worked Example 18.2.1. How many transactions does the AoS layout need, and what is its bus utilization?
3. A kernel loads its shared-memory tile and then begins its convolution loop immediately, without calling `barrier()` first. Some threads finish their load-and-compute before other threads in the same block have even finished loading. What specific kind of wrong value can this produce, and which threads are affected?
4. Using this chapter's `4×4` input and `3×3` kernel with `padding=1`, compute the padded output value at the top-right corner, `output(0,3)`.
5. A warp-level reduction kernel has `4` of its `32` lanes return early, before the `shuffle_down` calls, because a bounds check found `idx >= size` for those four threads. Why is this a problem, and which earlier section of this same chapter introduced the pattern now causing it?

## Where We Go Next

Chapter 19 (`part5/02-performance-optimization.md`) turns from kernel *design* to kernel *measurement*: SIMD vectorization on the CPU side, loop fusion, compile-time specialization, and the benchmarking harness used to tell whether any of this chapter's optimizations — ceiling-division launches, SoA layouts, shared-memory tiling, warp shuffles — actually helped, rather than simply looking like they should have.

## Worked Solutions

**1.** `num_blocks = (5,000 + 511) // 512 = 5,511 // 512 = 10`. Total threads launched: `10 × 512 = 5,120`. Wasted: `5,120 - 5,000 = 120`. The last block is block `9`, covering global indices `9 × 512 = 4,608` through `5,119`. Valid indices are `4,608` through `4,999` — `392` active threads — leaving `512 - 392 = 120` idle threads in that one block, exactly matching the wasted count computed from the totals.

**2.** Record `0`'s `duration` sits at float-index `0×8+6=6`; record `1`'s at `8+6=14`; record `2`'s at `16+6=22`; record `3`'s at `24+6=30`. Dividing by `4` to find each `16`-byte chunk: `6→chunk 1`, `14→chunk 3`, `22→chunk 5`, `30→chunk 7` — four distinct chunks, so `4` transactions are needed. Bytes moved: `4×16=64`; bytes used: `4×4=16`; utilization: `16/64=25%` — structurally identical to Worked Example 18.2.1's `risk_free_rate` case, confirming the AoS penalty isn't specific to any one field.

**3.** This produces a race condition: a thread that reaches the convolution loop before every thread assigned to load a value it needs has actually written that value reads whatever garbage or stale data happened to already be sitting in that shared-memory slot, not the input value that thread was supposed to load. The affected threads are unpredictable and can vary from run to run — specifically, any thread whose `3×3` window includes a tile cell that a *slower* sibling thread (one still mid-load, or not yet scheduled) hasn't written yet. `barrier()` exists precisely to rule this out, by making every thread in the block wait until all of them have finished the load phase before any of them is allowed to start reading.

**4.** `output(0,3)`'s window needs input columns `2` through `4` and rows `-1` through `1`. Row `-1` is entirely padding and contributes `0` regardless of kernel weights. For row `0`: `input[0,2]=3` × kernel `[1][0]=0` → `0`; `input[0,3]=4` × kernel `[1][1]=1` → `4`; column `4` is padding × kernel `[1][2]=0` → `0`. For row `1`: `input[1,2]=7` × kernel `[2][0]=1` → `7`; `input[1,3]=8` × kernel `[2][1]=0` → `0`; column `4` padding × kernel `[2][2]=1` → `0`. Total: `4 + 7 = 11`.

**5.** `shuffle_down` reads a value from another lane's live register in the same warp — it has no defined result when the lane it's reading from has already exited the kernel. The four early-returning lanes are no longer executing at all by the time the remaining lanes reach `shuffle_down`, so any exchange involving them reads undefined content instead of a meaningful partial sum, silently corrupting the reduction. The pattern responsible is Section 18.1's own bounds-check idiom (`if idx < size: return` or its early-exit variants) — entirely correct and necessary on its own, as Section 18.1 established, but unsafe to combine with a later warp shuffle unless every lane is guaranteed to still reach the shuffle call, typically by substituting the reduction's identity value (`0` for a sum) for out-of-range lanes instead of letting them return early.
