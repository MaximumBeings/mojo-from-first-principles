# Chapter 9: GPU Kernel Implementation

GPU acceleration is not “run the CPU loop somewhere else.” A host allocates device buffers and launches a grid; thousands of device threads each compute a global index and own a small slice of the work. The examples in this chapter target Mojo 1.0's portable `std.gpu` API rather than CUDA-only names or an approximate wrapper.

## 9.1 One thread, one element

The safest first kernel maps one global thread index to one output element and guards the tail. Mojo 1.0 forbids host-sized `Int` and `UInt` as GPU kernel arguments, so lengths crossing the host/device boundary use a fixed-width integer.

```mojo
from std.gpu import global_idx

def scale_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    scale: Float32,
    size: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(size):
        output[i] = input[i] * scale
```

**Manual worked example.** For five inputs `[1,2,3,4,5]`, scale 2, and a block of eight threads, global indices 0–4 write `[2,4,6,8,10]`. Threads 5–7 fail `i<5` and write nothing. Without the guard, those three threads would access memory past the allocation.

## 9.2 Launch geometry covers the tail

The host uses `DeviceContext`, allocates a `DeviceBuffer`, and launches with the current single-function form `enqueue_function[kernel]`. Ceiling division chooses enough blocks to cover every element.

```mojo
from std.gpu.host import DeviceContext

def launch_scale(ctx: DeviceContext, output, input, scale: Float32, size: Int32) raises:
    comptime block_size = 256
    var grid_size = (Int(size) + block_size - 1) // block_size
    ctx.enqueue_function[scale_kernel](
        output, input, scale, size,
        grid_dim=grid_size,
        block_dim=block_size,
    )
```

**Manual worked example.** With `size=1,000` and `block_size=256`, `(1000+255)//256=4` blocks launch 1,024 threads. The first 1,000 process data and the last 24 take the guarded exit. Three blocks would launch only 768 threads and leave 232 outputs untouched.

## 9.3 Coalescing is a layout property

Adjacent threads should access adjacent addresses. A structure-of-arrays layout makes thread `i` and thread `i+1` read neighboring rates, whereas an array of bond structs inserts the other fields between those reads.

```mojo
from std.gpu import global_idx

def discount_kernel(
    present_value: UnsafePointer[Float32],
    face: UnsafePointer[Float32],
    rate: UnsafePointer[Float32],
    years: UnsafePointer[Float32],
    size: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(size):
        present_value[i] = face[i] * exp(-rate[i] * years[i])
```

**Manual worked example.** Threads 0–3 read `rate[0]` through `rate[3]`, four consecutive `Float32` values occupying 16 consecutive bytes. For faces `[100,100]`, rates `[0.05,0.06]`, and years `[1,2]`, the writes are `100e^-0.05≈95.1229` and `100e^-0.12≈88.6920`.

## 9.4 Shared memory and barriers require block-wide reasoning

Shared memory is useful when threads in one block reuse a tile. Every participating thread writes its element, all threads reach `barrier()`, and only then may any thread consume neighbors' elements. Mojo's higher-level `TileTensor` is preferred for production tiling because it carries layout information and composes with device buffers; raw shared-memory code is reserved for kernels whose access pattern cannot be expressed clearly at the tile level.

```mojo
from std.gpu.sync import barrier

def two_phase_block_step(mut tile, local_i: Int, value: Float32) -> Float32:
    tile[local_i] = value
    barrier()
    return tile[local_i] + tile[local_i + 1]
```

**Manual worked example.** Suppose four threads write `[2,4,6,8]` into a tile and threads 0–2 then add a right neighbor. After the barrier their results are `[2+4,4+6,6+8]=[6,10,14]`. Without the barrier, thread 0 could read slot 1 before thread 1 writes 4, making the result schedule-dependent. A real kernel must also ensure the final thread does not read past the tile.

## 9.5 Warp primitives are an optimization, not a starting point

Warp shuffles and reductions can remove shared-memory traffic, but warp size and supported primitives vary by accelerator. Establish correctness with a portable block reduction, benchmark it, and specialize only the measured bottleneck behind a tested dispatch path.

```mojo
def reduction_round(mut values: List[Float32], active: Int):
    var half = active // 2
    for i in range(half):
        values[i] += values[i + half]
```

**Manual worked example.** Start with `[1,2,3,4,5,6,7,8]`. The first round adds opposite halves and leaves `[6,8,10,12]`; the next leaves `[16,20]`; the final leaves `[36]`. Since `1+2+...+8=36`, the tree is correct before any warp-specific rewrite is attempted.

The production sequence is now explicit: prove the scalar result, map work to guarded global indices, use graph-owned device buffers, verify host/device copies, then optimize layout, tiling, and warp communication in that order.
