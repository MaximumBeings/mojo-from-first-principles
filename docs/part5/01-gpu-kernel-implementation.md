# Chapter 9: GPU Kernel Implementation

Part 0.4 introduced the GPU execution model in the abstract — grids, blocks, threads, and the memory hierarchy. Part 5 puts real workloads on it: the CUDA-style kernels that back every tensor operation from Part 2, written to actually earn the speedup the hardware promises rather than merely compile for it.

## 9.1 CUDA-style Kernel Design

The thread hierarchy from Part 0.4 maps directly onto every kernel in this framework: a **grid** of **blocks**, each block a fixed number of **threads**, and every thread computing exactly one output element (or a small tile of them, in the more advanced kernels below).

```mojo
# GPU Programming Concepts, concretely:
#   Grid:   collection of thread blocks
#   Block:  collection of threads (up to 1024)
#   Thread: one execution unit
#
#   Global memory:  large, slow, visible to every thread
#   Shared memory:  fast, scoped to one block
#   Local memory:   private per-thread
#
#   Kernel: the function that runs on the GPU
#   Host:   the CPU code that launches it
#   Device: the GPU executing it

from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

fn generic_elementwise_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    """The template every Chapter 3 kernel follows: compute a global
    thread index, bounds-check it, do one unit of work."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * Float32(2.0)
```

The launch configuration is the other half of "CUDA-style": pick a block size (256 is a common default — large enough to hide memory latency, small enough that many blocks fit per streaming multiprocessor), then compute how many blocks cover the whole tensor. Work it out for a tensor of exactly `1,000,000` elements: `num_blocks = (1,000,000 + 255) // 256 = 1,000,255 // 256 = 3907` (integer division truncates). Those 3907 blocks of 256 threads each launch `3907 × 256 = 1,000,192` threads in total — 192 more than the tensor actually has. The last block, block `3906`, covers global indices `999,936` through `1,000,191`; of its 256 threads, only the first 64 satisfy `idx < size` and do real work, and the bounds check (`if idx < size`) is what stops the remaining 192 threads in that one block from writing past the end of the buffer. This "round up, then let the bounds check absorb the remainder" pattern is why every kernel in this book pairs `(size + N - 1) // N` block counts with an `if idx < size` guard inside the kernel body — the ceiling division without the guard would corrupt memory; the guard without the ceiling division would silently skip the tail elements.

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

## 9.2 Memory Coalescing Optimization

A warp of GPU threads reading 32 consecutive `Float32`s in one transaction is dramatically faster than the same warp reading 32 scattered addresses — this is memory coalescing, and it is the single biggest reason Part 1's tensor layout decisions (Struct-of-Arrays over Array-of-Structs, from Part 0.3) matter for performance rather than just style.

```mojo
# Array of Structs (AoS) -- poor coalescing
#   Memory: [face,maturity,rate,spread][face,maturity,rate,spread]...
#   A kernel reading every bond's `rate` jumps 4 fields between reads.
#
# Struct of Arrays (SoA) -- optimal coalescing
#   Memory: face:[f0,f1,f2,...]  maturity:[m0,m1,m2,...]  rate:[r0,r1,...]
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

This is not a hypothetical: it's the exact struct that prices a bond portfolio in [Part 7](../part7/01-quantitative-finance-examples.md), where switching from an AoS `Bond` struct to `ZeroCouponBondSystemSoA` is measured directly by comparing memory stride — 32 bytes between the same field in adjacent AoS records versus 4 bytes in SoA, an 8× reduction:

```
Memory layout comparison:
  AoS stride between same attributes: 32 bytes
  SoA stride between adjacent elements: 4 bytes
  Coalescing improvement: ~8x better
```

## 9.3 Shared Memory Utilization

Global memory bandwidth is the bottleneck for most of the kernels above, but any kernel where multiple threads *reuse* the same input values — a 2D convolution being the clearest example — benefits from staging that input into a block's shared memory once, rather than every thread re-reading it from global memory:

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

As written, neighboring output threads redundantly re-read overlapping input patches straight from global memory — a `3×3` kernel means each input element is read up to 9 times. The shared-memory version each block cooperatively loads once — every thread copies one element of its block's input tile (plus a kernel-sized halo of neighbors) into a `__shared__`-equivalent buffer, synchronizes, then every thread's convolution reads exclusively from that fast on-chip memory instead of re-hitting global memory nine times over.

Padding the input before convolving (rather than shrinking the output, as the unpadded kernel above does) trades a small amount of extra memory for output that matches the input's spatial size — the standard "same" convolution used throughout Part 6's neural network layers:

```mojo
fn mojo_conv2d_padded(input: GPUTensor, kernel: GPUTensor, padding: Int) -> GPUTensor:
    # Pad input on all sides by `padding` zeros, then run the same
    # kernel loop as above against the padded buffer -- output_h/w
    # now equal input_h/w instead of shrinking by (kernel_size - 1).
    var padded = pad_tensor(input, padding)
    return mojo_conv2d_basic(padded, kernel)
```

## 9.4 Warp-level Primitives

The reduction kernels from [Chapter 5](../part2/03-reduction-operations.md) are written generically over thread count, but on real hardware the last few rounds of a tree reduction — once the active thread count drops to 32 or fewer — execute within a single warp, where every thread runs in lockstep. Warp-level shuffle instructions let those final rounds exchange values directly between registers instead of round-tripping through shared or global memory, collapsing the last `log2(32) = 5` reduction rounds into a handful of register-to-register operations. The framework's `sum_reduce_kernel` is written the portable way (through memory, at every level) specifically so it's correct on any GPU generation; a warp-shuffle specialization is the natural next optimization once profiling (Chapter 10) shows the reduction tail is actually a bottleneck.

Chapter 10 turns from kernel *design* to kernel *measurement* — SIMD vectorization on the CPU side, loop fusion, compile-time specialization, and the benchmarking harness used to tell whether any of these optimizations actually helped.
