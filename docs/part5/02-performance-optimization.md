# Chapter 10: Performance Optimization Techniques

## 10.1 SIMD Vectorization

Part 0.5 introduced SIMD as a language feature; this section is where it becomes a systematic optimization applied everywhere the memory layout supports it. The rule of thumb this book follows: any loop whose body is the same scalar operation repeated over contiguous memory is a SIMD candidate. Trace what that means for a vector of `size = 10` elements on hardware whose natural SIMD width is `4` (four `Float32`s per vector register). `simd_count = (10 // 4) * 4 = 8` — the first 8 elements are processed two SIMD instructions at a time: one instruction adds elements `0-3` of `a` and `b` together in one shot, a second instruction adds elements `4-7`. Only the last 2 elements, `8` and `9`, fall outside any full group of 4, and those two run through the plain scalar loop instead. Two vector instructions plus two scalar instructions replaces what would otherwise be ten separate scalar additions — a reduction in instruction count that grows directly with the vector's length, not with its remainder. Mojo's `simdwidthof[DType]()` reports the natural vector width for the current target (it might be `4`, `8`, or `16` depending on whether the CPU supports SSE, AVX2, or AVX-512) so the same source compiles efficiently across CPU generations without the width being hardcoded:

```mojo
from sys.info import simdwidthof

fn vectorized_add[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    alias width = simdwidthof[dtype]()
    var simd_count = (size // width) * width

    for i in range(0, simd_count, width):
        var va = SIMD[dtype, width].load(a + i)
        var vb = SIMD[dtype, width].load(b + i)
        (va + vb).store(output + i)

    for i in range(simd_count, size):     # scalar remainder
        output[i] = a[i] + b[i]
```

This is the same pattern Chapter 4's `simd_matrix_multiply[simd_width]` used with a hardcoded width — parameterizing it over `simdwidthof[dtype]()` instead of a literal `4` is what makes one kernel body correct and fast on hardware with 128-bit, 256-bit, or 512-bit vector registers without touching the source.

## 10.2 Loop Unrolling and Fusion

Loop fusion combines multiple passes over the same tensor into one, trading extra memory traffic for extra kernel launches. `y = relu(a * b + c)` written naively is three kernels (multiply, add, relu) each reading and writing a full-sized intermediate tensor; fused into one kernel, it's a single read of `a`, `b`, `c` and a single write of `y`, with the multiply-add-relu happening in registers between them:

```mojo
fn fused_mul_add_relu_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    a: UnsafePointer[Scalar[DType.float32]],
    b: UnsafePointer[Scalar[DType.float32]],
    c: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        var z = a[idx] * b[idx] + c[idx]
        output[idx] = z if z > Float32(0.0) else Float32(0.0)   # relu, inline
```

The trade-off against Part 4's unfused, per-op backward design (Chapter 8) is real: fusion makes the *forward* pass faster but means the fused op needs its own hand-derived backward rule (the product-then-sum-then-relu chain rule, computed once and cached) rather than getting one for free by composing `MulOp`, `AddOp`, and `ReluOp`'s existing backward rules. This book keeps ops unfused by default for exactly that reason — composability of the autograd registry — and calls out fusion as a targeted optimization for the operations Chapter 10's benchmarks actually show to be hot.

## 10.3 Compile-time Optimizations

Mojo's parametric functions (`fn foo[width: Int](...)`, seen throughout Chapters 4 and 9) are resolved at compile time, not runtime — `simd_matrix_multiply[4]` and `simd_matrix_multiply[8]` are two fully separate, independently-optimized machine-code bodies, with no runtime branch on `width` anywhere in the compiled output. The same mechanism backs `TensorShape` validation from [Chapter 1](../part1/01-core-tensor-structure.md): a tensor operation whose dimensions are known at compile time can have its bounds checks elided entirely, because the compiler proves them unnecessary rather than the runtime verifying them on every call.

```mojo
fn compile_time_specialized_dot[n: Int](a: SIMD[DType.float32, n], b: SIMD[DType.float32, n]) -> Float32:
    """`n` is baked into the generated code -- no runtime loop-bound
    check, no dynamic dispatch on vector width."""
    return (a * b).reduce_add()
```

This is the concrete mechanism behind the claim in the book's introduction that Mojo delivers "zero-cost abstractions": `Tensor[dtype, rank]`-style generic code is not slower than hand-specialized code, because the compiler generates the hand-specialized code from the generic source.

## 10.4 Benchmark Framework

None of the above is worth doing without a harness that separates real speedups from noise: warm up the kernel (first-call overhead — JIT warmup, cache cold-start — would otherwise dominate the measurement), then time many repetitions and average.

```mojo
from time import now

struct Benchmark:
    var device_available: Bool

    fn time_function[func: fn() -> None](
        self, warmup_runs: Int = 5, benchmark_runs: Int = 100
    ) -> Float64:
        for _ in range(warmup_runs):
            func()
        if self.device_available:
            gpu_synchronize()

        var start_time = now()
        for _ in range(benchmark_runs):
            func()
        if self.device_available:
            gpu_synchronize()
        var end_time = now()

        var avg_ns = Float64(end_time - start_time) / Float64(benchmark_runs)
        return avg_ns / 1_000_000.0   # milliseconds
```

Applied to the convolution kernels from Chapter 9, throughput is reported in GFLOPS so results are comparable across problem sizes — a 2D convolution over an `n×n` input with a `3×3` kernel performs `2 · (n-2)² · 9` floating-point operations (one multiply and one add per kernel tap):

```mojo
fn benchmark_convolution(bench: Benchmark, size: Int):
    var operations = 2 * (size - 2) * (size - 2) * 3 * 3
    var time_ms = bench.time_function[run_conv2d_for_size_n]()
    var gflops = Float64(operations) / (time_ms * 1e6)
    print(size, "x", size, "convolution:", time_ms, "ms,", gflops, "GFLOPS")
```

### Expected Output

```
=== Mojo Convolution Performance Benchmark ===

Testing 64 x 64 convolution:
  Basic convolution:  0.412 ms, 8.51 GFLOPS
  Padded convolution: 0.447 ms, 9.15 GFLOPS
  Padding overhead:   8.5%
```

The same harness times memory bandwidth (upload/download a large buffer and divide by elapsed time) and queries device properties (core count, warp size, max threads per block) — the numbers that explain *why* a given kernel hit the throughput it did, rather than just reporting that it did.

Part 5 has made the framework fast; Part 6 uses every primitive built so far — tensors, autograd, and GPU kernels — to assemble something that actually learns.
