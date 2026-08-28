# Chapter 19: Performance Optimization Techniques — Telling a Real Speedup From a Rounding Error

> "Chapter 18 built kernels that don't waste threads, don't waste memory transactions, and don't waste global-memory reads a shared tile could have avoided. This chapter is about the layer above any single kernel: packing more arithmetic into fewer instructions, collapsing several kernels into one, letting the compiler bake a parameter in so thoroughly that checking it costs nothing at runtime, and — the part every other technique in this chapter is worthless without — a way to measure whether any of it actually helped."

**What you will understand by the end of this chapter:**

- SIMD vectorization as a compile-time-portable main-loop-plus-remainder pattern — the exact shape Chapter 5's `simd_matrix_multiply` already used — traced on two different sizes and vector widths to see where the remainder loop does and doesn't disappear
- Loop unrolling and loop fusion as two genuinely different optimizations that are easy to blur together: unrolling cuts loop-control overhead without changing the arithmetic instruction count, while fusion cuts *memory traffic* by collapsing several kernels' reads and writes into one — counted exactly, not just asserted, for both
- Why fusing operations trades a *faster forward pass* for a backward rule that has to be hand-derived once, rather than composed for free from the individually-registered `MulOp`/`AddOp`/`ReluOp` backward rules Chapter 16 already built
- Compile-time specialization as the actual mechanism behind this book's "zero-cost abstractions" claim — a parametric function like `compile_time_specialized_dot[n]` compiles to a genuinely separate, independently-optimized function body per value of `n`, not one flexible function with a hidden runtime check — and the real cost that mechanism doesn't eliminate: one compiled body per distinct instantiation
- A benchmarking harness built around the one measurement mistake that invalidates everything else: timing a kernel's first call, before caches are warm and (for GPU work) before the device has even been given time to catch up to a host that queued work and kept running

**What you need to know first:**

- Chapter 5 in full — `simdwidthof[DType]()`, the `(size // width) * width` main-loop-plus-remainder shape, and `simd_matrix_multiply`'s own already-flagged inefficiency (rereading a row from memory once per output cell) — this chapter's SIMD section is that same shape applied generically, and its loop-fusion section is one way to fix exactly that kind of inefficiency.
- Chapter 16 (the `Differentiable` trait and the registered `AddOp`/`MulOp`/`ReluOp` backward rules) — Section 19.2's fusion trade-off is stated directly in terms of what those individually-registered ops give you for free.
- Chapter 9 (parametric `fn foo[dtype: DType](...)` functions like `create_identity`) — a second, already-published example of the same compile-time specialization mechanism Section 19.3 covers.
- Chapter 18 (the convolution kernels `mojo_conv2d_kernel` and its padded variant) — Section 19.4's benchmarking example measures exactly these two kernels.

## 19.1 SIMD Vectorization `[FOUNDATIONAL]`

<a id="101-simd-vectorization"></a>

### Intuition

A print shop's press can stamp one letter per strike, or it can load a plate with four letters cast side by side and stamp all four in one strike — same ink, same press, one motion instead of four. A run of exactly `40` letters takes `10` four-letter strikes instead of `40` single strikes. A run of `42` letters takes `10` four-letter strikes for the first `40`, and then the operator has no choice but to fall back to two individual single-letter strikes for the two that don't fill a fourth plate. The press doesn't get slower or more complicated for having to do this — it just does almost all of the job at four-times throughput, and the awkward leftover at the old, one-at-a-time rate.

### Background

Chapter 5 already established the shape: split a loop into a vectorized main body plus a scalar remainder, where `simd_count = (size // width) * width` marks the boundary between them.

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

`simdwidthof[dtype]()` reports the natural vector width for whatever CPU this compiles for — `4`, `8`, or `16` `Float32`s depending on SSE, AVX2, or AVX-512 support — so this same source, unmodified, runs at each generation's native width instead of the hardcoded `4` Chapter 5's own `simd_matrix_multiply[simd_width]` used as a worked example. The trade-off this makes explicit: a wider vector register processes more elements per instruction, but the remainder loop's *maximum possible size* also grows with the width — a `width=16` register can leave up to `15` elements to the scalar loop, versus `width=4`'s worst case of `3`.

| Size | Width | `simd_count` | Vector instructions | Scalar remainder instructions | Total instructions | Scalar-only baseline |
|---|---|---|---|---|---|---|
| 10 | 4 | 8 | 2 | 2 | 4 | 10 |
| 16 | 4 | 16 | 4 | 0 | 4 | 16 |
| 17 | 8 | 16 | 2 | 1 | 3 | 17 |

### Worked Example 19.1.1 — `size=10`, `width=4`, traced instruction by instruction

`simd_count = (10 // 4) * 4 = 8`. The main loop runs twice: one instruction adds elements `0`-`3` of `a` and `b` in a single vector op, a second adds elements `4`-`7`. Elements `8` and `9` fall outside any full group of four and run through the scalar loop, one at a time. Total: `2` vector instructions plus `2` scalar instructions — `4` instructions replacing what a pure scalar loop would have needed `10` separate additions to do.

### Worked Example 19.1.2 — `size=17`, `width=8`, where the remainder is nearly a full group

`simd_count = (17 // 8) * 8 = 16`. The main loop runs twice (elements `0`-`7`, then `8`-`15`), and exactly *one* element, index `16`, falls to the scalar remainder. Total: `2` vector instructions plus `1` scalar instruction — `3` instructions total against a `17`-instruction scalar baseline, a better ratio than Worked Example 19.1.1's despite the remainder being nonzero in both cases, because `17`'s one leftover element is a much smaller fraction of `17` than `10`'s two leftover elements are of `10`.

```
[COMMON TRAP]  Assuming one dtype's natural width applies to every dtype

simdwidthof[DType.float32]() and simdwidthof[DType.float64]() are NOT
guaranteed to return the same number on the same hardware. A 256-bit
AVX2 register holds eight 4-byte float32 values but only four 8-byte
float64 values -- the register's total BIT width is fixed, so the
natural element width is inversely proportional to each element's
size. A vectorized_add[DType.float64] instantiation compiled and
tuned around an assumed width of 8 (copied from a float32 benchmark)
would silently process half as many elements per instruction as
intended, on the same physical register. This is exactly why
vectorized_add reads simdwidthof[dtype]() as a compile-time parameter
of ITS OWN dtype rather than hardcoding a width borrowed from testing
a different one.
```

## 19.2 Loop Unrolling and Fusion `[FOUNDATIONAL]`

### Intuition

Two related but different savings live under this heading, and it's worth telling them apart with two different pictures. **Unrolling**: an inspector checking a conveyor belt one item at a time re-aims, refocuses, and re-decides "is this the last item?" once per item — even though most of that per-item overhead is identical bookkeeping, not the inspection itself. Handing the inspector four items at once to check in a single glance removes three of every four "is this the last one?" decisions, without making the inspection of any individual item any different. **Fusion**: three separate quality-control stations, each requiring the part to be set down on a table, inspected, and picked back up before moving to the next station, versus one station where the inspector holds the part in their hands through all three checks and only sets it down once, at the very end.

### Background

**Unrolling** trades loop-control overhead (the increment, the bounds comparison, the branch) for a bigger loop body doing more work per iteration — it does not change how many arithmetic operations are performed, only how many times the loop's own bookkeeping runs:

```mojo
fn unrolled_add(output: UnsafePointer[Float32], a: UnsafePointer[Float32],
                 b: UnsafePointer[Float32], size: Int):
    alias UNROLL = 4
    var unrolled_count = (size // UNROLL) * UNROLL

    for i in range(0, unrolled_count, UNROLL):
        output[i]     = a[i]     + b[i]
        output[i + 1] = a[i + 1] + b[i + 1]
        output[i + 2] = a[i + 2] + b[i + 2]
        output[i + 3] = a[i + 3] + b[i + 3]

    for i in range(unrolled_count, size):     # remainder, one at a time
        output[i] = a[i] + b[i]
```

This is the identical `(size // N) * N` main-loop-plus-remainder shape Section 19.1's SIMD code uses — but where SIMD's main loop issues *one* vector instruction covering `width` elements, unrolling's main loop still issues `UNROLL` separate scalar instructions; it has simply moved `UNROLL` of them inside one loop iteration instead of spreading them across `UNROLL` iterations. The two techniques solve different problems and are frequently combined — an unrolled loop whose body is itself a SIMD load/add/store gets both the reduced loop overhead and the reduced arithmetic-instruction count at once.

**Fusion** combines multiple kernels' worth of work into one kernel body, and its saving is countable directly in memory operations, not just instructions. `y = relu(a * b + c)` written as three separate kernels reads `a` and `b` (multiply), writes an intermediate; reads that intermediate and `c` (add), writes a second intermediate; reads that second intermediate (relu), writes `y` — five tensor-sized reads and three tensor-sized writes, eight memory operations total. Fused into one kernel, the same computation reads `a`, `b`, and `c` exactly once each and writes `y` exactly once — four memory operations, with the multiply, add, and relu happening in a register between the reads and the one write:

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

### Worked Example 19.2.1 — Unrolling's overhead saving, with no change in arithmetic

For `size = 1000` and `UNROLL = 4`: `unrolled_count = (1000 // 4) * 4 = 1000` — this size divides evenly, so there is no remainder at all. The unrolled loop runs `250` iterations, each doing `4` additions in its body, for `1000` additions total — the *same* `1000` additions a plain scalar loop would perform. What changed is the loop-control overhead: a plain loop increments, compares, and branches `1000` times; this unrolled loop does so `250` times, a `4×` reduction in control overhead for zero change in arithmetic work.

### Worked Example 19.2.2 — Fusion's memory-traffic saving, counted exactly

For a tensor of any size `N`, the unfused three-kernel version of `relu(a*b+c)` performs `5` tensor-sized reads and `3` tensor-sized writes (`8` total memory operations, as counted above); the fused single-kernel version performs `3` reads and `1` write (`4` total) — exactly half the memory traffic, independent of `N`. The cost isn't free: Chapter 16 registered `AddOp`, `MulOp`, and `ReluOp` as three separate `Differentiable` implementations specifically so their backward rules compose automatically through `chain_rule_step` — `fused_mul_add_relu_kernel` has no such registry entry, and differentiating through it requires hand-deriving one combined backward rule for the whole fused expression (computed once, then reused, rather than assembled for free from three already-registered pieces). This is exactly why this book keeps ops unfused by default and treats fusion as a targeted optimization for whichever operations Section 19.4's benchmarks actually show to be hot, not a blanket policy.

```
[COMMON TRAP]  Fusing an op without registering a matching backward rule

A kernel that fuses forward computation but is dropped into the graph
under one of the ALREADY-registered op names (say, calling
chain_rule_step("mul", ...) on a node that actually ran the fused
multiply-add-relu) would run MulOp's backward rule -- which only knows
how to differentiate a plain multiplication -- against a node whose
forward pass did three operations, not one. The gradient it returns
would be a plausible-looking Tensor of the right shape, silently wrong
in value, with nothing about the shape mismatch to catch it. A fused
op needs its OWN registered Differentiable entry with a backward rule
derived for the whole fused expression -- reusing an existing op's
name for a kernel that no longer matches what that name's backward
rule assumes is exactly how a correct-looking forward pass ends up
paired with an incorrect gradient.
```

## 19.3 Compile-time Optimizations `[FOUNDATIONAL]`

### Intuition

A print shop could cast one adjustable metal plate that reads its own configuration before every single stamp — "am I set to 4-wide or 8-wide today?" — paying that small check on every strike, forever. Or it could cast two entirely separate, purpose-built plates ahead of time, one fixed at 4-wide and one fixed at 8-wide, and simply reach for whichever one a given job needs. The second shop pays a cost the first one doesn't — two plates to store instead of one — but neither plate ever asks itself a question at strike time, because the question was already answered back when it was cast.

### Background

Mojo's parametric functions — `fn foo[width: Int](...)`, seen throughout Chapters 5 and 9 — are resolved at compile time, not runtime. `compile_time_specialized_dot[4]` and `compile_time_specialized_dot[8]` are two fully separate, independently-optimized machine-code bodies, with no runtime branch on the vector width anywhere in either compiled output:

```mojo
fn compile_time_specialized_dot[n: Int](a: SIMD[DType.float32, n], b: SIMD[DType.float32, n]) -> Float32:
    """`n` is baked into the generated code -- no runtime loop-bound
    check, no dynamic dispatch on vector width."""
    return (a * b).reduce_add()
```

This is the concrete mechanism behind this book's "zero-cost abstractions" claim, and it isn't a new idea introduced here — Chapter 5's `simd_matrix_multiply[simd_width]` and Chapter 9's `create_identity[dtype]` are both already-published examples of the exact same thing: a generic-looking function that the compiler turns into as many separate, fully specialized bodies as the program actually instantiates, none of which pay a runtime cost for the genericity the source code appears to have.

### Worked Example 19.3.1 — Two instantiations, two independent answers

`compile_time_specialized_dot[4]` called with `a = [1, 2, 3, 4]` and `b = [5, 6, 7, 8]`: `1·5 + 2·6 + 3·7 + 4·8 = 5 + 12 + 21 + 32 = 70`. `compile_time_specialized_dot[2]` called with `a = [2, 3]` and `b = [4, 5]`: `2·4 + 3·5 = 8 + 15 = 23`. These aren't two calls to one function with a runtime-varying length — `n=4` and `n=2` select two *different, separately-compiled* functions, each hardcoded to its own vector width, the way `compile_time_specialized_dot[4]`'s generated code has no code path at all for handling a 2-wide input, and vice versa.

```
[COMMON TRAP]  Zero runtime cost is not zero cost

Every distinct value of n a program actually instantiates
compile_time_specialized_dot with produces its OWN compiled function
body. A program that calls this function at n=2, n=4, n=8, and n=16
somewhere in its source compiles four separate machine-code bodies,
not one generic body handling four cases -- the same "code bloat"
trade-off C++ template instantiation has always made. The cost compile-
time specialization eliminates is a RUNTIME one (a branch or a
dictionary lookup on which width to use); it does not eliminate cost
altogether, it moves the cost to compile time (longer builds) and to
binary size (more distinct function bodies shipped) instead.
```

## 19.4 Benchmark Framework `[FOUNDATIONAL]`

### Intuition

Timing a runner's very first sprint of the day, straight off the bench with cold muscles, measures how slow a cold start is — not how fast that runner actually runs. A fair measurement lets them run a few warm-up sprints first, uncounted, and only then starts the stopwatch on several representative sprints, averaged together, so one unusually good or bad rep doesn't dominate the result. GPU work has an extra wrinkle a runner doesn't: the host can *launch* a kernel and immediately move on to launching the next line of code, well before the device has actually finished — timing the host's launch loop alone measures how fast the host can hand off work, not how fast the device does it, unless the host is made to wait for the device to actually catch up first.

### Background

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
            gpu_synchronize()      # drain the queue before the clock starts

        var start_time = now()
        for _ in range(benchmark_runs):
            func()
        if self.device_available:
            gpu_synchronize()      # and again before it stops
        var end_time = now()

        var avg_ns = Float64(end_time - start_time) / Float64(benchmark_runs)
        return avg_ns / 1_000_000.0   # milliseconds
```

The two `gpu_synchronize()` calls bracket the timed region for exactly the reason above: without the first one, warmup work still queued on the device could bleed into the timed region; without the second, the host's timer could stop before the device has actually finished the last of the `benchmark_runs` launches, undercounting the true elapsed time. Applied to Chapter 18's convolution kernels, throughput is reported in GFLOPS so results are comparable across problem sizes rather than only across identical ones — a `3×3`-kernel 2D convolution performs one multiply and one add per kernel tap, so `2` floating-point operations per tap, times `9` taps, times however many output positions there are:

```mojo
fn benchmark_convolution(bench: Benchmark, size: Int):
    var basic_ops = 2 * (size - 2) * (size - 2) * 3 * 3    # valid conv: (size-2)^2 outputs -- Chapter 18's naive/tiled kernels
    var basic_time_ms = bench.time_function[run_conv2d_basic_for_size_n]()
    var basic_gflops = Float64(basic_ops) / (basic_time_ms * 1e6)

    var padded_ops = 2 * size * size * 3 * 3               # same-shaped conv: size^2 outputs -- Chapter 18's padded variant
    var padded_time_ms = bench.time_function[run_conv2d_padded_for_size_n]()
    var padded_gflops = Float64(padded_ops) / (padded_time_ms * 1e6)

    print(size, "x", size, "convolution:")
    print("  Basic convolution: ", basic_time_ms, "ms,", basic_gflops, "GFLOPS")
    print("  Padded convolution:", padded_time_ms, "ms,", padded_gflops, "GFLOPS")
```

Note the FLOP count itself differs between the two variants, not just the timing: the unpadded ("valid") kernel produces `(size-2)²` outputs, while the padded ("same") kernel produces `size²` — more total work for the same input, which is exactly the "padding overhead" the two GFLOPS figures let a reader compare directly.

### Worked Example 19.4.1 — Converting a hypothetical timing into GFLOPS

No GPU run backs a real number here — this book's Mojo has never been compiled or run — but the formula itself is worth tracing on concrete numbers. For `size = 64`: `basic_ops = 2 × (64-2)² × 3 × 3 = 2 × 3,844 × 9 = 69,192`. If a run's timer *hypothetically* measured `basic_time_ms = 0.05`: `basic_gflops = 69,192 / (0.05 × 1,000,000) ≈ 1.384`. For the padded variant on the same `size = 64`: `padded_ops = 2 × 64² × 9 = 73,728` — more total operations for the same input, consistent with `size² > (size-2)²`. If that run *hypothetically* measured `padded_time_ms = 0.06`: `padded_gflops = 73,728 / (0.06 × 1,000,000) ≈ 1.229`. Slightly more total work (`73,728` vs `69,192` operations) taking slightly longer (`0.06` ms vs `0.05` ms) is exactly the trade-off "padding overhead" describes — this worked example demonstrates how the formula converts a timing into a comparable throughput number, not what any particular piece of hardware would actually report.

The same harness times memory bandwidth (upload or download a large buffer and divide its size by the elapsed time) and queries device properties (core count, warp size, max threads per block) using the identical warmup-then-average pattern — the numbers that explain *why* a kernel hit the throughput it did, rather than only reporting that it did.

## 19.5 Reference Implementations

None of this chapter's Mojo has been compiled or run, consistent with the rest of this book's newly-composed material — there is no captured benchmark session to reproduce, and the specific millisecond and GFLOPS figures in an earlier draft of this section did not actually follow from the FLOP-counting formula given alongside them; Worked Example 19.4.1 above replaces those with an explicitly hypothetical trace of the same formula instead. What follows is every function this chapter derived, assembled in one place:

```mojo
from sys.info import simdwidthof
from time import now

# ---- 19.1: SIMD vectorization ----

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
    for i in range(simd_count, size):
        output[i] = a[i] + b[i]


# ---- 19.2: loop unrolling and fusion ----

fn unrolled_add(output: UnsafePointer[Float32], a: UnsafePointer[Float32],
                 b: UnsafePointer[Float32], size: Int):
    alias UNROLL = 4
    var unrolled_count = (size // UNROLL) * UNROLL
    for i in range(0, unrolled_count, UNROLL):
        output[i]     = a[i]     + b[i]
        output[i + 1] = a[i + 1] + b[i + 1]
        output[i + 2] = a[i + 2] + b[i + 2]
        output[i + 3] = a[i + 3] + b[i + 3]
    for i in range(unrolled_count, size):
        output[i] = a[i] + b[i]

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
        output[idx] = z if z > Float32(0.0) else Float32(0.0)


# ---- 19.3: compile-time specialization ----

fn compile_time_specialized_dot[n: Int](a: SIMD[DType.float32, n], b: SIMD[DType.float32, n]) -> Float32:
    return (a * b).reduce_add()


# ---- 19.4: benchmarking ----

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
        return avg_ns / 1_000_000.0

fn benchmark_convolution(bench: Benchmark, size: Int):
    var basic_ops = 2 * (size - 2) * (size - 2) * 3 * 3
    var basic_time_ms = bench.time_function[run_conv2d_basic_for_size_n]()
    var basic_gflops = Float64(basic_ops) / (basic_time_ms * 1e6)

    var padded_ops = 2 * size * size * 3 * 3
    var padded_time_ms = bench.time_function[run_conv2d_padded_for_size_n]()
    var padded_gflops = Float64(padded_ops) / (padded_time_ms * 1e6)

    print(size, "x", size, "convolution:")
    print("  Basic convolution: ", basic_time_ms, "ms,", basic_gflops, "GFLOPS")
    print("  Padded convolution:", padded_time_ms, "ms,", padded_gflops, "GFLOPS")
```

### Expected Output

There is no captured run to reproduce here — see the note above. Worked Examples 19.1.1 through 19.4.1 remain the source of truth for what this chapter's formulas compute on concrete numbers; this section is an index into them, not a substitute for the by-hand arithmetic above.

## Chapter Summary

SIMD vectorization splits a loop into a vectorized main body plus a scalar remainder at the boundary `simd_count = (size // width) * width`, and a wider vector width doesn't just process more elements per instruction — it also widens the worst-case remainder, which is why `size=17, width=8`'s single leftover element still beats `size=10, width=4`'s two leftover elements in overall instruction count, despite both remainders being nonzero. Unrolling and fusion solve genuinely different problems that share a similar-sounding name: unrolling cuts loop-control overhead (`4×` fewer increments, comparisons, and branches for `UNROLL=4`) without touching the arithmetic instruction count at all, while fusion cuts memory traffic directly — `8` tensor-sized memory operations collapsed to `4` for `relu(a*b+c)` — at the cost of losing Chapter 16's free backward-rule composition and requiring one hand-derived gradient for the whole fused expression instead. Compile-time specialization is the concrete mechanism behind this book's zero-cost-abstraction claim: `compile_time_specialized_dot[n]` genuinely compiles to a separate, independently-optimized function body for every distinct `n` a program instantiates, eliminating the *runtime* cost of genericity entirely — while quietly relocating that cost to compile time and binary size, one compiled body per instantiation, exactly the trade-off C++ templates have always made. None of this is worth doing without a benchmarking harness built around warmup runs (so first-call cache-cold overhead doesn't dominate the measurement) and, for GPU work specifically, explicit synchronization bracketing the timed region — without it, a host that queues work and races ahead measures its own launch-loop speed, not the device's actual execution time.

## Self-Check Questions

1. For `size = 33` and `width = 16`, compute `simd_count`, the number of vector instructions, and the number of scalar remainder instructions.
2. `unrolled_add` with `UNROLL = 4` is run on `size = 4,002`. How many full unrolled iterations run, how many elements are left to the scalar remainder loop, and how many total addition operations are performed — does that total differ from what a plain, non-unrolled scalar loop would perform on the same input?
3. A team fuses `sigmoid(x @ W + b)` into a single kernel for forward-pass speed, then wires it into the computation graph under the existing registered name `"matmul"` so it reuses `MatMulOp`'s already-derived backward rule. What goes wrong when this graph is differentiated, and why doesn't a shape check catch it?
4. `compile_time_specialized_dot` is called at `n=4`, `n=8`, and `n=16` across a program's source. How many separate compiled function bodies does this produce, and what is the actual resource cost being paid in exchange for zero runtime branching?
5. A benchmark measures a GPU kernel's `time_function` result *without* either `gpu_synchronize()` call. What specifically is being measured instead of the kernel's true execution time, and in which direction (too fast or too slow) would you expect the reported time to be biased?

## Where We Go Next

Part 5 has made the framework's tensor and GPU-kernel layers fast, in the same sense Chapter 18 made them correct: with numbers traced by hand rather than asserted. Part 6 spends every primitive built through both parts — tensors, the autograd engine, and now performance-tuned kernels — assembling something that actually learns: a multi-layer neural network trained end to end on this same codebase.

## Worked Solutions

**1.** `simd_count = (33 // 16) * 16 = 2 * 16 = 32`. The main loop covers elements `0`-`15` and `16`-`31` in two vector instructions. One element, index `32`, falls to the scalar remainder — `1` scalar instruction. Total: `2` vector instructions plus `1` scalar instruction.

**2.** `unrolled_count = (4,002 // 4) * 4 = 1,000 * 4 = 4,000`. That's `1,000` full unrolled iterations (each doing `4` additions), covering elements `0` through `3,999`. The remaining `2` elements (`4,000` and `4,001`) run through the scalar remainder loop one at a time. Total additions performed: `4,000 + 2 = 4,002` — identical to what a plain scalar loop over the same `4,002` elements would compute; unrolling changes only how many times the loop's own increment/compare/branch overhead runs (`1,000 + 2 = 1,002` times here, versus `4,002` times for a non-unrolled loop), not the arithmetic total.

**3.** `MatMulOp.backward` computes `grad_a = grad_output @ Bᵀ` and `grad_b = Aᵀ @ grad_output` — the correct gradient for a *plain* matrix multiplication, with no sigmoid and no bias-add anywhere in that formula. Run against a node whose forward pass actually computed `sigmoid(x @ W + b)`, this backward rule returns a gradient that completely ignores both the `+ b` and the `sigmoid`'s own local derivative (`output · (1-output)` from Chapter 16.5) — a Tensor of exactly the right shape (since `MatMulOp.backward`'s output shapes only depend on `x` and `W`'s shapes, which are unchanged), just numerically wrong by a large, silent margin. A shape check can't catch this because the bug is entirely about *which formula* ran, not about the shapes those formulas produce — `grad_output @ Bᵀ` and `Aᵀ @ grad_output` have exactly the shapes of `x` and `W` respectively, whether or not a sigmoid or a bias-add happened in between.

**4.** Three separate compiled function bodies — one each for `n=4`, `n=8`, and `n=16` — because each distinct compile-time parameter value produces its own independently-generated machine code, not one shared body with a runtime branch on `n`. The resource cost being paid is compile time (three function bodies to generate and optimize instead of one) and binary size (three compiled bodies shipped in the final program instead of one) — the runtime cost of choosing between them is what's eliminated, not cost altogether.

**5.** Without the first `gpu_synchronize()`, warmup work still queued on the device can bleed into the timed region, and without the second, the host's clock can stop before the device has actually finished the last of the `benchmark_runs` launches — in both cases, the timer is measuring how long the *host* took to issue `benchmark_runs` kernel launches, not how long the *device* took to execute them. Since launching work asynchronously is typically far faster than actually running it, the reported time would be biased too fast — understating the kernel's true execution time, potentially by a large margin if the device queue backs up well past when the host's loop finishes.
