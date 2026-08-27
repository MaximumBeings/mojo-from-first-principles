# Chapter 12: Element-wise Operations — One Thread, One Position, No Dependencies

> "Chapter 4's whole argument was about how efficiently a warp's threads fetch the cargo they need to compute something. Element-wise operations are the simplest possible thing to do once that cargo has arrived: one thread, one output position, and nothing computed by any other thread that this one needs to wait for. It's almost too simple to dwell on — until the one place in this chapter where the launch configuration and the kernel body quietly stop agreeing with each other."

**What you will understand by the end of this chapter:**

- `vector_add_kernel`'s indexing formula — `thread_idx.x` alone — and precisely what breaks if you scale its launch configuration up to a production-sized vector without also updating that formula to match what every later kernel in this chapter actually uses
- Local derivatives for `+`, `-`, `×`, `÷`, `pow`, and `exp`, each checked against a real finite-difference nudge on the chapter's own numbers, as a first, concrete preview of the chain rule
- Why `exp` gets a dedicated kernel with a self-referencing derivative (`d/dx[eˣ] = eˣ`) that a later backward pass can exploit by reusing the cached forward output instead of recomputing anything
- The stride-0 broadcasting trick from Chapter 7.4, now applied literally inside a GPU kernel body rather than as tensor metadata — and exactly which broadcasting shapes `broadcast_add_kernel` handles versus the fully general rule Chapter 7.4 built

**What you need to know first:**

- Chapter 4 (the host/device split, `thread_idx`/`block_idx`, and kernel launch mechanics — every kernel in this chapter is a direct instance of "one thread per output element")
- Chapter 7.4 (broadcasting via stride-0 dimensions, computed as pure shape logic — this chapter puts that exact mechanism to work inside a kernel for the first time)
- Chapter 5 (SIMD width and the CPU-side analogue of "apply the same operation to every position independently")

<a id="31-addition-and-subtraction"></a>
## 12.1 Addition and Subtraction: One Thread Per Output Position `[FOUNDATIONAL]`

### Intuition

Element-wise addition is the purest case of "no thread needs anything any other thread computed": `C[i] = A[i] + B[i]` for every `i`, independently, all at once. This is essentially Puzzle 2 from the GPU-programming exercises Chapter 4 was built around, and it's worth treating as the baseline every other kernel in this chapter (and the chain-rule machinery several chapters from now) gets compared against.

### Background

```mojo
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE

fn vector_add_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]]
):
    """C[i] = A[i] + B[i] for all i -- one thread per element."""
    var i = thread_idx.x
    if i < SIZE:
        output[i] = a[i] + b[i]
```

### Worked Example 12.1.1 — Four threads, four sums

`a = [1, 2, 3, 4]`, `b = [10, 20, 30, 40]`. With one block of `SIZE = 4` threads, thread `0` computes `output[0] = a[0] + b[0] = 1 + 10 = 11`; thread `1` computes `2 + 20 = 22`; thread `2` computes `3 + 30 = 33`; thread `3` computes `4 + 40 = 44` — matching the recorded output `[11.0, 22.0, 33.0, 44.0]` exactly, all four computed simultaneously since none depends on any other.

### Worked Example 12.1.2 — Subtraction, same launch, flipped operator

The same launch configuration with `output[i] = a[i] - b[i]` on the same vectors gives `[10-1, 20-2, 30-3, 40-4] = [9, 18, 27, 36]` — no new kernel structure, just a different one-character arithmetic operator inside the identical per-thread body.

### Worked Example 12.1.3 — Scaling the launch, without scaling the kernel

For a production-sized, one-million-element vector, the natural move is to retile: pick `THREADS_PER_BLOCK = 256` and compute `BLOCKS_PER_GRID = (1_000_000 + 255) // 256 = 3907` blocks, with a `if i < SIZE` bounds check protecting the tail block (covering global indices `999,936` through `1,000,191`, of which only `64` fall inside real data). That reasoning is correct — *for a kernel that computes its index as* `block_idx.x * block_dim.x + thread_idx.x`, the pattern every kernel from Section 12.2 onward in this chapter actually uses.

```
[COMMON TRAP]  vector_add_kernel's own indexing formula is not that kernel

Look again at the kernel body above: `var i = thread_idx.x` — no `block_idx.x`
anywhere in it. That's correct only because this section's launch uses a
single block (`BLOCKS_PER_GRID = 1`), so `thread_idx.x` alone already ranges
over every valid index, `0` through `SIZE - 1`. Retile the launch to 3907
blocks of 256 threads each, as Worked Example 12.1.3 describes, *without*
also rewriting the kernel body to `var i = block_idx.x * block_dim.x +
thread_idx.x`, and every one of those 3907 blocks still computes `i` as
`thread_idx.x` alone — a value in `[0, 255]` in every single block,
regardless of which block it is. All 3907 blocks would redundantly (and
racily) write to `output[0]` through `output[255]` only; the other
999,744 positions in the buffer are never touched by any thread, in any
block, at any point. The bounds check `if i < SIZE` doesn't catch this
either, since `i` never grows large enough to trip it — the bug is that `i`
never reflects which block is running at all. Every kernel from
elementwise_mul_kernel onward in this chapter uses the combined formula
correctly; vector_add_kernel is the one exception, and it's only safe
because this section never actually launches more than one block.
```

**Local derivative:** `∂(a+b)/∂a = 1` and `∂(a+b)/∂b = 1` — a change to either input passes straight through to the output, unscaled.

## 12.2 Multiplication and Division: The Same Pattern, Sharper Arithmetic `[FOUNDATIONAL]`

### Intuition

Multiplication and division run the identical one-thread-per-element pattern Section 12.1 established — the only thing that changes is which operator sits inside the `if i < size` body, and, this time, the indexing formula that actually generalizes past a single block.

### Background

```mojo
fn elementwise_mul_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = a[i] * b[i]

fn elementwise_div_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        # Division by zero produces IEEE-754 inf/nan rather than a
        # trap -- a later chapter's autograd engine checks for both
        # before accepting a gradient into an optimizer step.
        output[i] = a[i] / b[i]
```

Both kernels compute `i` as `block_idx.x * block_dim.x + thread_idx.x` — the general form Worked Example 12.1.3's `[COMMON TRAP]` showed `vector_add_kernel` is missing, and the form that correctly identifies a thread's global position across however many blocks a launch actually uses.

### Worked Example 12.2.1 — Forward values and a finite-difference check

`a = [2, 3, 4]`, `b = [5, 6, 7]`: element-wise multiply gives `[2×5, 3×6, 4×7] = [10, 18, 28]`; element-wise divide gives `[2/5, 3/6, 4/7] = [0.4, 0.5, 0.571...]`. Checking the multiplication derivative numerically at position `0` (`a=2, b=5, c=10`): nudging `a` from `2` to `2.01` moves `c` from `10` to `2.01 × 5 = 10.05`, a change of `0.05` over a nudge of `0.01` — a slope of `5`, matching `∂c/∂a = b = 5` exactly, since `c = a × b`'s partial derivative with respect to `a` is just `b`.

### Worked Example 12.2.2 — Division's two derivatives

For `c = a / b`: `∂c/∂a = 1/b` and `∂c/∂b = -a/b²`. At position `0` (`a=2, b=5`): `∂c/∂a = 1/5 = 0.2` and `∂c/∂b = -2/25 = -0.08`. Both backward kernels a later chapter builds from these two rules are one more elementwise pass over the same-shaped buffers — no new infrastructure beyond what Section 12.1 already established, just a kernel body multiplying by whichever of these local derivatives applies to that operand.

## 12.3 Power and Exponential Functions: Dedicated Kernels for the Two That Recur Everywhere `[FOUNDATIONAL]`

### Intuition

`pow` and `exp` show up in essentially every loss function and activation function later in this book, which is why they get their own dedicated kernels here rather than being routed through a generic "apply this scalar function" abstraction — a dedicated kernel lets the compiler specialize and inline the underlying `exp`/`pow` intrinsic directly, instead of dispatching through a function pointer or a branch on which operation was requested.

### Background

```mojo
fn elementwise_pow_kernel(
    output: UnsafePointer[Scalar[dtype]],
    base: UnsafePointer[Scalar[dtype]],
    exponent: Float32,
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = math_pow(base[i], exponent)

fn elementwise_exp_kernel(
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = exp(input[i])
```

### Worked Example 12.3.1 — Squaring three values, and checking the derivative by hand

`x = [1, 2, 3]`: `pow(x, 2)` gives `[1, 4, 9]`. Checking the derivative at `x=2`: `d/dx[xⁿ] = n·x^(n-1)`, so with `n=2` that's `2×2 = 4`. A finite-difference check confirms it: `pow(2.01, 2) = 4.0401`, and `(4.0401 - 4.0) / 0.01 = 4.01 ≈ 4` — the small residual above `4` is exactly the expected error of a finite forward-difference approximation, not a discrepancy in the calculus.

### Worked Example 12.3.2 — The derivative that is the function

`exp(x)` on the same input gives `[e¹, e², e³] ≈ [2.71828, 7.38906, 20.0855]`. Its derivative, `d/dx[eˣ] = eˣ`, is the function itself — at `x=2` that's `e² ≈ 7.38906`, exactly the forward value already computed for that position, not a separately-derived number. This self-referencing property is exactly why a later chapter's `ExpOp.backward` reads the *cached* forward output during the backward pass rather than recomputing `exp(x)` a second time — the forward pass already computed the one number the backward pass needs.

## 12.4 Broadcasting: The Stride-0 Trick, Now Inside a Kernel `[FOUNDATIONAL]`

### Intuition

Every operation so far in this chapter assumed both inputs were the same shape. Broadcasting is what lets `Tensor[2, 3] + Tensor[1, 3]` work anyway, by treating the smaller tensor's missing dimension as if it were silently repeated — and Section 7.4 already built the shape-level machinery for deciding *when* that's legal and what the resulting strides should be. This section is where that machinery gets consumed: not as tensor metadata computed once ahead of time, but as a literal stride value read by every single GPU thread, every time it computes its own input address.

### Background

```
A = 1  2  3        B = 10  20  30
    4  5  6
```

```mojo
fn broadcast_add_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    a_stride_row: Int, b_stride_row: Int,   # 0 if that operand is broadcast along rows
    rows: Int, cols: Int,
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row < rows and col < cols:
        var a_val = a[row * a_stride_row + col]
        var b_val = b[row * b_stride_row + col]
        output[row * cols + col] = a_val + b_val
```

Setting a tensor's stride to `0` along a dimension — Section 7.4's `BroadcastSpec` computing exactly this value ahead of the kernel launch — means "don't advance the memory address at all as this dimension's index increases," which is precisely "keep re-reading the same values." This kernel is a deliberately narrower instrument than Section 7.4's general rule, though: it only ever broadcasts along the row dimension, via one `_stride_row` parameter per operand, rather than handling an arbitrary number of dimensions each independently aligned from the right the way `BroadcastSpec` does. It's the 2-D specialization of that general rule, not a reimplementation of the whole thing.

### Worked Example 12.4.1 — Tracing `b_stride_row = 0`

`A` is `2×3`, `B` is a single row of 3 values meant to add to every row of `A`. With `b_stride_row = 0`: for row `0`, the kernel reads `b[0×0 + col] = b[col]`; for row `1`, it reads `b[1×0 + col] = b[col]` — the *same* address both times, `B`'s one and only row, read twice. `A`'s stride is unmodified (a normal, non-zero row stride), so each row of `A` still reads its own distinct data. The result: `[[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]] = [[11, 22, 33], [14, 25, 36]]`, with `B` never copied — no extra memory traffic beyond what a same-shaped add would already cost, which is the entire point of doing broadcasting at the stride level instead of the copy level.

### Worked Example 12.4.2 — The symmetric case

`a_stride_row = 0` (with `b_stride_row` left at its normal nonzero value) produces the mirror image: every row of the output grid re-reads row `0` of `A` while `B` supplies a genuinely different row each time — a virtual broadcast of `A` down `B`'s shape instead of the other way around. Nothing about the kernel body needs to know in advance which operand is the one being broadcast; both `a_stride_row` and `b_stride_row` are ordinary parameters, and whichever one the caller sets to `0` is the one that gets silently repeated.

## 12.5 Reference Implementations

Only Section 12.1's kernel was ever compiled and run as a standalone file with captured console output — every other kernel in this chapter (multiplication, division, power, exponential, broadcasting) is presented in the original source as an annotated snippet with hand-traced values in the surrounding prose, the same style Chapter 11 used throughout. The one real captured run and the four illustrative snippets are both reproduced verbatim below, exactly as they appear in the source.

### File: `puzzle_02_vector_add.mojo` — Section 12.1 (captured run)

**Run:** `pixi run mojo puzzle_02_vector_add.mojo`

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32

fn vector_add_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]]
):
    """C[i] = A[i] + B[i] for all i -- one thread per element."""
    var i = thread_idx.x
    if i < SIZE:
        output[i] = a[i] + b[i]

fn main() raises:
    print("=== Element-wise vector addition ===")
    with DeviceContext() as ctx:
        var vector_a = UnsafePointer[Scalar[dtype]].alloc(SIZE)
        var vector_b = UnsafePointer[Scalar[dtype]].alloc(SIZE)
        var output_data = UnsafePointer[Scalar[dtype]].alloc(SIZE)

        for i in range(SIZE):
            vector_a[i] = Float32(i + 1)          # [1, 2, 3, 4]
            vector_b[i] = Float32((i + 1) * 10)   # [10, 20, 30, 40]

        ctx.enqueue_function[vector_add_kernel](
            output_data, vector_a, vector_b,
            grid_dim=(BLOCKS_PER_GRID, 1, 1),
            block_dim=(THREADS_PER_BLOCK, 1, 1)
        )
        ctx.synchronize()

        print("\nOutput vector (a + b):")
        for i in range(SIZE):
            print("output[", i, "] =", output_data[i])

        vector_a.free()
        vector_b.free()
        output_data.free()
```

### Expected Output for `puzzle_02_vector_add.mojo`

```
=== Element-wise vector addition ===

Output vector (a + b):
output[ 0 ] = 11.0
output[ 1 ] = 22.0
output[ 2 ] = 33.0
output[ 3 ] = 44.0
```

### Multiplication and division kernels — Section 12.2 (illustrative, uncaptured)

```mojo
fn elementwise_mul_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = a[i] * b[i]

fn elementwise_div_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        # Division by zero produces IEEE-754 inf/nan rather than a
        # trap -- a later chapter's autograd engine checks for both
        # before accepting a gradient into an optimizer step.
        output[i] = a[i] / b[i]
```

### Power and exponential kernels — Section 12.3 (illustrative, uncaptured)

```mojo
from math import exp, pow as math_pow

fn elementwise_pow_kernel(
    output: UnsafePointer[Scalar[dtype]],
    base: UnsafePointer[Scalar[dtype]],
    exponent: Float32,
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = math_pow(base[i], exponent)

fn elementwise_exp_kernel(
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < size:
        output[i] = exp(input[i])
```

### Broadcast kernel — Section 12.4 (illustrative, uncaptured)

```mojo
fn broadcast_add_kernel(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    a_stride_row: Int, b_stride_row: Int,   # 0 if that operand is broadcast along rows
    rows: Int, cols: Int,
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row < rows and col < cols:
        var a_val = a[row * a_stride_row + col]
        var b_val = b[row * b_stride_row + col]
        output[row * cols + col] = a_val + b_val
```

## Chapter Summary

Every kernel in this chapter shares one structure: one thread, one output position, no dependency on any other thread's result — the property that makes element-wise operations both the highest-frequency operation in a training loop and the easiest to parallelize. `vector_add_kernel` computes that structure correctly for a single block (`thread_idx.x` alone is a valid global index when there's only one block), but this chapter traced exactly what breaks if that kernel's launch configuration is scaled up to a production-sized vector without also updating its indexing formula to the combined `block_idx.x * block_dim.x + thread_idx.x` form every later kernel in the chapter actually uses — every block would redundantly write to the same first `THREADS_PER_BLOCK` output positions, leaving the rest of the buffer untouched. Multiplication, division, power, and exponential all follow the same one-thread-per-element shape, each paired with a local derivative checked against a real finite-difference nudge on the chapter's own numbers — `∂c/∂a = b` for multiplication, `∂c/∂a = 1/b` and `∂c/∂b = -a/b²` for division, `n·x^(n-1)` for power, and the self-referencing `eˣ` for exponential, whose backward pass can reuse a cached forward value instead of recomputing anything. Broadcasting closed the chapter by taking Chapter 7.4's stride-0 mechanism — computed there as pure shape metadata — and applying it literally inside a kernel body for the first time: `b_stride_row = 0` makes every row of the output grid re-read the exact same row of `B`, with zero extra memory traffic and zero copying, though `broadcast_add_kernel` only ever broadcasts along rows, a narrower case than Chapter 7.4's fully general, arbitrary-dimension rule.

## Self-Check Questions

1. `vector_add_kernel` is launched with `BLOCKS_PER_GRID = 2` and `THREADS_PER_BLOCK = 4` for an 8-element vector, with `SIZE` updated to `8`, but the kernel body is left completely unchanged. Which output positions actually get written, and which ones never get touched by any thread?
2. Using `∂c/∂b = -a/b²` for `c = a/b`, compute the local derivative with respect to `b` at `a=4, b=2`, and verify it with a finite-difference nudge (`b` from `2` to `2.01`) the way Worked Example 12.2.1 checked multiplication.
3. `pow(x, 3)` is applied to `x = 2`. Using `d/dx[xⁿ] = n·x^(n-1)`, what is the exact derivative at that point, and what finite-difference nudge and result would confirm it the way Worked Example 12.3.1 did for `n=2`?
4. `broadcast_add_kernel` is called with `a_stride_row` set to `A`'s normal (nonzero) row stride and `b_stride_row` set to `0`, on a `3×4` `A` and a `1×4` `B`. Write out, for each of the three rows, which literal element of `B` gets added to `A[row, 2]`.
5. Why does `elementwise_exp_kernel`'s corresponding backward pass not need to call `exp()` a second time, when `elementwise_pow_kernel`'s backward pass (computing `n·x^(n-1)`) generally does need to re-touch `x` and recompute a power?

## Where We Go Next

Chapter 13 (`part2/02-matrix-operations.md`) moves from per-element operations to operations that mix elements across a whole row or column — matrix multiplication, transpose, and the general tensor contraction both generalize from — the first place in this book where one output position genuinely depends on more than one input position at once.

## Worked Solutions

**1.** With `BLOCKS_PER_GRID = 2` and `THREADS_PER_BLOCK = 4`, `thread_idx.x` ranges over `0, 1, 2, 3` in *both* blocks (block index isn't part of the formula at all). Block `0`'s four threads compute `output[0]` through `output[3]`; block `1`'s four threads also compute `i = 0, 1, 2, 3` and write to the exact same `output[0]` through `output[3]` — a redundant, racing rewrite of the same four positions. `output[4]` through `output[7]` are never written by any thread in either block, even though `SIZE = 8` and the bounds check `if i < SIZE` would happily allow indices up to `7` — nothing in the kernel ever produces an `i` value of `4` or higher, because `thread_idx.x` alone tops out at `3` regardless of which block is running.

**2.** `∂c/∂b = -a/b²` at `a=4, b=2`: `-4/4 = -1`. Finite-difference check: `c = a/b = 4/2 = 2.0` at `b=2`; at `b=2.01`, `c = 4/2.01 ≈ 1.99005`. The change in `c` is `1.99005 - 2.0 = -0.00995` over a nudge of `0.01`, giving a slope of approximately `-0.995 ≈ -1` — matching `∂c/∂b = -1` (the small residual is the expected finite-difference approximation error, the same pattern Worked Example 12.3.1 saw for `pow`).

**3.** `d/dx[x³] = 3x²`; at `x=2`, that's `3×4 = 12`. A finite-difference check nudges `x` from `2` to `2.01`: `pow(2.01, 3) = 8.120601`, and `pow(2, 3) = 8.0`, so `(8.120601 - 8.0)/0.01 = 12.0601 ≈ 12` — confirming the derivative, with the small residual above `12` again being the expected forward-difference approximation error rather than a discrepancy.

**4.** With `b_stride_row = 0`, every row reads `b[row × 0 + col] = b[col]` — the same single row of `B` regardless of `row`. For column `2` specifically, every one of the three rows (`row = 0`, `row = 1`, `row = 2`) adds the exact same element, `B[0, 2]` — the third value in `B`'s one and only row — to `A[row, 2]`. Which row of `A` is involved changes; which element of `B` is read does not.

**5.** `exp`'s derivative is `eˣ` itself — the *same value* the forward pass already computed and, per this chapter's Worked Example 12.3.2, cached as `output`. The backward pass can multiply the incoming gradient by that cached value directly, with no further computation needed to obtain `eˣ`. `pow(x, n)`'s derivative, `n·x^(n-1)`, is a *different* value from the forward result (`xⁿ`) for any `n ≠ 1` — computing it requires re-reading `x` (not `output`) and evaluating a new power, `x^(n-1)`, that was never computed during the forward pass at all, so there's no cached value the backward pass could reuse in its place.
