# Chapter 3: Element-wise Operations

With tensors, views, and memory management in place, Part 2 starts putting the infrastructure to work: real arithmetic on real data. Element-wise operations — addition, subtraction, multiplication, division, and broadcasting — apply the same scalar operation to every position in a tensor independently, which sounds almost too simple to dwell on, except that "independently" is precisely what makes them the easiest operation in this book to parallelize, and the highest-frequency operation in any training loop. This is where SIMD width and memory layout choices from Part 0 start paying off directly, and it's also the first chapter whose operations Chapter 7 will need to differentiate — so every section below ends by stating the local derivative alongside the arithmetic, in preparation.

## 3.1 Addition and Subtraction

Element-wise addition takes two same-shaped tensors and produces a third, where each output position holds the sum of the two inputs at that same position: `C[i] = A[i] + B[i]`. On a CPU this is one loop; on a GPU it's an opportunity to run every position simultaneously, one GPU thread per output element, since none of the additions depend on each other. This is essentially "Puzzle 2" from the GPU-programming exercises that motivated Part 0.4, worked with concrete vectors: `a = [1, 2, 3, 4]` and `b = [10, 20, 30, 40]` should produce `[11, 22, 33, 44]` — thread 0 computes `1+10=11`, thread 1 computes `2+20=22`, and so on, all at once.

### File: `puzzle_02_vector_add.mojo`

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

### Expected Output

```
=== Element-wise vector addition ===

Output vector (a + b):
output[ 0 ] = 11.0
output[ 1 ] = 22.0
output[ 2 ] = 33.0
output[ 3 ] = 44.0
```

Subtraction is the same kernel with `output[i] = a[i] - b[i]` — on the same vectors, `[10-1, 20-2, 30-3, 40-4] = [9, 18, 27, 36]`. The interesting engineering question here is never the arithmetic, it's the launch configuration: this example uses one block of `SIZE` threads because `SIZE` is tiny. For a production-sized vector of, say, one million elements, you tile it instead: pick `THREADS_PER_BLOCK = 256` and compute `BLOCKS_PER_GRID = (1_000_000 + 255) // 256 = 3907` blocks, with the same `if i < SIZE` bounds check protecting the tail block (which covers indices `999,936` through `1,000,191`, running 256 threads but only 64 of them landing inside the real data) from writing past the buffer.

**Local derivative, for Chapter 7:** `∂(a+b)/∂a = 1` and `∂(a+b)/∂b = 1` — a change to either input passes straight through to the output, unscaled.

## 3.2 Multiplication and Division

Multiplication and division follow the identical one-thread-per-element pattern, on data where the arithmetic matters more directly: take `a = [2, 3, 4]` and `b = [5, 6, 7]`. Element-wise multiply gives `[2×5, 3×6, 4×7] = [10, 18, 28]`; element-wise divide gives `[2/5, 3/6, 4/7] = [0.4, 0.5, 0.571...]`.

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
        # trap -- the autograd engine in Part 4 checks for both
        # before accepting a gradient into an optimizer step.
        output[i] = a[i] / b[i]
```

**Local derivatives, worked on the numbers above.** For `c = a * b` at position 0 (`a=2, b=5, c=10`): `∂c/∂a = b = 5` and `∂c/∂b = a = 2` — nudge `a` from `2` to `2.01` and `c` moves from `10` to `2.01×5=10.05`, a change of `0.05` over a nudge of `0.01`, giving a slope of `5`, matching `∂c/∂a = b = 5` exactly. For `c = a / b`: `∂c/∂a = 1/b` and `∂c/∂b = -a/b²`; at position 0 that's `∂c/∂a = 1/5 = 0.2` and `∂c/∂b = -2/25 = -0.08`. Both backward kernels Chapter 7.2 builds from these rules are one more elementwise pass over the same-shaped buffers — no new infrastructure, just a different kernel body multiplying by whichever of these local derivatives applies.

## 3.3 Power and Exponential Functions

Power and exponential are the two element-wise operations that show up in every loss function and activation in Part 6, so they're worth their own kernels rather than a generic "apply scalar function" abstraction — a dedicated kernel lets the compiler specialize and inline the `exp`/`pow` intrinsic. Take `x = [1, 2, 3]`: squaring (`pow(x, 2)`) gives `[1, 4, 9]`; `exp(x)` gives `[e¹, e², e³] ≈ [2.71828, 7.38906, 20.0855]`.

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

**Local derivatives, worked at `x=2`.** For `pow(x, n)`, `d/dx[xⁿ] = n·x^(n-1)`; with `n=2` that's `2·x = 2×2 = 4` — check it: `pow(2.01, 2) = 4.0401`, and `(4.0401 - 4.0) / 0.01 = 4.01 ≈ 4`. For `exp(x)`, `d/dx[eˣ] = eˣ`, meaning the derivative *is* the function itself — at `x=2` that's `e² ≈ 7.38906`, exactly the forward value already computed. This self-referencing property is exactly why [Chapter 7.2](../part4/01-backward-function-implementation.md#72-element-wise-operation-gradients)'s `ExpOp.backward` reads the cached `output` tensor rather than recomputing `exp(x)` a second time during backward.

## 3.4 Broadcasting Implementation

Every operation so far assumed both inputs were the same shape. Broadcasting is what lets `Tensor[2, 3] + Tensor[1, 3]` work anyway, by treating the smaller tensor's missing dimension as if it were silently repeated. Take a concrete pair: `A` is a 2×3 matrix, `B` is a single row of 3 values meant to be added to *every* row of `A`:

```
A = 1  2  3        B = 10  20  30
    4  5  6
```

Broadcasting `A + B` conceptually repeats `B` down to match `A`'s shape before adding — `[[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]] = [[11, 22, 33], [14, 25, 36]]` — but the framework never actually copies `B`; instead it reads the *same* row of `B` twice, once per row of `A`, using the stride-0 trick already implemented as pure shape logic in [1.2.5](../part1/02-memory-layout-design.md#part-125-broadcasting-layout-preparation). Setting a tensor's stride to `0` along a dimension means "don't advance the memory address at all as this dimension's index increases" — which is exactly "keep re-reading the same values."

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

Trace what `b_stride_row = 0` does on the example above: for row `0`, the kernel reads `b[0*0 + col] = b[col]`; for row `1`, it reads `b[1*0 + col] = b[col]` — the *same* address, `B`'s one and only row, both times. That's the entire mechanism: `a_stride_row = 0` would make every column of the grid re-read row `0` of `A` instead, a virtual broadcast with no extra memory traffic beyond what a same-shaped add would already cost, which is the entire point of doing broadcasting at the stride level instead of the copy level.

Chapter 4 moves from per-element operations to operations that mix elements across a whole row or column: matrix multiplication, transpose, and the general tensor contraction that both generalize from.
