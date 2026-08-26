# Chapter 3: Element-wise Operations

With tensors, views, and memory management in place, Part 2 starts putting the infrastructure to work: real arithmetic on real data. Element-wise operations — addition, subtraction, multiplication, division, and broadcasting — are the simplest tensor operations and also the highest-frequency ones in any training loop, so this is where SIMD width and memory layout choices from Part 0 start paying off directly.

## 3.1 Addition and Subtraction

The clearest way to see element-wise addition is on the GPU, one thread per output element — this is essentially "Puzzle 2" from the GPU-programming exercises that motivated Part 0.4:

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

Subtraction is the same kernel with `output[i] = a[i] - b[i]`; the interesting engineering question is not the arithmetic, it's the launch configuration. This example uses one block of `SIZE` threads because `SIZE` is tiny. For a production-sized vector (millions of elements) you tile it: pick `THREADS_PER_BLOCK = 256` and compute `BLOCKS_PER_GRID = (n + 255) // 256`, with the same `if i < SIZE` bounds check protecting the tail block from writing past the buffer.

## 3.2 Multiplication and Division

Wired into the `Tensor` struct from Chapter 1, element-wise multiply/divide reuse the same kernel shape but need one more piece: a gradient-aware wrapper, since these are the operations Part 4's backward pass differentiates through.

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

The local derivatives worth remembering ahead of Part 4: for `c = a * b`, `dc/da = b` and `dc/db = a` (each input's gradient is a multiply against the *other* input, and against the incoming upstream gradient); for `c = a / b`, `dc/da = 1/b` and `dc/db = -a/b²`. Both backward kernels are one more elementwise pass over the same-shaped buffers — no new infrastructure, just a different kernel body.

## 3.3 Power and Exponential Functions

Power and exponential are the two element-wise operations that show up in every loss function and activation in Part 6, so they're worth their own kernels rather than a generic "apply scalar function" abstraction (Mojo can express that too, but a dedicated kernel lets the compiler specialize and inline the `exp`/`pow` intrinsic):

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

`d/dx[x^n] = n * x^(n-1)` and `d/dx[e^x] = e^x` — the exponential's derivative being itself is exactly why the forward pass caches `output` on the graph node in Part 3 rather than recomputing it during backward.

## 3.4 Broadcasting Implementation

Broadcasting lets `Tensor[2, 3] + Tensor[1, 3]` (or `+ scalar`) work without materializing a copy of the smaller operand. The broadcasting *rules* were already implemented as pure shape logic in [1.2.5](../part1/02-memory-layout-design.md#part-125-broadcasting-layout-preparation) — this section is where that shape logic drives an actual kernel, using stride-0 for any broadcast dimension so the same memory address is read by every thread that maps to it:

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

Passing `a_stride_row = 0` makes every row of the grid re-read row 0 of `a` — a virtual broadcast with no extra memory traffic beyond what a same-shaped add would already cost, which is the entire point of doing broadcasting at the stride level instead of the copy level.

Chapter 4 moves from per-element operations to operations that mix elements across a whole row or column: matrix multiplication, transpose, and the general tensor contraction that both generalize from.
