# Chapter 4: Matrix Operations

## 4.1 Matrix Multiplication

Matrix multiplication is tensor contraction restricted to two dimensions, so it's worth deriving the general rule once here before Chapter 4 needs the higher-dimensional version for batched operations in Part 6.

**The fundamental rule of tensor contraction:** the resulting tensor's dimensions are the concatenation of the two input tensors' dimensions, *without* the dimensions listed in the contraction pairs.

Worked example: contract tensor `X` with shape `{2, 3}` against tensor `M` with shape `{3, 2}`, using contraction pair `{1, 0}` — `X`'s dimension 1 contracts with `M`'s dimension 0. This is exactly `X @ M`.

```
X (2×3):                    M (3×2):
┌         ┐                 ┌     ┐
│ 1  2  3 │  ← row 0        │ 1 2 │  ← row 0 (contracts with X col 0)
│ 4  5  6 │  ← row 1        │ 3 4 │  ← row 1 (contracts with X col 1)
└         ┘                 │ 5 6 │  ← row 2 (contracts with X col 2)
                             └     ┘
```

Remaining dimensions after removing the contracted pair: `X`'s dimension 0 (size 2) and `M`'s dimension 1 (size 2) → output shape `{2, 2}`. For every output element:

```
Y(i,j) = X(i,0)·M(0,j) + X(i,1)·M(1,j) + X(i,2)·M(2,j)
```

```
Y(0,0) = 1·1 + 2·3 + 3·5 = 1 + 6 + 15  = 22
Y(0,1) = 1·2 + 2·4 + 3·6 = 2 + 8 + 18  = 28
Y(1,0) = 4·1 + 5·3 + 6·5 = 4 + 15 + 30 = 49
Y(1,1) = 4·2 + 5·4 + 6·6 = 8 + 20 + 36 = 64

Y (2×2):
┌         ┐
│ 22  28  │
│ 49  64  │
└         ┘
```

That triple-nested sum is the scalar baseline every matmul implementation is checked against:

```mojo
struct Matrix:
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
        return self.data[row * self.cols + col]     # row-major, from Part 0.3

    fn set(self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn fill_identity(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.set(i, j, Float32(1.0) if i == j else Float32(0.0))

fn scalar_matrix_multiply(a: Matrix, b: Matrix, result: Matrix):
    """Classic O(n^3): the baseline SIMD and GPU versions are checked against."""
    for i in range(a.rows):
        for j in range(b.cols):
            var sum: Float32 = 0.0
            for k in range(a.cols):
                sum += a.get(i, k) * b.get(k, j)
            result.set(i, j, sum)
```

And the SIMD-vectorized inner product, which packs `simd_width` multiply-accumulates into a single vector instruction:

```mojo
fn simd_matrix_multiply[simd_width: Int](a: Matrix, b: Matrix, result: Matrix):
    for i in range(a.rows):
        for j in range(b.cols):
            var sum: Float32 = 0.0
            var simd_count = (a.cols // simd_width) * simd_width
            var vector_sum = SIMD[DType.float32, simd_width](0.0)

            for k in range(0, simd_count, simd_width):
                var a_vals = SIMD[DType.float32, simd_width](0)
                var b_vals = SIMD[DType.float32, simd_width](0)
                for l in range(simd_width):
                    a_vals[l] = a.get(i, k + l)
                    b_vals[l] = b.get(k + l, j)
                vector_sum += a_vals * b_vals   # single SIMD FMA

            for l in range(simd_width):
                sum += vector_sum[l]
            for k in range(simd_count, a.cols):  # remainder, scalar
                sum += a.get(i, k) * b.get(k, j)

            result.set(i, j, sum)
```

### Expected Output

Verified against a 64×64 sequential matrix `A` multiplied by an identity matrix `B` (so the correct answer is `A` itself):

```
Matrix operations on 64 x 64 matrices
Matrix A: Sequential values [0, 1, 2, ...]
Matrix B: Identity matrix

1. Matrix Multiplication (A * B):
  Scalar multiplication completed
  SIMD4 multiplication completed
  Verification: PASS

3. Sample Results (top-left 4x4):
  Original A * B (should equal A since B is identity):
    [0.0, 1.0, 2.0, 3.0]
    [64.0, 65.0, 66.0, 67.0]
    [128.0, 129.0, 130.0, 131.0]
    [192.0, 193.0, 194.0, 195.0]
```

## 4.2 Transpose Operations

Transpose (`output[j][i] = input[i][j]`) is a pure data-movement operation — no arithmetic, so its cost is entirely about memory access pattern. Reading a row of `input` sequentially while writing scattered elements into `output` columns is exactly the access pattern SIMD loads/stores and the SoA layout from Part 0.3 are meant to make cheap:

```mojo
fn matrix_transpose_simd[simd_width: Int](input: Matrix, output: Matrix):
    for i in range(input.rows):
        var simd_count = (input.cols // simd_width) * simd_width
        for j in range(0, simd_count, simd_width):
            var row_vals = SIMD[DType.float32, simd_width](0)
            for k in range(simd_width):
                row_vals[k] = input.get(i, j + k)
            for k in range(simd_width):
                output.set(j + k, i, row_vals[k])   # scatter into columns
        for j in range(simd_count, input.cols):      # remainder
            output.set(j, i, input.get(i, j))
```

Verification checks `input[i][j] == transposed[j][i]` over a submatrix — this is the same verification pattern (compare against a known-correct scalar baseline) used throughout the book, because a SIMD or GPU kernel that's fast but wrong is worse than one that's merely slow.

## 4.3 Reshaping and View Operations

Reshape is free when the tensor is contiguous: it doesn't touch a single byte of data, only the `TensorShape` and `TensorStrides` metadata built in [1.1](../part1/01-core-tensor-structure.md) and [1.2](../part1/02-memory-layout-design.md#part-121-stride-calculation-system). A `[2, 6]` tensor reshaping to `[3, 4]` is valid precisely because both shapes have the same total element count (12) and the underlying memory is one contiguous C-order run — the reshape just recomputes strides for the new shape and returns a `TensorView` sharing the original `RefCountedBuffer` from [Chapter 2](../part1/06-memory-management-system.md#21-reference-counting-implementation). Reshaping a *non-contiguous* view (for example, one already produced by a transpose) requires a copy, since there's no stride pattern that reproduces the requested shape's row-major layout over scattered memory — the framework detects this via the same contiguity check used in `TensorView.is_contiguous()` and falls back to `matrix_transpose_simd`-style data movement automatically.

## 4.4 Advanced Linear Algebra

The specialized tensor types from [1.3.4](../part1/04-specialized-tensor-types.md) exist specifically to make advanced linear algebra cheap: multiplying by a `DiagonalTensor` is O(n) instead of O(n²) because the framework dispatches on tensor *kind*, not just shape, and multiplying an `IdentityTensor` by anything returns the other operand unchanged in O(1) — which is exactly the verification trick Section 4.1 used above. Triangular tensors from the same section back forward/backward substitution, the standard way to solve `Ax = b` without an explicit, numerically unstable matrix inverse — a technique Part 7's bond-pricing chapter leans on when solving for a discount curve that reprices a whole set of instruments simultaneously.

Chapter 5 turns to the operations that collapse a tensor's dimensions instead of preserving them: sums, means, norms, and the reductions every loss function ultimately produces a scalar through.
