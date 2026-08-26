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

Transpose (`output[j][i] = input[i][j]`) is a pure data-movement operation — no arithmetic at all, so its cost is entirely about memory access pattern rather than compute. Worked on a small, asymmetric example so rows and columns can't be confused: a 2×3 matrix

```
      1  2  3
A =
      4  5  6
```

transposes to the 3×2 matrix formed by turning A's rows into columns:

```
       1  4
Aᵀ =   2  5
       3  6
```

Read off `A[0,2]=3` and check it lands at `Aᵀ[2,0]=3` — it does, because transposing simply swaps which index is "row" and which is "column" for every element, without changing a single value. Reading a row of `input` sequentially while writing scattered elements into `output` columns is exactly the access pattern SIMD loads/stores and the SoA layout from Part 0.3 are meant to make cheap:

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

Reshape is free when the tensor is contiguous: it doesn't touch a single byte of data, only the `TensorShape` and `TensorStrides` metadata built in [1.1](../part1/01-core-tensor-structure.md) and [1.2](../part1/02-memory-layout-design.md#part-121-stride-calculation-system). Take the twelve values `[0,1,2,...,11]` sitting in one contiguous run of memory. Viewed as `[2, 6]` (row-major, Part 0.3), that memory reads as

```
0  1  2  3  4  5
6  7  8  9 10 11
```

Reshaping to `[3, 4]` re-slices the *exact same twelve bytes* into a different grid, with no data movement at all:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
```

Both are valid readings of one flat sequence, `0` through `11`, because `2×6 = 3×4 = 12` and the underlying memory is one contiguous C-order run — the reshape just recomputes strides for the new shape (row stride `6` becomes row stride `4`) and returns a `TensorView` sharing the original `RefCountedBuffer` from [Chapter 2](../part1/06-memory-management-system.md#21-reference-counting-implementation). Reshaping a *non-contiguous* view (for example, one already produced by a transpose) requires a genuine copy instead: `Aᵀ` from Section 4.2 above is laid out in memory as `[1,4,2,5,3,6]`, and there is no stride pattern that re-slices *that* order into `[3,4]`'s expected row-major reading of `0..11` — the framework detects this via the same contiguity check used in `TensorView.is_contiguous()` and falls back to `matrix_transpose_simd`-style data movement automatically.

## 4.4 Advanced Linear Algebra

The specialized tensor types from [1.3.4](../part1/04-specialized-tensor-types.md) exist specifically to make advanced linear algebra cheap, and the saving is easiest to see by counting multiplications by hand. A general `3×3` matrix times a `3×3` matrix is 27 scalar multiplications (Section 4.1's triple loop: 3 output rows × 3 output columns × 3 terms summed per entry). Now make the second matrix diagonal — say `D = diag(2, 5, 10)`, meaning `2, 5, 10` on the diagonal and `0` everywhere else. Multiplying any matrix `A` by `D` just scales `A`'s columns: column `0` of the result is column `0` of `A` times `2`, column `1` is column `1` of `A` times `5`, column `2` is column `2` of `A` times `10` — 9 multiplications total (one per entry of `A`, since each entry is scaled by exactly one diagonal value), not 27. A `DiagonalTensor` dispatches to this column-scaling code instead of the general matmul path because the framework tracks tensor *kind*, not just shape — the 3× saving here becomes an `n`-times saving (`O(n)` instead of `O(n²)` multiplications) as the matrix grows. Push the same idea to its limit with the identity matrix — `diag(1, 1, 1)` — and every scale factor is `1`, so multiplying by it returns the other operand completely unchanged, in `O(1)`: exactly the verification trick Section 4.1 used to check that `A @ I = A`.

Triangular tensors from the same section back forward/backward substitution, the standard way to solve `Ax = b` without an explicit, numerically unstable matrix inverse — a technique Part 7's bond-pricing chapter leans on when solving for a discount curve that reprices a whole set of instruments simultaneously.

Chapter 5 turns to the operations that collapse a tensor's dimensions instead of preserving them: sums, means, norms, and the reductions every loss function ultimately produces a scalar through.
