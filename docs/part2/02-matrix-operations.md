# Chapter 13: Matrix Operations — When One Output Depends on More Than One Input

> "Every kernel in the last chapter answered its question by looking at exactly one input position — thread `i` read `a[i]` and `b[i]` and nothing else existed as far as it was concerned. Matrix multiplication is the first place in this book where that stops being true. Computing a single output element means reading an entire row and an entire column, multiplying them together position by position, and the moment the shapes stop lining up, the loop keeps running anyway — it just reads whatever happens to be sitting in memory next and calls that the answer."

**What you will understand by the end of this chapter:**

- The tensor-contraction rule that matrix multiplication is a special case of, worked out completely by hand on one running example (`X` (2×3) times `M` (3×2)) that every subsequent worked example in this chapter checks its own answer against
- Exactly how `simd_matrix_multiply`'s blocking-plus-remainder loop reduces to the identical scalar triple sum — traced lane by lane, for a vector width that divides the contraction dimension evenly and one that doesn't
- Why neither `scalar_matrix_multiply` nor `simd_matrix_multiply` ever checks that `a.cols == b.rows`, and precisely what ends up in `result` when that invariant is silently violated
- The difference between a reshape that's free (same buffer, new strides, zero bytes moved) and one that requires an actual copy — traced concretely on `A`'s own transpose, including what a *naive* reshape of that transpose would wrongly produce if the framework didn't check for contiguity first
- Why multiplying by a diagonal or identity tensor from Chapter 9 does asymptotically less work than the general path, counted exactly rather than asserted — and how the same accounting extends to solving a triangular system without ever forming a matrix inverse

**What you need to know first:**

- Chapter 6 (`TensorShape` and `TensorStrides` — the linear-index formula `Matrix.get` below reimplements for exactly two dimensions, with a hand-rolled `row * cols + col` in place of the general dot-product-with-strides version)
- Chapter 7.1 and 7.2 (stride math and zero-copy views — Section 13.3's "reshape is free" argument is the same "recompute strides, touch no data" idea Chapter 7 built, applied to a concrete pair of shapes)
- Chapter 9 (the `IdentityTensor`, `DiagonalTensor`, and triangular structures Section 13.4 dispatches to instead of paying for the general `O(n³)` path)
- Chapter 5 (SIMD width and the main-loop-plus-remainder shape — `simd_matrix_multiply` and `matrix_transpose_simd` both reuse that exact shape unchanged, just with a dot product and a scatter-write in place of Chapter 5's simpler per-element work)

<a id="41-matrix-multiplication"></a>
## 13.1 Matrix Multiplication: Tensor Contraction in Two Dimensions `[FOUNDATIONAL]`

### Intuition

Element-wise addition, Chapter 12's whole subject, never needed to know a shape beyond "how many positions are there" — every output position was a pure function of the *one* input position sitting at the same address. Matrix multiplication breaks that immediately: `Y(i,j)` needs an entire row of `X` and an entire column of `M`, and it needs them to line up — the row's third entry paired with the column's third entry, not its first. That pairing-and-summing operation has a name outside of matrices entirely: **tensor contraction**. A contraction takes two tensors, a shared "contracted" dimension between them, and produces one tensor whose shape is everything *except* that shared dimension, with every remaining position computed by summing products over every value the contracted dimension can take. Ordinary 2-D matrix multiplication is what a contraction looks like when both tensors happen to be rank 2 and exactly one dimension from each is contracted — a special case now, but the exact same machinery a later chapter's batched, higher-rank operations reuse without any new rule.

```
X (2×3):                    M (3×2):
┌         ┐                 ┌     ┐
│ 1  2  3 │  ← row 0        │ 1 2 │  ← row 0 (contracts with X col 0)
│ 4  5  6 │  ← row 1        │ 3 4 │  ← row 1 (contracts with X col 1)
└         ┘                 │ 5 6 │  ← row 2 (contracts with X col 2)
                             └     ┘
```

Contracting `X`'s dimension 1 (size 3) against `M`'s dimension 0 (size 3) — the two dimensions have to agree in size, since every value on one side needs a matching value on the other to multiply against — leaves `X`'s dimension 0 (size 2) and `M`'s dimension 1 (size 2) as the output's shape: `{2, 2}`. Every output position is one full dot product:

```
Y(i,j) = X(i,0)·M(0,j) + X(i,1)·M(1,j) + X(i,2)·M(2,j)
```

### Background

Two implementations of that same triple sum appear in this section: a scalar baseline that is the ground truth every faster version gets checked against, and a SIMD-vectorized version that packs several of the sum's terms into one vector instruction.

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
        return self.data[row * self.cols + col]     # row-major, from Chapter 3

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

`Matrix.get`'s `row * self.cols + col` is Chapter 6's linear-index formula specialized to exactly two dimensions — no general stride array, because a plain 2-D matrix only ever needs one stride (`cols`) and an implicit second stride of `1`. `scalar_matrix_multiply` is nothing more than the contraction formula above, written as three nested loops: the outer two pick an output position `(i, j)`, the innermost walks the contracted dimension and accumulates.

The SIMD version restructures only the innermost loop — it still computes exactly the same sum, just several terms at a time:

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

This is exactly Chapter 5's main-loop-plus-remainder shape, transplanted into the innermost dimension of a dot product instead of a flat vector: `simd_count` rounds `a.cols` down to the nearest multiple of `simd_width`, the main loop packs `simd_width` values from `a`'s row and `b`'s column into two SIMD registers and multiplies them in one instruction, and whatever doesn't divide evenly falls through to the scalar remainder loop at the bottom — the same tail-handling logic Chapter 5 traced for a flat vector, now handling the leftover terms of one dot product.

| Implementation | Multiply-adds per output element | Vector instructions per output element (width `w`) | Checked against |
|---|---|---|---|
| `scalar_matrix_multiply` | `a.cols` | 0 — fully scalar | the definition of matrix multiplication itself |
| `simd_matrix_multiply[w]` | `a.cols` (identical total work) | `a.cols // w` fused multiplies, plus `a.cols % w` scalar remainder terms | `scalar_matrix_multiply`'s output, element-by-element, within a `0.001` tolerance |
| A GPU tiled kernel | `a.cols`, split across threads/blocks instead of SIMD lanes | — | not introduced until Part 5's kernel-implementation chapter |

### Worked Example 13.1.1 — The full contraction, by hand

For `i=0, j=0`: `Y(0,0) = 1·1 + 2·3 + 3·5 = 1 + 6 + 15 = 22`. For `i=0, j=1`: `Y(0,1) = 1·2 + 2·4 + 3·6 = 2 + 8 + 18 = 28`. For `i=1, j=0`: `Y(1,0) = 4·1 + 5·3 + 6·5 = 4 + 15 + 30 = 49`. For `i=1, j=1`: `Y(1,1) = 4·2 + 5·4 + 6·6 = 8 + 20 + 36 = 64`.

```
Y (2×2):
┌         ┐
│ 22  28  │
│ 49  64  │
└         ┘
```

Every one of these four values is what `scalar_matrix_multiply(X, M, Y)` computes, and every other implementation in this chapter — SIMD, and eventually GPU — exists purely to reach this same answer faster.

### Worked Example 13.1.2 — The SIMD dot product, lane by lane

Trace `simd_matrix_multiply[2]` (`simd_width = 2`) computing `Y(0,0)` on the same `X` and `M`. `a.cols = 3`, so `simd_count = (3 // 2) * 2 = 2` — the contraction dimension does *not* divide evenly by `2`, which is exactly what makes the remainder loop worth watching here.

The main loop runs once, at `k = 0`: `a_vals = [X.get(0,0), X.get(0,1)] = [1, 2]`, `b_vals = [M.get(0,0), M.get(1,0)] = [1, 3]`. The single SIMD instruction `vector_sum += a_vals * b_vals` computes `[1×1, 2×3] = [1, 6]` in one shot, so `vector_sum = [1, 6]` — there is no second main-loop iteration, since `k = 2` is not less than `simd_count = 2`. Summing the two lanes: `sum = 1 + 6 = 7`.

The remainder loop then runs for `k` in `range(2, 3)` — just `k = 2`: `sum += X.get(0,2) * M.get(2,0) = 3 × 5 = 15`. Final total: `sum = 7 + 15 = 22` — exactly `Y(0,0)` from Worked Example 13.1.1, reached by one vectorized multiply covering two of the three terms and one scalar multiply covering the third.

A second trace, `Y(1,1)`, confirms the pattern rather than being a coincidence of the first: `a_vals = [X.get(1,0), X.get(1,1)] = [4, 5]`, `b_vals = [M.get(0,1), M.get(1,1)] = [2, 4]`, so `vector_sum = [4×2, 5×4] = [8, 20]`, lane sum `= 28`. Remainder: `X.get(1,2) × M.get(2,1) = 6 × 6 = 36`. Total: `28 + 36 = 64` — `Y(1,1)`, again matching the scalar answer exactly. Every output position in this chapter's example takes the identical two-vectorized-plus-one-scalar shape, because every position shares the same `a.cols = 3` contraction length.

### Worked Example 13.1.3 — Multiplication counts at three scales

The contraction formula performs exactly `a.cols` multiplications per output position, and there are `a.rows × b.cols` output positions, so the total scalar multiply count is `a.rows × a.cols × b.cols`. For this chapter's running `2×3` by `3×2` example: `2 × 3 × 2 = 12` multiplications total — four output positions, three each. For a square `n×n` by `n×n` case, that's `n × n × n = n³`: at `n = 3`, `27` multiplications, the exact figure Section 13.4 below uses as its own baseline. At `n = 64` — the size of this chapter's one captured test run — the same formula gives `64³ = 262{,}144` multiplications, none of which changes what `simd_matrix_multiply` computes, only how many vector instructions it takes to compute it: at `simd_width = 4`, roughly a quarter as many fused-multiply instructions as `scalar_matrix_multiply` would issue, for the identical `262{,}144` total multiply-adds.

```
[COMMON TRAP]  Neither multiply function ever checks a.cols == b.rows

scalar_matrix_multiply and simd_matrix_multiply both use a.cols as the
number of terms to sum, and both call b.get(k, j) for every k up to
a.cols - 1. Nothing in either function compares a.cols against b.rows
first. If a caller passes an `a` that is 2x3 and a `b` that is only 2x2
(so b.rows = 2, not 3), the loop still runs k = 0, 1, 2 regardless,
because it never looked at b.rows at all. b.get(2, j) then evaluates
b.data[2 * b.cols + j] -- an index built entirely from b's own row-major
formula, with no awareness that row 2 doesn't logically exist for a
2-row matrix. If b's underlying buffer happens to be large enough, this
silently reads whatever value sits past b's intended data (often the
start of some unrelated allocation) and folds it into the sum as if it
were a legitimate matrix entry -- no crash, no exception, just a wrong
number in `result` that looks exactly as valid as a correct one. If
b's buffer is not large enough, it's an out-of-bounds read instead,
which may or may not fault depending on what else happens to be mapped
nearby. Either way, the failure mode is silent-wrong-answer-or-maybe-a-
crash, never "an error message telling you the shapes don't match" --
that check simply does not exist in this code, unlike e.g. NumPy, which
raises immediately on a shape mismatch.
```

## 13.2 Transpose Operations: Rearranging Indices, Not Values `[FOUNDATIONAL]`

### Intuition

Transpose is the first operation in this book that changes *nothing* about any value and *everything* about where that value lives. `output[j][i] = input[i][j]` swaps which index means "row" and which means "column" — every number in the matrix survives untouched, only its address changes. That makes transpose a pure memory-movement problem: there is no arithmetic to get wrong, only an access pattern to get efficient, which is exactly the kind of problem Chapter 3's bus-utilization argument and Chapter 5's SIMD loads/stores were built to address.

### Background

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

The access pattern is deliberately asymmetric: `input.get(i, j+k)` reads `simd_width` *consecutive* values from one row of `input` — a single, cheap, sequential SIMD load, exactly the pattern the SoA argument in Chapter 3 favors — and then `output.set(j+k, i, ...)` writes those same values to `simd_width` *different rows* of `output`, one row apart each time, which is a scatter rather than a sequential store. Reading fast and writing scattered (rather than the other way around) is a deliberate choice: sequential reads are the cheaper resource to protect, since a strided or scattered *read* would otherwise waste part of every cache line fetched.

### Worked Example 13.2.1 — Transposing a 2×3 matrix by hand

```
      1  2  3
A =
      4  5  6
```

transposes to the 3×2 matrix formed by turning `A`'s rows into columns:

```
       1  4
Aᵀ =   2  5
       3  6
```

Spot-check one element rather than trusting the picture: `A[0,2] = 3`. Transposing swaps the two indices, so that value should land at `Aᵀ[2,0]` — and it does: `3`. Every element in the matrix obeys the identical rule, `Aᵀ[j,i] = A[i,j]`, with no value ever recomputed, only relocated.

### Worked Example 13.2.2 — The SIMD scatter, traced write by write

Run `matrix_transpose_simd[2]` on the same `2×3` `A`. `input.cols = 3`, `simd_width = 2`, so `simd_count = (3 // 2) × 2 = 2` — again a contraction-style dimension that doesn't divide evenly, so both the vectorized loop and the remainder loop fire.

For `i = 0` (row `[1, 2, 3]`): the main loop's only iteration is `j = 0`. `row_vals = [A.get(0,0), A.get(0,1)] = [1, 2]`, loaded as one sequential pair. The scatter then writes `output.set(0, 0, 1)` and `output.set(1, 0, 2)` — two separate rows of `output`, one write apart. The remainder loop handles `j = 2`: `output.set(2, 0, A.get(0,2)) = output.set(2, 0, 3)`.

For `i = 1` (row `[4, 5, 6]`): main loop `j = 0`: `row_vals = [4, 5]`, scattered to `output.set(0, 1, 4)` and `output.set(1, 1, 5)`. Remainder `j = 2`: `output.set(2, 1, 6)`.

Collecting every write: `output[0] = [1, 4]`, `output[1] = [2, 5]`, `output[2] = [3, 6]` — exactly `Aᵀ` from Worked Example 13.2.1, assembled from one sequential 2-wide read per row plus one scalar remainder read per row, scattered into six individual column writes total.

## 13.3 Reshaping and View Operations `[FOUNDATIONAL]`

### Intuition

Reshape asks a deceptively simple question: can these same bytes be read as a different shape without moving any of them? The answer is yes exactly when the data is *contiguous* in the new shape's row-major order — and the honest, sometimes-surprising answer is no otherwise, in which case "reshape" secretly becomes "copy."

### Background

Reshape recomputes only the `TensorShape` and `TensorStrides` metadata built in Chapter 6 and Chapter 7.1, and returns a new `TensorView` sharing the *original* `RefCountedBuffer` from Chapter 11.1 — no allocation, no data movement, just new arithmetic for turning a coordinate into an offset. That's only valid when the requested shape is a genuine re-slicing of the same flat sequence of values in the same order; a non-contiguous view (one already produced by a transpose, a stride-0 broadcast, or a slice) has no stride pattern that can make reshape's cheap trick work, and the framework falls back to `matrix_transpose_simd`-style data movement — a real copy — via the same contiguity check `TensorView.is_contiguous()` performs before every reshape.

### Worked Example 13.3.1 — A free reshape

Take twelve values `[0, 1, 2, ..., 11]` sitting in one contiguous run of memory. Viewed as shape `[2, 6]` (row-major, Chapter 3), that memory reads as:

```
0  1  2  3  4  5
6  7  8  9 10 11
```

Reshaping to `[3, 4]` re-slices the *exact same twelve bytes* into a different grid, with zero data movement:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
```

Both are valid readings of one flat sequence because `2×6 = 3×4 = 12` and the underlying memory is one contiguous run — reshape only recomputes the row stride (`6` becomes `4`) and hands back a view of the same buffer.

### Worked Example 13.3.2 — Why a non-contiguous reshape can't take the same shortcut

`Aᵀ` from Section 13.2 is stored in memory, element by element, as `[1, 4, 2, 5, 3, 6]` — the order `matrix_transpose_simd`'s scatter writes actually produced. Suppose reshape ignored the contiguity check and simply reinterpreted those six values as shape `[2, 3]` using ordinary row-major strides, the same shortcut Worked Example 13.3.1 used legitimately: row `0` would be the first three stored values, `[1, 4, 2]`, and row `1` would be the last three, `[5, 3, 6]`. Compare that to the real `A = [[1, 2, 3], [4, 5, 6]]` this data supposedly represents — they don't match in a single position past the first. The naive reshape isn't slightly wrong, it's wrong everywhere except `[0,0]`, because `Aᵀ`'s storage order was never `A`'s row-major order to begin with; it was written column-by-column by the transpose's scatter. This is precisely why `is_contiguous()` has to run *before* reshape takes the free path: the free path is only correct when the requested shape's row-major reading of the buffer matches what's actually stored, and a transposed buffer's storage order simply isn't that.

```
[COMMON TRAP]  A reshape that succeeds is still a view, not a copy

Even a legitimate, contiguous reshape -- like [2,6] to [3,4] in Worked
Example 13.3.1 -- returns a TensorView sharing the original
RefCountedBuffer, not an independent copy. Writing through the reshaped
view therefore writes through to the original tensor as well, and vice
versa, exactly as Chapter 7.2's zero-copy views intended. Code that
expects reshape to behave like a snapshot -- safe to mutate without
touching the tensor it came from -- will corrupt the original tensor's
data instead. When an independent copy is actually needed, that has to
be requested explicitly; reshape's entire reason to exist is to *avoid*
copying whenever the shapes allow it.
```

## 13.4 Advanced Linear Algebra: Structure-Aware Multiplication `[FOUNDATIONAL]`

### Intuition

Section 13.1's `O(n³)` cost assumes nothing about either matrix beyond its shape. The specialized tensor types from Chapter 9 exist specifically because assuming nothing is often wasteful — a matrix with known internal structure (all zeros except the diagonal, all zeros below the diagonal, or a single scale factor on the diagonal) can skip almost every multiplication the general path would blindly perform, simply by knowing in advance which terms are guaranteed to be zero and never computing them at all.

### Background

The saving is easiest to see by counting multiplications directly, the same accounting Worked Example 13.1.3 used for the general case. A general `3×3` matrix `A` times a general `3×3` matrix costs `27` scalar multiplications — Section 13.1's triple loop, `3` output rows times `3` output columns times `3` terms summed per entry. Now make the second matrix a `DiagonalTensor`, `D = diag(2, 5, 10)` — `2, 5, 10` on the diagonal, `0` everywhere else. Multiplying `A` by `D` just scales `A`'s columns: column `0` of the result is column `0` of `A` times `2`, column `1` is column `1` of `A` times `5`, column `2` is column `2` of `A` times `10`. That's `9` multiplications total — one per entry of `A`, since each entry is scaled by exactly one diagonal value — not `27`. A `DiagonalTensor` dispatches to this column-scaling code instead of the general matmul path because the framework tracks tensor *kind*, not just shape; the `3×` saving here becomes an `n`-times saving (`O(n)` instead of `O(n²)` multiplications, since the general path's `O(n³)` drops one full factor of `n`) as the matrix grows.

Push the idea to its limit with the identity matrix — `diag(1, 1, 1)`, Chapter 9's `IdentityTensor` — and every scale factor is `1`, so multiplying by it returns the other operand completely unchanged, in `O(1)`: zero multiplications needed at all, which is exactly the verification trick Worked Example 13.1.1's real captured test used to check that `A @ I = A` on a full `64×64` matrix.

Triangular tensors from the same chapter back **forward and backward substitution** — the standard way to solve `Ax = b` without ever forming a numerically unstable matrix inverse. This is illustrative rather than book source code, since no runnable forward-substitution function appears in this chapter's own material, but the arithmetic is worth working through by hand because Part 7's bond-pricing chapter leans on exactly this technique to solve for a discount curve that reprices a whole set of instruments simultaneously.

### Worked Example 13.4.1 — Counting the saving at n = 5

Using the same accounting on a larger case: a general `5×5` by `5×5` multiplication costs `5³ = 125` scalar multiplications. Dispatching to the diagonal path instead costs `1` multiplication per entry of the `5×5` operand — `25` total. The ratio is `125 / 25 = 5`, matching the `n`-times pattern exactly: at `n = 3` the saving was `3×` (`27` vs `9`), and at `n = 5` it's `5×` (`125` vs `25`) — the general path's cost grows as `n³` while the diagonal path's grows only as `n²`, so the ratio between them is always exactly `n`.

### Worked Example 13.4.2 — Solving a triangular system by forward substitution

Take a small lower-triangular system `Lx = b` with `L = [[2, 0], [3, 4]]` and `b = [6, 23]`. Forward substitution solves the *first* row first, since it only involves one unknown: `2·x₀ = 6`, so `x₀ = 3`. The second row now has only one unknown left, because `x₀` is already known: `3·x₀ + 4·x₁ = 23`, so `4·x₁ = 23 - 3×3 = 23 - 9 = 14`, giving `x₁ = 3.5`. Checking the full system: `L @ [3, 3.5] = [2×3 + 0×3.5, 3×3 + 4×3.5] = [6, 9 + 14] = [6, 23]` — matching `b` exactly. Notice what never happened: no matrix inverse was computed, and no general `O(n³)` solve was needed — each unknown was resolved from a single equation the moment every other unknown in that equation was already known, which is only possible because `L`'s triangular structure guarantees row `0` has exactly one unknown, row `1` has at most two, and so on.

## 13.5 Reference Implementations

Only one combined run of this chapter's code was ever captured with console output — a `64×64` sequential matrix `A` multiplied by a `64×64` identity matrix `B`, verifying the scalar and SIMD results against each other and against the mathematical fact that `A @ I = A`. The source material for this chapter includes the `Matrix` struct and the three functions above, and the resulting output, but not the driver/`main()` code that actually built the two `64×64` matrices and called these functions — unlike Chapters 6 through 12, where a complete file with its own `main()` was available to reproduce verbatim, this chapter's source only preserves the pieces and the result. Rather than invent a driver that was never actually part of the book's material, the functions are reproduced exactly as given below, followed by the real captured output they were verified against.

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
        return self.data[row * self.cols + col]     # row-major, from Chapter 3

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

Transpose's verification is described in the source only as checking `input[i][j] == transposed[j][i]` over a submatrix — the same "compare against a known-correct baseline" pattern used throughout this book — but no separate captured console output for that check survives alongside this chapter's material.

## Chapter Summary

Matrix multiplication is tensor contraction restricted to two dimensions: every output position is a full dot product between a row and a column, `a.cols` multiplications each, `a.rows × b.cols` positions total — `12` multiplications for this chapter's `2×3` by `3×2` running example, `27` for a general `3×3`, and `262{,}144` for the `64×64` case this chapter's one real captured run actually exercised. `simd_matrix_multiply` computes that identical sum, just packing `simd_width` terms into one vector instruction at a time and falling through to a scalar remainder loop for whatever doesn't divide evenly — traced lane by lane in this chapter down to the exact same `22` and `64` the scalar version produces. Neither multiply function ever checks that `a.cols` equals `b.rows` before running, which means a shape mismatch doesn't raise an error — it silently reads past where a matrix's real data ends and folds whatever it finds into the answer. Transpose moves values without changing any of them, reading sequentially and scattering into distant rows; reshape avoids moving values at all when the data is already contiguous in the target shape, and falls back to a real copy when it isn't — attempting a naive reshape of a transposed buffer without that contiguity check produces the wrong matrix in every position but one. Finally, the specialized tensor types from Chapter 9 turn "assume nothing about the operand" into "exploit exactly what's known": a diagonal operand cuts multiplication counts by a factor of `n`, an identity operand cuts them to zero, and a triangular operand replaces an unstable matrix inverse with forward or backward substitution — resolving one unknown per equation, in order, the way this chapter's `2×2` worked example resolved `x₀ = 3` and then `x₁ = 3.5` without ever inverting a matrix.

## Self-Check Questions

1. `simd_matrix_multiply[3]` is called on this chapter's running `X (2×3)` by `M (3×2)` example — that is, `simd_width` set to `3`, exactly equal to `a.cols`. What does `simd_count` evaluate to, and does the remainder loop ever execute for any output position?
2. Trace `simd_matrix_multiply[2]` computing `Y(1,0)` on the same `X` and `M` (row `1` of `X` is `[4,5,6]`; column `0` of `M` is `[1,3,5]`). What are `a_vals`, `b_vals`, the lane sum, and the remainder contribution — and does the total match the scalar answer, `49`, from Worked Example 13.1.1?
3. `scalar_matrix_multiply` is called with `a` shaped `2×3` and `b` shaped `2×2` (so `a.cols = 3` but `b.rows = 2`). Using `Matrix.get`'s formula `row * self.cols + col`, what does `b.get(2, j)` actually evaluate to, and why does nothing in the code stop this call from proceeding?
4. Using `matrix_transpose_simd[2]` on `A = [[1,2,3],[4,5,6]]`, which exact `output.set` calls (in order) reconstruct row `1` of `A` (`[4,5,6]`) into column `1` of `output`?
5. A general `4×4` matrix is multiplied by a `4×4` diagonal matrix. Using the counting method from Worked Example 13.4.1, how many scalar multiplications does the general path take, how many does the diagonal-dispatch path take, and what ratio do they confirm?

## Where We Go Next

Chapter 14 (`part2/03-reduction-operations.md`) turns to the operations that collapse a tensor's dimensions instead of preserving them: sums, means, norms, and the reductions every loss function ultimately produces a single scalar through.

## Worked Solutions

**1.** `simd_count = (3 // 3) × 3 = 3`, exactly equal to `a.cols`. The remainder loop runs `for k in range(3, 3)` — an empty range — for every single output position, so it never executes at all when `simd_width` divides the contraction dimension evenly; the entire dot product is computed by one vectorized multiply per output element.

**2.** `a_vals = [X.get(1,0), X.get(1,1)] = [4, 5]`; `b_vals = [M.get(0,0), M.get(1,0)] = [1, 3]`; `vector_sum = [4×1, 5×3] = [4, 15]`, lane sum `= 4 + 15 = 19`. Remainder (`k=2`): `X.get(1,2) × M.get(2,0) = 6 × 5 = 30`. Total: `19 + 30 = 49` — matching the scalar answer for `Y(1,0)` exactly.

**3.** `b.get(2, j)` evaluates `b.data[2 * b.cols + j] = b.data[2×2 + j] = b.data[4 + j]`. For a `2×2` matrix (`4` total elements, valid indices `0`–`3`), this reads at index `4` or higher — one past the end of `b`'s allocated buffer, an out-of-bounds read. Nothing stops the call because neither `scalar_matrix_multiply` nor `Matrix.get` ever compares `a.cols` against `b.rows`; the loop bound is taken entirely from `a.cols`, with no reference to `b`'s actual dimensions at all.

**4.** Row `1` of `A` is `[4, 5, 6]`. The main loop (`j=0`, since `simd_count = 2`) loads `row_vals = [A.get(1,0), A.get(1,1)] = [4, 5]` and writes `output.set(0, 1, 4)` then `output.set(1, 1, 5)`. The remainder loop (`j=2`) writes `output.set(2, 1, A.get(1,2)) = output.set(2, 1, 6)`. In order: `output.set(0,1,4)`, `output.set(1,1,5)`, `output.set(2,1,6)` — reconstructing column `1` of the output as `[4, 5, 6]`, row `1` of `A`, exactly as transpose requires.

**5.** General path: `4³ = 64` multiplications. Diagonal-dispatch path: one multiplication per entry of the `4×4` operand — `16` total. Ratio: `64 / 16 = 4`, confirming the `n`-times pattern at `n = 4` — consistent with `3×` at `n=3` and `5×` at `n=5`.
