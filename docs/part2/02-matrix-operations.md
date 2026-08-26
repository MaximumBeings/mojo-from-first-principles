# Chapter 4: Matrix Operations

Matrix operations combine coordinates rather than treating them independently. The reference algorithms keep shapes and index arithmetic explicit; SIMD, tiling, and vendor kernels are interchangeable optimizations only after these answers are verified.

## 4.1 Matrix multiplication

For `C=A@B`, A's column count must equal B's row count. Each output is one dot product between a row of A and a column of B.

```mojo
def matmul(a: Tensor, b: Tensor) raises -> Tensor:
    var m = a.shape.dims[0]
    var k = a.shape.dims[1]
    var n = b.shape.dims[1]
    if b.shape.dims[0] != k:
        raise Error("matmul contraction dimensions differ")
    var out = Tensor(Shape([m, n]))
    for i in range(m):
        for j in range(n):
            for p in range(k):
                out.data[i * n + j] += a.data[i * k + p] * b.data[p * n + j]
    return out
```

**Manual worked example.** `[[1,2],[3,4]] @ [[5,6],[7,8]]` gives `[[1×5+2×7,1×6+2×8],[3×5+4×7,3×6+4×8]]=[[19,22],[43,50]]`.

## 4.2 Transpose changes the coordinate mapping

A materialized transpose swaps axes and copies `A[i,j]` to `out[j,i]`. A view can instead swap shape and strides without copying, as Section 1.2 showed.

```mojo
def transpose_2d(a: Tensor) raises -> Tensor:
    var rows = a.shape.dims[0]
    var cols = a.shape.dims[1]
    var out = Tensor(Shape([cols, rows]))
    for i in range(rows):
        for j in range(cols):
            out.data[j * rows + i] = a.data[i * cols + j]
    return out
```

**Manual worked example.** `[[1,2,3],[4,5,6]]` becomes `[[1,4],[2,5],[3,6]]`. Source `(1,2)` is 6 and maps to destination `(2,1)`, flat destination offset `2×2+1=5`.

## 4.3 Reshape is metadata-only only for contiguous storage

Reshape preserves the flat element order and requires equal element counts. A non-contiguous view must first be materialized in the requested logical order.

```mojo
def reshape_contiguous(mut tensor: Tensor, new_shape: Shape) raises:
    if tensor.shape.numel() != new_shape.numel():
        raise Error("reshape changes element count")
    tensor.shape = new_shape
    tensor.strides = row_major_strides(new_shape)
```

**Manual worked example.** Flat `[0,1,2,3,4,5]` reshaped from `[2,3]` to `[3,2]` reads `[[0,1],[2,3],[4,5]]`. Both shapes contain six elements; `[4,2]` would require eight and raises.

## 4.4 Linear solves beat explicit inverses

To solve `Ax=b`, factor or eliminate directly. Forming `A⁻¹` usually performs more work and amplifies rounding error. The 2×2 case makes the distinction concrete.

```mojo
def solve_2x2(a: Float64, b: Float64, c: Float64, d: Float64, y0: Float64, y1: Float64) raises -> Tuple[Float64, Float64]:
    var det = a * d - b * c
    if abs(det) < 1e-12:
        raise Error("singular matrix")
    return ((d * y0 - b * y1) / det, (-c * y0 + a * y1) / det)
```

**Manual worked example.** Solve `[[2,1],[1,3]]x=[5,7]`. Determinant is `2×3-1×1=5`; `x0=(3×5-1×7)/5=8/5=1.6`, `x1=(-1×5+2×7)/5=9/5=1.8`. Substitution gives `[2×1.6+1.8,1.6+3×1.8]=[5,7]`.

The backward rule for matmul in Chapter 7 is built from these same transpose and contraction operations.
