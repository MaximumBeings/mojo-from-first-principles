# 1.3.4 Specialized Tensor Types

Dense storage is the right default, but matrices with known structure should preserve that structure as data. A specialized tensor stores only independent values, implements direct indexed access, and converts to dense form only at an explicit boundary. The payoff is both memory and algorithmic clarity.

## 1.3.4A Identity tensors

An identity tensor is completely described by its size and scale. Storing `n²` zeros and ones would throw away the most valuable fact about it: every off-diagonal value is known without memory access.

```mojo
@fieldwise_init
struct IdentityTensor:
    var size: Int
    var scale: Float32

    def get(self, row: Int, col: Int) -> Float32:
        debug_assert(0 <= row and row < self.size)
        debug_assert(0 <= col and col < self.size)
        return self.scale if row == col else 0
```

**Manual worked example.** `IdentityTensor(3,2)` represents `[[2,0,0],[0,2,0],[0,0,2]]` using two fields. `get(1,1)=2`, while `get(1,2)=0`. Multiplying it by `[4,5,6]` gives `[8,10,12]`, so the scale is applied exactly once to each matching coordinate.

## 1.3.4B Diagonal tensors

A diagonal with offset `k` stores one one-dimensional buffer. For `k=0`, coordinate `(i,i)` maps to buffer index `i`; a positive offset stores entries above the main diagonal and a negative offset stores entries below it.

```mojo
struct DiagonalTensor:
    var size: Int
    var offset: Int
    var values: List[Float32]

    def get(self, row: Int, col: Int) -> Float32:
        if col - row != self.offset:
            return 0
        var index = row if self.offset >= 0 else col
        return self.values[index]
```

**Manual worked example.** For size 4, offset `+1`, and values `[5,6,7]`, the stored coordinates are `(0,1)=5`, `(1,2)=6`, and `(2,3)=7`. `get(1,2)` computes `2-1=1`, then reads index 1 and returns 6. `get(1,1)` is off the selected diagonal and returns 0.

## 1.3.4C Sparse COO tensors

Coordinate format stores one row, column, and value per nonzero. It is ideal for incremental construction and teaching because every stored item is visible. Production arithmetic often canonicalizes duplicates and converts to CSR/CSC for faster traversal.

```mojo
struct SparseCOO:
    var rows: Int
    var cols: Int
    var row_index: List[Int]
    var col_index: List[Int]
    var values: List[Float32]

    def append(mut self, row: Int, col: Int, value: Float32):
        debug_assert(0 <= row and row < self.rows)
        debug_assert(0 <= col and col < self.cols)
        if value != 0:
            self.row_index.append(row)
            self.col_index.append(col)
            self.values.append(value)
```

**Manual worked example.** Append `(0,1,5)` and `(2,0,-2)` to a 3×3 tensor. The three arrays become rows `[0,2]`, columns `[1,0]`, and values `[5,-2]`. Dense reconstruction is `[[0,5,0],[0,0,0],[-2,0,0]]`; only 2 of 9 coordinates are stored.

## 1.3.4D Packed triangular tensors

A lower triangular matrix stores rows of lengths 1 through `n`. The number of stored values is the triangular number `n(n+1)/2`, and coordinate `(row,col)` maps after every value in earlier rows.

```mojo
def lower_packed_index(row: Int, col: Int) -> Int:
    debug_assert(0 <= col and col <= row)
    return row * (row + 1) // 2 + col
```

**Manual worked example.** A 4×4 lower triangle stores `4×5/2=10` values. Coordinate `(3,2)` begins after rows 0–2, which contain `1+2+3=6` values, then advances two positions: `6+2=8`. The formula gives `3×4/2+2=8`, matching the hand count.

## 1.3.4E Dispatch without premature densification

Specialized operations should exploit algebraic structure. Multiplying by identity scales, multiplying by a diagonal scales rows or columns, and triangular solves use substitution. Densification is a fallback, not the default implementation.

```mojo
def identity_matvec(identity: IdentityTensor, x: List[Float32]) -> List[Float32]:
    debug_assert(len(x) == identity.size)
    var out = List[Float32](capacity=len(x))
    for value in x:
        out.append(identity.scale * value)
    return out
```

**Manual worked example.** With scale 2 and `x=[4,5,6]`, the loop appends `2×4=8`, `2×5=10`, and `2×6=12`. No 3×3 dense matrix is allocated and no zero is multiplied, yet the answer matches dense identity multiplication exactly.

These representations establish a general rule for the rest of the framework: keep mathematical structure in the type until an operation truly requires a generic dense buffer.
