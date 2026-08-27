# Chapter 9: Specialized Tensor Types — Identity, Diagonal, Sparse, and Triangular

> "A dense tensor stores every number you could conceivably want. A specialized tensor stores only the numbers that make it what it is, and computes everything else from a rule instead of a lookup. The four structures in this chapter are four different bets about which rule is worth hard-coding — and, in one case, a bet whose own arithmetic doesn't do what it claims."

**What you will understand by the end of this chapter:**

- Why `IdentityTensor` needs no data buffer at all — the entire matrix is answered by a single `col - row == offset` comparison — and why that makes its memory usage exactly constant, not just small, at any size
- The genuine trade-off `DiagonalTensor` makes: real, size-proportional storage (unlike identity's O(1)) in exchange for direct O(1) access to any diagonal position, and where that trade-off actually loses to plain dense storage once fixed per-structure overhead is counted honestly
- The coordinate-list (COO) pattern behind `SparseTensorCOO`, why its element lookup is linear rather than constant time, and why its own memory accounting is dominated by a number (`capacity`) that has nothing to do with how sparse the tensor actually is
- How to derive, from first principles, the correct packed-storage index formula for a row-major upper triangular matrix — and how to use that derivation to catch a real off-by-one in `TriangularTensor`'s own formula, one that silently aliases distinct matrix positions onto the same storage slot
- Why memory efficiency numbers that look impressive in isolation (7%! 0.07%! 50%!) still need a size threshold attached before they mean anything, since every structure in this chapter has a break-even point below which "specialized" loses to "just use a dense matrix"

**What you need to know first:**

- Chapter 6 (row-major stride math, and `Tensor`'s dense, every-element-stored baseline that this chapter deliberately departs from)
- Chapter 7 (`_owns_data` and the RAII discipline — every struct below that allocates also frees, in `__del__`, exactly as before)
- Chapter 8.1 (`FactoryTensor`'s `eye` with a diagonal offset — that function *materializes* a shifted identity into a dense buffer; this chapter's `IdentityTensor` represents the identical mathematical object without ever allocating one)
- Chapter 2.4 (constructors, destructors, and manual memory as the load-bearing pattern behind every struct in this chapter)

## 9.1 Identity Tensors: When the Data Is the Rule `[FOUNDATIONAL]`

### Intuition

A mailroom that forwards every package addressed to room `N` straight to room `N` doesn't need a directory. It needs one sentence of policy — "room number in equals room number out" — and the sentence answers every possible query, for a building of 10 rooms or 10,000. Chapter 8's `eye()` built a shifted identity matrix the way a mailroom skeptical of that policy would: it allocated a full grid of pigeonholes, zeroed every one, and then wrote a `1` into the handful that were actually on the diagonal — a dense tensor that happens to be mostly zero. `IdentityTensor` in this chapter is the mailroom that trusts the policy. It stores three numbers — `size`, `scale`, and `offset` — and a validation flag, and answers every `get_item(row, col)` by checking one condition: does `col - row` equal the offset? If so, return `scale`. If not, return zero. No buffer, no allocation, no `__del__` to write, because there is no owned memory to free — `IdentityTensor` has no `UnsafePointer` field at all, the first struct in this book's tensor hierarchy that doesn't.

### Background

```mojo
struct IdentityTensor[dtype: DType]:
    var size: Int
    var scale: Scalar[dtype]
    var offset: Int  # Diagonal offset (0 = main diagonal, +n = upper, -n = lower)
    var _validation_enabled: Bool
```

| | `IdentityTensor` | `DenseTensor` (Ch. 6/8 style) |
|---|---|---|
| What's stored | 3 scalars + 1 bool, regardless of `size` | every one of `size × size` elements |
| Memory growth | O(1) — constant | O(n²) |
| `get_item` cost | one comparison: `col - row == offset` | one pointer dereference into a precomputed buffer |
| Can be mutated element-by-element? | no — there is no per-element storage to write into | yes |

That last row matters: `IdentityTensor` has no `set_item` at all. It has `scale_by` and `transpose`, which each return a *new* `IdentityTensor` built from a transformed `size`/`scale`/`offset`, because those are the only operations that stay expressible as "a size, a scale, and an offset." Anything that would require an individual element to differ from the rule — setting exactly one entry to a different value — falls outside what this struct can represent, and the codebase reaches for `DiagonalTensor` or `DenseTensor` instead. Specialization is a trade: you get free storage in exchange for giving up arbitrary mutation.

### Worked Example 9.1.1 — Memory that provably does not grow

`get_memory_usage()` returns `3 * sizeof[Int]() + sizeof[Scalar[dtype]]() + sizeof[Bool]()`. For `dtype = float32`, that's `3×8 + 4 + 1 = 29` bytes — and the test suite confirms it stays exactly `29` for a `10×10`, a `100×100`, and a `1000×1000` identity tensor alike, because `size` never appears in that formula at all:

| Size | `IdentityTensor` | Dense equivalent | Efficiency |
|---|---|---|---|
| 10×10 | 29 bytes | 400 bytes | 7.25% |
| 100×100 | 29 bytes | 40,000 bytes | 0.0725% |
| 1000×1000 | 29 bytes | 4,000,000 bytes | 0.000725% |

Each row down that table is a 10× jump in matrix side length, a 100× jump in dense bytes, and a flat line in identity bytes. The "efficiency" percentage isn't shrinking because `IdentityTensor` is doing progressively cleverer work — it's shrinking because the numerator is frozen at 29 and the denominator is the one actually growing. That's the entire content of "O(1) versus O(n²)" made concrete: not "identity is efficient," but "identity's storage is a constant that dense storage's quadratic term runs away from."

### Worked Example 9.1.2 — Diagonal length and transpose, traced through the offset

`get_diagonal_length()` returns `max(0, size - offset)` when `offset >= 0`, and `max(0, size + offset)` when `offset < 0`. Traced against the three sizes the test suite checks:

| Size | Main diag (offset 0) | Super-diag (offset +2) | Sub-diag (offset -1) |
|---|---|---|---|
| 3 | 3 | max(0, 3-2) = 1 | max(0, 3-1) = 2 |
| 5 | 5 | max(0, 5-2) = 3 | max(0, 5-1) = 4 |
| 10 | 10 | max(0, 10-2) = 8 | max(0, 10-1) = 9 |

Every value here matches the recorded output exactly. `transpose()` returns a new `IdentityTensor` with `offset` negated and everything else unchanged — for the main diagonal (`offset = 0`), `-0` is still `0`, so a captured "original" and "transposed" identity print as structurally identical, which is the correct mathematical answer (the identity matrix is its own transpose) arrived at almost by accident, since the code never special-cases `offset == 0`; it just negates whatever's there. `scale_by(factor)` composes multiplicatively rather than replacing: scaling a `scale = 2.0` tensor `by 3.0` produces `scale = 6.0`, not `3.0` — exactly what the recorded "Original scale: 2.0 / New scale: 6.0" output shows, and the reason two successive `scale_by` calls behave like `scale_by(f1 * f2)` in one step rather than like `scale_by(f2)` alone.

```
[COMMON TRAP]  "29 bytes" describes the *tensor*, not the matrix it represents

get_memory_usage() answers "how many bytes does this IdentityTensor struct
occupy" — three Ints, one Scalar, one Bool. It is easy to misread the printed
"Memory usage: 29 bytes" next to a 1000x1000 matrix as some kind of
compression ratio on the matrix's data. There is no compression happening,
because there is no matrix data to compress: a 1000x1000 identity was never
materialized in the first place. The 29 bytes is the cost of remembering the
*rule* ("this is an identity matrix of this size, this scale, this offset"),
which is exactly why it doesn't depend on `size` — the rule is the same
length regardless of how large a matrix it describes. Compare this to
Section 9.2, where DiagonalTensor's memory usage genuinely does grow with
size, because a diagonal tensor stores real per-element data that an
identity tensor doesn't need at all.
```

## 9.2 Diagonal Tensors: Storage Proportional to What Actually Varies `[FOUNDATIONAL]`

### Intuition

An identity tensor is a rule with no exceptions: every diagonal entry is the same `scale`. The moment you need the diagonal entries to differ from each other — a payroll ledger where employee 0 got a $2 raise, employee 1 got a $5 raise, employee 2 got nothing — the rule-only representation stops working, and you need actual per-entry storage back. But you still don't need the *whole* ledger, the one that also records the (nonexistent) relationship between employee 0's raise and employee 7's salary. You need exactly `n` numbers for `n` employees. `DiagonalTensor` is that: a `size`-parameterized rule ("only entries where `col - row == offset` can be nonzero") *plus* a compact `n`-length buffer holding the actual values, indexed by position along the diagonal rather than by full `(row, col)` coordinates.

### Background

```mojo
struct DiagonalTensor[dtype: DType]:
    var diagonal_data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var offset: Int
    var diagonal_length: Int
    var _owns_data: Bool
    var _validation_enabled: Bool
```

`get_diagonal_index(row, col)` first checks `col - row == offset`; if not, returns `-1` (not stored). If so, it returns `row` when `offset >= 0`, or `col` when `offset < 0` — the compact index into `diagonal_data`, not a full matrix offset. `set_item` enforces the same rule in the other direction: writing a *nonzero* value off the diagonal raises `"Cannot set non-zero value outside diagonal"`, but writing an off-diagonal *zero* is silently accepted as a no-op, since a zero there is already correct.

| | Identity (9.1) | Diagonal (9.2) | Dense |
|---|---|---|---|
| Memory | O(1) | O(n) | O(n²) |
| Can differ per position? | no | yes, along the diagonal | yes, everywhere |
| `get_memory_usage()` formula | constant `29` | `34 + diagonal_length × sizeof[Scalar]` | `n² × sizeof[Scalar]` |

`get_memory_usage()`'s base term — `4 × sizeof[Int]() + 2 × sizeof[Bool]() = 34` bytes for the struct's own bookkeeping (`size`, `offset`, `diagonal_length`, a fourth accounted `Int`-equivalent slot, plus the two `Bool` flags) — is a fixed cost paid once per `DiagonalTensor`, on top of the genuinely size-proportional `diagonal_length × 4` bytes of actual data (for `float32`). That fixed cost is small next to a single diagonal, but it does not disappear when you start combining diagonals, as Worked Example 9.2.3 traces exactly.

### Worked Example 9.2.1 — Sparsity as a direct fraction

`get_sparsity()` returns `diagonal_length / (size × size)`. For a `4×4` main diagonal (`diagonal_length = 4`): `4/16 = 25.0%` — matching the recorded output exactly. For a `4×4` super- or sub-diagonal at `offset = ±1` (`diagonal_length = 3`): `3/16 = 18.75%`, also matching. This is the same `get_diagonal_length` arithmetic from Section 9.1, now expressed as a fraction of total matrix area rather than a raw count — the further a diagonal sits from the main one, the shorter it runs off the edge of the matrix, and the smaller its sparsity percentage gets.

### Worked Example 9.2.2 — Building a tridiagonal matrix by hand

`create_tridiagonal(4, main_val=2.0, super_val=1.0, sub_val=-1.0)` builds three separate `DiagonalTensor`s — one at `offset=-1` filled with `-1.0`, one at `offset=0` filled with `2.0`, one at `offset=+1` filled with `1.0` — and wraps them in a `MultiBandDiagonal`, whose `get_item(row, col)` computes `offset = col - row` and linearly scans its three stored offsets for a match. Tracing every position of the resulting `4×4` matrix:

```
row 0: col-row = 0,1,2,3  -> matches offset 0 (2.0), offset 1 (1.0), none, none
row 1: col-row = -1,0,1,2 -> matches offset -1 (-1.0), offset 0 (2.0), offset 1 (1.0), none
row 2: col-row = -2,-1,0,1 -> none, offset -1 (-1.0), offset 0 (2.0), offset 1 (1.0)
row 3: col-row = -3,-2,-1,0 -> none, none, offset -1 (-1.0), offset 0 (2.0)
```

which is exactly the recorded matrix:

```
Row 0: 2.0 1.0 0.0 0.0
Row 1: -1.0 2.0 1.0 0.0
Row 2: 0.0 -1.0 2.0 1.0
Row 3: 0.0 0.0 -1.0 2.0
```

### Worked Example 9.2.3 — Where "sparse" costs more than "dense"

`MultiBandDiagonal.get_total_memory_usage()` is `3 × sizeof[Int]() + sizeof[Bool]() = 25` bytes of its own bookkeeping, plus the sum of each band's own `DiagonalTensor.get_memory_usage()`. For the `4×4` tridiagonal above: the `offset=-1` and `offset=+1` bands each have `diagonal_length = 3` (`34 + 3×4 = 46` bytes each), and the `offset=0` band has `diagonal_length = 4` (`34 + 4×4 = 50` bytes). Total: `25 + 46 + 50 + 46 = 167` bytes — exactly the recorded `"Total memory usage: 167 bytes"`. The dense `4×4` equivalent needs `4×4×4 = 64` bytes. `167 / 64 = 260.9375%` — matching the recorded efficiency figure precisely, and meaning the "memory-efficient" sparse-diagonal representation is **2.6 times larger** than just storing the dense matrix.

```
[COMMON TRAP]  three diagonals means three fixed overheads, paid before
                a single number of actual data is stored

Of that 167 bytes, only 3+4+3 = 10 scalars (40 bytes) are the tridiagonal
matrix's actual nonzero values. The other 127 bytes — three separate 34-byte
DiagonalTensor headers, plus the 25-byte MultiBandDiagonal wrapper around
them — are pure bookkeeping, paid once per band regardless of how few
elements that band stores. A single DiagonalTensor amortizes its 34-byte
header well once diagonal_length is large; three of them, each holding only
3-4 real values, do not. The lesson generalizes: any "memory-efficient"
structure with a fixed per-instance cost only wins once its variable,
size-proportional cost is the dominant term — and multi-band structures like
this one multiply that fixed cost by the band count, pushing the break-even
point further out. For a general n×n tridiagonal matrix built this way, the
break-even against dense (4n² bytes) is roughly where 25 + 3×34 + (3n-2)×4
bytes drops below 4n² bytes — comfortably true once n grows into the
hundreds, but false, as shown here, at n=4.
```

## 9.3 Sparse Tensors: Coordinates Instead of a Grid `[FOUNDATIONAL]`

### Intuition

A diagonal tensor knows exactly which positions can be nonzero before it stores a single value — the diagonal, by definition, is a fixed, predictable pattern. Real-world sparse data rarely comes with a pattern that clean. A seating chart for a mostly-empty stadium doesn't follow a rule like "every 5th seat is occupied" — it's just a list of the specific seats someone is actually sitting in: `(section 12, row 4, seat 9) -> "occupied"`. `SparseTensorCOO` is that seating chart formalized: a flat array of `(row, col, value)` triples — the Coordinate (COO) format — with no assumption at all about where the nonzero entries fall.

### Background

```mojo
struct SparseElement[dtype: DType]:
    var row: Int
    var col: Int
    var value: Scalar[dtype]

struct SparseTensorCOO[dtype: DType]:
    var elements: UnsafePointer[SparseElement[dtype]]
    var num_elements: Int
    var capacity: Int
    var shape: List[Int]
    var _owns_data: Bool
    var _validation_enabled: Bool
```

Giving up the diagonal's structural guarantee costs something: `_find_element_index(row, col)` has no formula to jump straight to the right slot the way `DiagonalTensor.get_diagonal_index` does. It linearly scans every stored element until it finds a coordinate match (or reaches the end and returns `-1`). Lookups are `O(nnz)` — proportional to the number of *stored* nonzero elements — rather than `O(1)`.

| | Diagonal (9.2) | Sparse COO (9.3) |
|---|---|---|
| Where nonzeros can live | a predictable diagonal | anywhere, no pattern assumed |
| Memory | O(n) | O(nnz) — proportional to stored elements |
| `get_item` cost | O(1) — direct index arithmetic | O(nnz) — linear scan over stored elements |
| Memory formula | `34 + diagonal_length × sizeof[Scalar]` | `34 + capacity × sizeof[SparseElement] + shape_bytes` |

That last row is the one worth sitting with before the worked examples: the memory formula depends on `capacity` — the number of slots *reserved* at construction — not on `num_elements`, the number actually in use.

### Worked Example 9.3.1 — Adding a zero doesn't add anything

`add_element(row, col, value)` builds a `SparseElement`, checks `element.is_zero()` (`abs(value) <= 1e-10` by default), and if so calls `remove_element(row, col)` instead of storing anything. In the recorded test, four elements are added to a `3×3` sparse tensor (`num_elements = 4`), and then `add_element(1, 0, 0.0)` is called. `0.0` is zero, so `remove_element(1, 0)` runs — but `(1, 0)` was never stored, so `_find_element_index` returns `-1`, and `remove_element` does nothing. `num_elements` stays at `4` both before and after, exactly as recorded — "adding a zero" here is really "attempting, and failing, to remove a coordinate that was never occupied."

### Worked Example 9.3.2 — `compress_storage()`, and why it has nothing to do

`compress_storage()` walks every stored element, keeps the ones that are *not* `is_zero()`, and shifts them down to fill any gaps. In the recorded test, two more elements are added — `(2, 0, 1e-12)`, which `is_zero()` catches immediately (so it never gets past `add_element` into storage, `num_elements` stays put), and `(2, 1, 1e-8)`, which is *not* zero by the `1e-10` threshold, so it's stored (`num_elements` becomes `5`). `compress_storage()` then scans those 5 stored elements for any that are `is_zero()` — and finds none, because `add_element` already refuses to store a zero value in the first place. "Before compression: 5" and "after compression: 5" are identical in the recorded output for exactly this reason: under the public API, a zero-valued element can never reach `self.elements` to begin with, which makes `compress_storage()` unreachable dead code in ordinary use. The only way to give it real work would be writing into `self.elements` directly, bypassing `add_element` entirely.

```
[COMMON TRAP]  get_memory_usage() reports the reservation, not the tenancy

get_memory_usage() is `34 + capacity * sizeof[SparseElement] + shape_bytes`.
Every sparse tensor in this section's tests uses the default
`capacity=1000`, and `sizeof[SparseElement[float32]]()` is 24 bytes (8 for
`row`, 8 for `col`, 4 for `value`, padded to a 24-byte aligned size) — so
`get_memory_usage()` reports `34 + 1000*24 + 16 = 24050` bytes for *every*
tensor built with the default capacity, whether it holds 0, 4, or 5 live
elements. That constant 24050 is why the recorded memory-efficiency
benchmark shows sparse storage costing *more* than dense at small sizes: a
50x50 matrix at 5% sparsity needs only 10,000 bytes dense, but the sparse
version — reserving room for 1000 elements it never gets close to using —
reports 24,050 bytes, a 240.5% "efficiency." Set `sparse_bytes` equal to
`dense_bytes` (`24050 = size^2 * 4`) and solve: the crossover for a
default-capacity sparse tensor sits around `size ~ 78`, meaning any square
matrix smaller than roughly 78x78 stores this sparse representation *less*
efficiently than a plain dense buffer, regardless of how sparse its actual
data is — the fixed reservation dominates until the matrix itself grows
large enough to catch up. Passing a smaller `capacity` explicit argument
moves that crossover down, but never removes it, because the allocation is
always sized to `capacity`, not to `num_elements` — and compress_storage()
does not help, for the same reason it did nothing in Worked Example 9.3.2:
it rearranges live elements within the existing allocation, it never calls
`.free()` and reallocates a smaller one.
```

## 9.4 Triangular Tensors: Packed Storage and a Collision Worth Finding `[FOUNDATIONAL]`

### Intuition

Fold a square napkin in half along its diagonal crease, and one half exactly covers the other. A triangular matrix — every entry below (or above) the diagonal fixed at zero — has that same redundancy: once you know it's triangular, half the grid tells you nothing you didn't already know. `TriangularTensor` acts on that by storing only the `n(n+1)/2` entries on and to one side of the diagonal, packed into a flat buffer with no gaps, instead of the full `n²`. That packing needs a formula to translate `(row, col)` into "which slot in the packed buffer" — and, as this section traces in detail, `TriangularTensor`'s own formula for the upper-triangular case gets that translation wrong.

### Background

```mojo
struct TriangularTensor[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var is_upper: Bool
    var _storage_size: Int   # size * (size + 1) // 2
    var _owns_data: Bool
    var _validation_enabled: Bool
```

The packing formula lives in `_get_storage_index`. For the lower-triangular case (`row >= col`), it's:

```mojo
return (row * (row + 1)) // 2 + col
```

and for the upper-triangular case (`col >= row`):

```mojo
return (row * (2 * size - row - 1)) // 2 + (col - row)
```

| | Triangular (packed) | Dense |
|---|---|---|
| Storage | `n(n+1)/2` elements | `n²` elements |
| Limiting efficiency as n grows | → 50% of dense | 100% (baseline) |
| `get_item` cost | O(1) — one formula evaluation | O(1) — direct index |

### Worked Example 9.4.1 — Deriving the correct formula, and finding where the code's differs

For row-major upper-triangular packing, the number of elements stored in every row *before* row `r` is `sum_{i=0}^{r-1} (size - i)`, since row `i` stores `size - i` entries (from the diagonal to the last column). That sum is `r × size - r(r-1)/2`, which can be rewritten as `r(2×size - r + 1) / 2`. So the correct base offset for row `r`, before adding `col - row`, is:

```
correct_base(r) = r * (2*size - r + 1) // 2
```

Compare this to the code's actual base term, `row * (2*size - row - 1) // 2` — the sign on that middle term is `-1` where the derivation above gives `+1`. For `size = 4`:

| row | correct base | code's base | difference |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 1×(8-1+1)/2 = 4 | 1×(8-1-1)/2 = 3 | −1 |
| 2 | 2×(8-2+1)/2 = 7 | 2×(8-2-1)/2 = 5 | −2 |
| 3 | 3×(8-3+1)/2 = 9 | 3×(8-3-1)/2 = 6 | −3 |

Row 0 is unaffected (there's nothing to be off by yet), but every row after it computes a base that's too small by exactly its own row number — and a base that's too small means later rows start writing into slots that earlier rows already claimed.

### Worked Example 9.4.2 — The collision, traced through an actual test run

The recorded "Upper Triangular Tensor Creation" test builds a `4×4` upper triangular tensor and calls `set_item` ten times, once for every valid `(row, col)` with `col >= row`, in this order: `(0,0)=1, (0,1)=2, (0,2)=3, (0,3)=4, (1,1)=5, (1,2)=6, (1,3)=7, (2,2)=8, (2,3)=9, (3,3)=10`. Using the code's actual (buggy) base values from Worked Example 9.4.1 — base(0)=0, base(1)=3, base(2)=5, base(3)=6 — every write's storage slot is `base(row) + (col - row)`:

```
(0,0) -> idx 0        (1,1) -> idx 3   <- SAME SLOT as (0,3)
(0,1) -> idx 1        (1,2) -> idx 4
(0,2) -> idx 2        (1,3) -> idx 5   <- SAME SLOT as (2,2)
(0,3) -> idx 3        (2,2) -> idx 5   <- SAME SLOT as (1,3)
(2,3) -> idx 6        (3,3) -> idx 6   <- SAME SLOT as (2,3)
```

Three collisions, and in every one the *later* write in program order simply overwrites the earlier one — there's no error, no warning, nothing to indicate two logically distinct matrix positions just started sharing one memory cell. Reading the matrix back afterward, `get_item(0,3)` returns whatever is at idx 3 *now*, which is `5.0` (from `(1,1)`, written after `(0,3)`), not the `4.0` that was set. Continuing the trace: idx 5 ends up holding `8.0` (from `(2,2)`, the later of the two writers), and idx 6 ends up holding `10.0` (from `(3,3)`, the later writer). Reading every position back gives:

```
Row 0: 1.0 2.0 3.0 5.0
Row 1: 0.0 5.0 6.0 8.0
Row 2: 0.0 0.0 8.0 10.0
Row 3: 0.0 0.0 0.0 10.0
```

— which is exactly, digit for digit, the recorded output. The `4.0`, `7.0`, and `9.0` that were explicitly set are simply gone, silently replaced by values from unrelated coordinates, because three pairs of positions were never given distinct storage slots to begin with.

### Worked Example 9.4.3 — The same bug, quietly destroying a diagonal

The recorded "Triangular Scaling" test (`size = 3`) is a smaller, arguably more alarming demonstration. It calls `fill_diagonal(1.0)` — intending to set `(0,0)`, `(1,1)`, and `(2,2)` all to `1.0` — and *then* calls `set_item(0,1,2.0)` and `set_item(0,2,3.0)`. Using the buggy bases for `size=3` (base(0)=0, base(1)=2, base(2)=3): `fill_diagonal` writes idx 0 = 1.0 (for (0,0)), idx 2 = 1.0 (for (1,1), since base(1)+0=2), idx 3 = 1.0 (for (2,2), since base(2)+0=3). Then `set_item(0,1,2.0)` writes idx 1 = 2.0 — no collision yet. But `set_item(0,2,3.0)` writes idx `base(0) + 2 = 2` — **the same slot `fill_diagonal` used for `(1,1)`** — overwriting it from `1.0` to `3.0`. The recorded "Before scaling" output, read back through `get_item`, shows exactly this corruption already baked in:

```
Row 0: 1.0 2.0 3.0
Row 1: 0.0 3.0 4.0
Row 2: 0.0 0.0 4.0
```

Position `(1,1)` — filled by `fill_diagonal` to be `1.0` — reads back as `3.0`, the value a completely unrelated call to `set_item(0,2,...)` wrote afterward. `(2,2)` similarly reads back as `4.0` (from the subsequent `set_item(1,2,4.0)`, which collides with `(2,2)`'s slot the same way). `scale_triangle(2.0)` then simply doubles every stored slot — `2.0, 4.0, 6.0, 8.0` — and reading the "after" matrix back reproduces the doubled, still-aliased values exactly as recorded (`Row 1: 0.0 6.0 8.0`, `Row 2: 0.0 0.0 8.0`). Nothing about `fill_diagonal` or `scale_triangle` is wrong in isolation; both simply inherit whatever `_get_storage_index` tells them, and that formula is the single point of failure for the whole struct.

```
[COMMON TRAP]  a packed-storage index needs to be checked as a bijection,
                not just "shaped like" the textbook formula

The lower-triangular branch of _get_storage_index — `(row*(row+1))//2 + col`
— is the textbook-correct formula and has no such bug (trace it the same
way: base(r) = r(r+1)/2 counts elements in all rows before r, and every
(row,col) with row>=col lands in a distinct slot). Only the upper-triangular
branch has the sign error traced above, which is part of why it's easy to
miss on a read-through: half of this struct is simply correct. The general
lesson carries beyond this one struct: verifying a packed-index formula
means checking that it defines a bijection between every valid (row, col)
pair and a unique slot in [0, n(n+1)/2) — not just confirming the formula
"looks like" the standard derivation, and not just running one test that
happens to touch few enough positions to avoid a collision. A linear-algebra
routine (forward or backward substitution, say) built on top of this
TriangularTensor without ever printing the full matrix back out would
compute silently wrong answers, with no exception raised anywhere.
```

### Worked Example 9.4.4 — Storage efficiency's approach to 50%

`get_storage_efficiency()` is `_storage_size / (size × size) = [size(size+1)/2] / size² = (size+1)/(2×size) = 0.5 + 1/(2×size)`. As `size` grows, that second term shrinks toward zero and efficiency approaches — but never quite reaches — 50%:

| Size | `(size+1)/(2×size)` | Recorded efficiency | Recorded memory savings |
|---|---|---|---|
| 10 | 11/20 = 0.55 | 55.0% | 45.0% |
| 100 | 101/200 = 0.505 | 50.5% | 49.5% |
| 500 | 501/1000 = 0.501 | 50.1% | 49.9% |

Every value matches the recorded output exactly, and the pattern makes the earlier bug easier to place in context: packed triangular storage is a genuine, well-founded win for large matrices (approaching half the memory of dense, with correct lower-triangular indexing to prove it's achievable) — which makes the upper-triangular aliasing bug more of a shame than a fatal flaw. The idea behind the struct is sound; one formula inside it isn't.

## 9.5 Complete Runnable Code

The four sections above are drawn from four independent, runnable Mojo files. Each is reproduced here in full, exactly as written, together with its own recorded output.

### File: `45_specialized_tensors_part_a.mojo` — Section 9.1

**Run:** `pixi run mojo 45_specialized_tensors_part_a.mojo`

```mojo

from memory import UnsafePointer
from collections import List
from sys import sizeof


# Core Tensor Infrastructure - Part 1.3.4A: Identity Matrix Implementation
#
# This section implements memory-efficient identity matrix structures optimized
# for mathematical operations and linear algebra computations. Provides specialized
# storage patterns and algorithms that exploit the mathematical properties of
# identity matrices for optimal performance and memory usage.
#
# Key Design Principles:
# - Memory-efficient storage (O(1) space complexity)
# - Fast element access and mathematical operations
# - Seamless integration with general tensor operations
# - Support for scaled and offset identity matrices
# - Zero-copy conversions where possible
#
# Implementation Features:
# 1. Memory-efficient identity matrix representation
# 2. Configurable scaling factors and diagonal offsets
# 3. Fast element access without full matrix storage
# 4. Optimized conversion to dense format when needed
# 5. Comprehensive validation and error handling
#
# Mathematical Properties Exploited:
# - Identity matrices are sparse with known structure
# - Only diagonal elements need explicit representation
# - Matrix operations can be optimized for identity structure
# - Memory usage independent of matrix size for basic operations

alias DEFAULT_IDENTITY_SIZE = 100
alias MAX_IDENTITY_SIZE = 10000


struct IdentityTensor[dtype: DType]:
    """Memory-efficient identity matrix implementation with configurable scaling."""
    var size: Int
    var scale: Scalar[dtype]
    var offset: Int  # Diagonal offset (0 = main diagonal, +n = upper, -n = lower)
    var _validation_enabled: Bool
    
    fn __init__(out self, size: Int, scale: Scalar[dtype] = Scalar[dtype](1), offset: Int = 0) raises:
        """Initialize identity tensor with size, scale, and diagonal offset."""
        if size <= 0:
            raise Error("Identity matrix size must be positive")
        if size > MAX_IDENTITY_SIZE:
            raise Error("Identity matrix size exceeds maximum allowed")
        if abs(offset) >= size:
            raise Error("Diagonal offset must be smaller than matrix size")
        
        self.size = size
        self.scale = scale
        self.offset = offset
        self._validation_enabled = True
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for identity tensor."""
        self.size = existing.size
        self.scale = existing.scale
        self.offset = existing.offset
        self._validation_enabled = existing._validation_enabled
    
    fn get_item(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element at specified position with bounds checking."""
        if self._validation_enabled:
            if row < 0 or row >= self.size or col < 0 or col >= self.size:
                return Scalar[dtype](0)
        
        # Check if on the specified diagonal
        if col - row == self.offset:
            return self.scale
        else:
            return Scalar[dtype](0)
    
    fn get_item_unsafe(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element without bounds checking for performance-critical code."""
        if col - row == self.offset:
            return self.scale
        else:
            return Scalar[dtype](0)
    
    fn is_on_diagonal(self, row: Int, col: Int) -> Bool:
        """Check if position is on the identity diagonal."""
        return col - row == self.offset
    
    fn get_diagonal_length(self) -> Int:
        """Get effective length of the diagonal."""
        if self.offset >= 0:
            return max(0, self.size - self.offset)
        else:
            return max(0, self.size + self.offset)
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes (constant for identity matrices)."""
        return 3 * sizeof[Int]() + sizeof[Scalar[dtype]]() + sizeof[Bool]()
    
    fn set_validation(mut self, enabled: Bool):
        """Enable or disable bounds checking for performance tuning."""
        self._validation_enabled = enabled
    
    fn transpose(self) raises -> IdentityTensor[dtype]:
        """Return transposed identity matrix (flipped offset)."""
        var transposed = IdentityTensor[dtype](self.size, self.scale, -self.offset)
        return transposed
    
    fn scale_by(self, factor: Scalar[dtype]) raises -> IdentityTensor[dtype]:
        """Return scaled identity matrix."""
        var scaled = IdentityTensor[dtype](self.size, self.scale * factor, self.offset)
        return scaled
    
    fn to_dense(self) raises -> DenseTensor[dtype]:
        """Convert to dense tensor representation."""
        var shape = List[Int]()
        shape.append(self.size)
        shape.append(self.size)
        
        var dense = DenseTensor[dtype](shape)
        
        for i in range(self.size):
            for j in range(self.size):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = self.get_item_unsafe(i, j)  # Use unsafe for performance
                dense.set_item(indices, value)
        
        return dense
    
    fn print_info(self):
        """Print identity tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var size_str: String = String(self.size)
        var scale_str: String = String(self.scale)
        var offset_str: String = String(self.offset)
        var diag_len_str: String = String(self.get_diagonal_length())
        var memory_str: String = String(self.get_memory_usage())
        
        print("IdentityTensor[" + dtype_str + "]")
        print("  Size: " + size_str + "x" + size_str)
        print("  Scale: " + scale_str)
        print("  Diagonal offset: " + offset_str)
        print("  Diagonal length: " + diag_len_str)
        print("  Memory usage: " + memory_str + " bytes")
        print("  Validation: " + ("enabled" if self._validation_enabled else "disabled"))


struct DenseTensor[dtype: DType]:
    """Dense tensor implementation for conversion and compatibility."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1
        self._owns_data = True
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
            self.shape.append(shape[i])
            self._total_elements *= shape[i]
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for dense tensor."""
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        self.ndim = existing.ndim
        self._total_elements = existing._total_elements
        self._owns_data = True
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
    
    fn _get_linear_index(self, indices: List[Int]) -> Int:
        """Convert multi-dimensional indices to linear index."""
        var linear_index = 0
        var stride = 1
        
        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]
        
        return linear_index
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        return self.data[linear_index]
    
    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        self.data[linear_index] = value
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        return self._total_elements
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        return self._total_elements * sizeof[Scalar[dtype]]()
    
    fn print_info(self):
        """Print tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var memory_str: String = String(self.get_memory_usage())
        
        print("DenseTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)
        print("  Memory usage: " + memory_str + " bytes")


fn create_identity[dtype: DType](size: Int, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> IdentityTensor[dtype]:
    """Create standard identity tensor."""
    return IdentityTensor[dtype](size, scale, 0)

fn create_scaled_identity[dtype: DType](size: Int, scale: Scalar[dtype]) raises -> IdentityTensor[dtype]:
    """Create scaled identity tensor."""
    return IdentityTensor[dtype](size, scale, 0)

fn create_super_diagonal[dtype: DType](size: Int, offset: Int, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> IdentityTensor[dtype]:
    """Create super-diagonal (upper offset) identity tensor."""
    if offset <= 0:
        raise Error("Super-diagonal offset must be positive")
    return IdentityTensor[dtype](size, scale, offset)

fn create_sub_diagonal[dtype: DType](size: Int, offset: Int, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> IdentityTensor[dtype]:
    """Create sub-diagonal (lower offset) identity tensor."""
    if offset >= 0:
        raise Error("Sub-diagonal offset must be negative")
    return IdentityTensor[dtype](size, scale, offset)

fn create_batch_identity[dtype: DType](size: Int, batch_size: Int, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> Int:
    """Create batch of identity tensors and return count."""
    if batch_size <= 0:
        raise Error("Batch size must be positive")
    
    # Create and test batch tensors
    for _ in range(batch_size):
        var identity = create_identity[dtype](size, scale)
        var _ = identity.get_item(0, 0)  # Force usage
    
    return batch_size


struct PerformanceTimer:
    """Simple performance timing utility."""
    var name: String
    
    fn __init__(out self, name: String):
        self.name = name
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
    
    fn start(self):
        """Start timing operation."""
        print("Starting: " + self.name)
    
    fn end(self, operations: Int = 1):
        """End timing and report results."""
        var ops_str: String = String(operations)
        print("Completed: " + self.name + " (" + ops_str + " operations)")

fn benchmark_identity_creation[dtype: DType](size: Int, iterations: Int) -> Float64:
    """Benchmark identity tensor creation performance."""
    var timer = PerformanceTimer("Identity Creation Benchmark")
    timer.start()
    
    try:
        for _ in range(iterations):
            var identity = create_identity[dtype](size)
            # Force some usage to prevent optimization
            var _ = identity.get_item(0, 0)
    except:
        print("Benchmark failed")
        return -1.0
    
    timer.end(iterations)
    return Float64(iterations) / 1000.0  # Simple metric

fn benchmark_memory_efficiency[dtype: DType](size: Int) -> Float64:
    """Compare memory usage: identity vs dense."""
    try:
        var identity = create_identity[dtype](size)
        var dense = identity.to_dense()
        
        var identity_memory = identity.get_memory_usage()
        var dense_memory = dense.get_memory_usage()
        
        var efficiency = Float64(identity_memory) / Float64(dense_memory)
        
        var identity_str: String = String(identity_memory)
        var dense_str: String = String(dense_memory)
        var efficiency_str: String = String(efficiency * 100.0)
        
        print("Memory Efficiency Analysis:")
        print("  Identity tensor: " + identity_str + " bytes")
        print("  Dense equivalent: " + dense_str + " bytes")
        print("  Efficiency: " + efficiency_str + "% of dense storage")
        
        return efficiency
        
    except:
        print("Memory benchmark failed")
        return -1.0


fn test_basic_identity_operations():
    """Test basic identity tensor functionality."""
    print("=== Testing Basic Identity Operations ===")
    
    try:
        print("\n1. Standard Identity Matrix Creation:")
        var identity = create_identity[DType.float32](4)
        identity.print_info()
        
        print("\nIdentity matrix values (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = identity.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Scaled Identity Matrix:")
        var scaled = create_scaled_identity[DType.float32](3, 2.5)
        scaled.print_info()
        
        print("Scaled identity values:")
        for i in range(3):
            for j in range(3):
                var value = scaled.get_item(i, j)
                if value != Scalar[DType.float32](0):
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var value_str: String = String(value)
                    print("  [" + i_str + ", " + j_str + "] = " + value_str)
        
        print("\n3. Super-diagonal Identity:")
        var super_diag = create_super_diagonal[DType.float32](4, 1, 3.0)
        super_diag.print_info()
        
        print("Super-diagonal values:")
        for i in range(4):
            for j in range(4):
                var value = super_diag.get_item(i, j)
                if value != Scalar[DType.float32](0):
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var value_str: String = String(value)
                    print("  [" + i_str + ", " + j_str + "] = " + value_str)
        
        print("Basic identity operations completed successfully")
        
    except:
        print("Error in basic identity operations")

fn test_identity_mathematical_operations():
    """Test mathematical operations on identity tensors."""
    print("\n=== Testing Identity Mathematical Operations ===")
    
    try:
        print("\n1. Identity Transpose:")
        var identity = create_identity[DType.float32](3, 2.0)
        var transposed = identity.transpose()
        
        print("Original identity:")
        identity.print_info()
        print("Transposed identity:")
        transposed.print_info()
        
        print("\n2. Identity Scaling:")
        var scaled = identity.scale_by(3.0)
        scaled.print_info()
        
        print("Original scale: " + String(identity.scale))
        print("New scale: " + String(scaled.scale))
        
        print("\n3. Diagonal Length Calculation:")
        var sizes = List[Int]()
        sizes.append(3)
        sizes.append(5)
        sizes.append(10)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var main_diag = create_identity[DType.float32](size)
            var super_diag = create_super_diagonal[DType.float32](size, 2)
            var sub_diag = create_sub_diagonal[DType.float32](size, -1)
            
            var size_str: String = String(size)
            var main_len_str: String = String(main_diag.get_diagonal_length())
            var super_len_str: String = String(super_diag.get_diagonal_length())
            var sub_len_str: String = String(sub_diag.get_diagonal_length())
            
            print("Size " + size_str + " matrix:")
            print("  Main diagonal length: " + main_len_str)
            print("  Super diagonal (+2) length: " + super_len_str)
            print("  Sub diagonal (-1) length: " + sub_len_str)
        
        print("Mathematical operations completed successfully")
        
    except:
        print("Error in mathematical operations")

fn test_identity_performance():
    """Test identity tensor performance characteristics."""
    print("\n=== Testing Identity Performance ===")
    
    try:
        print("\n1. Creation Performance:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(1000)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var iterations = 1000
            var size_str: String = String(size)
            print("Testing " + size_str + "x" + size_str + " identity creation...")
            
            var result = benchmark_identity_creation[DType.float32](size, iterations)
            var result_str: String = String(result)
            print("  Performance metric: " + result_str + " k-ops")
        
        print("\n2. Memory Efficiency:")
        for i in range(len(sizes)):
            var size = sizes[i]
            var size_str: String = String(size)
            print("Memory analysis for " + size_str + "x" + size_str + " matrix:")
            var _ = benchmark_memory_efficiency[DType.float32](size)
        
        print("\n3. Access Pattern Performance:")
        var large_identity = create_identity[DType.float32](1000)
        large_identity.set_validation(False)  # Disable bounds checking
        
        var timer = PerformanceTimer("Element Access (1M operations)")
        timer.start()
        
        var sum = Scalar[DType.float32](0)
        for i in range(1000):
            for j in range(1000):
                sum += large_identity.get_item_unsafe(i, j)
        
        timer.end(1000000)
        var sum_str: String = String(sum)
        print("Sum result: " + sum_str + " (validation)")
        
        print("Performance testing completed successfully")
        
    except:
        print("Error in performance testing")

fn test_identity_conversions():
    """Test conversions between identity and other formats."""
    print("\n=== Testing Identity Conversions ===")
    
    try:
        print("\n1. Identity to Dense Conversion:")
        var identity = create_scaled_identity[DType.float32](4, 2.0)
        var dense = identity.to_dense()
        
        print("Original identity:")
        identity.print_info()
        print("Converted dense:")
        dense.print_info()
        
        print("Dense matrix values:")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = dense.get_item(indices)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Verification of Conversion Accuracy:")
        var errors = 0
        for i in range(4):
            for j in range(4):
                var identity_val = identity.get_item(i, j)
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var dense_val = dense.get_item(indices)
                
                if identity_val != dense_val:
                    errors += 1
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var id_str: String = String(identity_val)
                    var dense_str: String = String(dense_val)
                    print("  Mismatch at [" + i_str + ", " + j_str + "]: " + id_str + " vs " + dense_str)
        
        var errors_str: String = String(errors)
        if errors == 0:
            print("Conversion accuracy: Perfect (0 errors)")
        else:
            print("Conversion errors: " + errors_str)
        
        print("\n3. Batch Identity Creation:")
        var batch_count = create_batch_identity[DType.float32](3, 5, 1.5)
        var batch_size_str: String = String(batch_count)
        print("Created batch of " + batch_size_str + " identity tensors")
        
        print("Each tensor in batch has:")
        var sample_identity = create_scaled_identity[DType.float32](3, 1.5)
        sample_identity.print_info()
        
        print("Conversion testing completed successfully")
        
    except:
        print("Error in conversion testing")

fn test_identity_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Identity Error Handling ===")
    
    var test_count = 0
    var passed_count = 0
    
    print("\n1. Invalid Size Errors:")
    test_count += 1
    try:
        var _ = create_identity[DType.float32](0)
        print("Should have failed with zero size")
    except:
        print("Correctly caught zero size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_identity[DType.float32](-5)
        print("Should have failed with negative size")
    except:
        print("Correctly caught negative size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_identity[DType.float32](20000)
        print("Should have failed with oversized matrix")
    except:
        print("Correctly caught oversized matrix error")
        passed_count += 1
    
    print("\n2. Invalid Offset Errors:")
    test_count += 1
    try:
        var _ = create_super_diagonal[DType.float32](5, 5)
        print("Should have failed with offset >= size")
    except:
        print("Correctly caught invalid offset error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_sub_diagonal[DType.float32](3, 1)
        print("Should have failed with positive offset for sub-diagonal")
    except:
        print("Correctly caught positive offset error for sub-diagonal")
        passed_count += 1
    
    print("\n3. Edge Case Testing:")
    test_count += 1
    try:
        var tiny = create_identity[DType.float32](1)
        var value = tiny.get_item(0, 0)
        if value == Scalar[DType.float32](1):
            print("1x1 identity matrix works correctly")
            passed_count += 1
        else:
            print("1x1 identity matrix incorrect value")
    except:
        print("1x1 identity matrix creation failed")
    
    test_count += 1
    try:
        var identity = create_identity[DType.float32](3)
        var out_of_bounds = identity.get_item(5, 5)
        if out_of_bounds == Scalar[DType.float32](0):
            print("Out-of-bounds access returns zero")
            passed_count += 1
        else:
            print("Out-of-bounds access incorrect value")
    except:
        print("Out-of-bounds access handling failed")
    
    var passed_str: String = String(passed_count)
    var total_str: String = String(test_count)
    print("\nError Handling Summary: " + passed_str + "/" + total_str + " tests passed")
    
    if passed_count == test_count:
        print("All error handling tests passed")
    else:
        print("Some error handling tests failed")

fn test_identity_memory_management():
    """Test memory management and resource cleanup."""
    print("\n=== Testing Identity Memory Management ===")
    
    try:
        print("\n1. Memory Usage Analysis:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(1000)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var identity = create_identity[DType.float32](size)
            var memory = identity.get_memory_usage()
            
            var size_str: String = String(size)
            var memory_str: String = String(memory)
            var total_elements = size * size
            var total_str: String = String(total_elements)
            
            print("  " + size_str + "x" + size_str + " matrix:")
            print("    Identity storage: " + memory_str + " bytes")
            print("    Dense would need: " + total_str + " elements")
            print("    Memory efficiency: Constant vs O(n²)")
        
        print("\n2. Copy Constructor Testing:")
        var original = create_scaled_identity[DType.float32](5, 3.0)
        var copied = original
        
        print("Original:")
        original.print_info()
        print("Copy:")
        copied.print_info()
        
        # Verify independence
        var orig_val = original.get_item(0, 0)
        var copy_val = copied.get_item(0, 0)
        
        if orig_val == copy_val:
            print("Copy constructor preserves values")
        else:
            print("Copy constructor failed")
        
        print("\n3. Large Identity Creation/Destruction:")
        var creation_timer = PerformanceTimer("Large Identity Creation/Destruction")
        creation_timer.start()
        
        for _ in range(100):
            var large_identity = create_identity[DType.float32](1000)
            var _ = large_identity.get_item(0, 0)  # Force usage
        
        creation_timer.end(100)
        print("Large identity creation/destruction completed")
        
        print("Memory management testing completed successfully")
        
    except:
        print("Error in memory management testing")

fn run_all_identity_tests():
    """Run comprehensive test suite for identity tensors."""
    print("=====================================")
    print("=== IDENTITY TENSOR TEST SUITE ===")
    print("=====================================")
    
    test_basic_identity_operations()
    test_identity_mathematical_operations()
    test_identity_performance()
    test_identity_conversions()
    test_identity_error_handling()
    test_identity_memory_management()
    
    print("\n=====================================")
    print("=== IDENTITY TENSOR TESTS COMPLETE ===")
    print("=====================================")


fn main():
    """Main function to run identity tensor implementation tests."""
    print("=== Specialized Tensor Types - Part 1.3.4A ===")
    print("Identity Matrix Implementation - Complete Standalone Module")
    print("")
    
    run_all_identity_tests()
    
    print("\n=== Identity Matrix Implementation Summary ===")
    print("+ Memory-efficient identity matrix representation")
    print("+ Configurable scaling factors and diagonal offsets")
    print("+ Fast element access without full matrix storage")
    print("+ Optimized mathematical operations (transpose, scaling)")
    print("+ Comprehensive error handling and validation")
    print("+ Performance benchmarking and memory analysis")
    print("+ Seamless conversion to dense format")
    print("+ Batch identity tensor creation utilities")
    print("+ Constant memory usage regardless of matrix size")
    print("+ Integration-ready for tensor library ecosystem")
    print("")
    print("Performance Characteristics:")
    print("- Creation time: O(1) - constant regardless of size")
    print("- Memory usage: O(1) - ~20 bytes per identity tensor")
    print("- Element access: O(1) - direct mathematical computation")
    print("- Conversion to dense: O(n²) - only when explicitly needed")
    print("")
    print("Mathematical Features:")
    print("- Standard identity matrices (I)")
    print("- Scaled identity matrices (cI)")
    print("- Super-diagonal matrices (offset > 0)")
    print("- Sub-diagonal matrices (offset < 0)")
    print("- Transpose operations (flipped offset)")
    print("- Scaling operations (multiplicative factors)")
    print("")
    print("Identity Matrix Implementation (Part 1.3.4A) Complete")
    print("Ready for integration with remaining specialized tensor types")
```

### Expected Output for `45_specialized_tensors_part_a.mojo`

```
=== Specialized Tensor Types - Part 1.3.4A ===
Identity Matrix Implementation - Complete Standalone Module

=====================================
=== IDENTITY TENSOR TEST SUITE ===
=====================================
=== Testing Basic Identity Operations ===

1. Standard Identity Matrix Creation:
IdentityTensor[float32]
  Size: 4x4
  Scale: 1.0
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 29 bytes
  Validation: enabled

Identity matrix values (4x4):
  Row 0: 1.0 0.0 0.0 0.0 
  Row 1: 0.0 1.0 0.0 0.0 
  Row 2: 0.0 0.0 1.0 0.0 
  Row 3: 0.0 0.0 0.0 1.0 

2. Scaled Identity Matrix:
IdentityTensor[float32]
  Size: 3x3
  Scale: 2.5
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled
Scaled identity values:
  [0, 0] = 2.5
  [1, 1] = 2.5
  [2, 2] = 2.5

3. Super-diagonal Identity:
IdentityTensor[float32]
  Size: 4x4
  Scale: 3.0
  Diagonal offset: 1
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled
Super-diagonal values:
  [0, 1] = 3.0
  [1, 2] = 3.0
  [2, 3] = 3.0
Basic identity operations completed successfully

=== Testing Identity Mathematical Operations ===

1. Identity Transpose:
Original identity:
IdentityTensor[float32]
  Size: 3x3
  Scale: 2.0
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled
Transposed identity:
IdentityTensor[float32]
  Size: 3x3
  Scale: 2.0
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled

2. Identity Scaling:
IdentityTensor[float32]
  Size: 3x3
  Scale: 6.0
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled
Original scale: 2.0
New scale: 6.0

3. Diagonal Length Calculation:
Size 3 matrix:
  Main diagonal length: 3
  Super diagonal (+2) length: 1
  Sub diagonal (-1) length: 2
Size 5 matrix:
  Main diagonal length: 5
  Super diagonal (+2) length: 3
  Sub diagonal (-1) length: 4
Size 10 matrix:
  Main diagonal length: 10
  Super diagonal (+2) length: 8
  Sub diagonal (-1) length: 9
Mathematical operations completed successfully

=== Testing Identity Performance ===

1. Creation Performance:
Testing 10x10 identity creation...
Starting: Identity Creation Benchmark
Completed: Identity Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 100x100 identity creation...
Starting: Identity Creation Benchmark
Completed: Identity Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 1000x1000 identity creation...
Starting: Identity Creation Benchmark
Completed: Identity Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops

2. Memory Efficiency:
Memory analysis for 10x10 matrix:
Memory Efficiency Analysis:
  Identity tensor: 29 bytes
  Dense equivalent: 400 bytes
  Efficiency: 7.249999999999999% of dense storage
Memory analysis for 100x100 matrix:
Memory Efficiency Analysis:
  Identity tensor: 29 bytes
  Dense equivalent: 40000 bytes
  Efficiency: 0.0725% of dense storage
Memory analysis for 1000x1000 matrix:
Memory Efficiency Analysis:
  Identity tensor: 29 bytes
  Dense equivalent: 4000000 bytes
  Efficiency: 0.000725% of dense storage

3. Access Pattern Performance:
Starting: Element Access (1M operations)
Completed: Element Access (1M operations) (1000000 operations)
Sum result: 1000.0 (validation)
Performance testing completed successfully

=== Testing Identity Conversions ===

1. Identity to Dense Conversion:
Original identity:
IdentityTensor[float32]
  Size: 4x4
  Scale: 2.0
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 29 bytes
  Validation: enabled
Converted dense:
DenseTensor[float32]
  Shape: [4, 4]
  Elements: 16
  Memory usage: 64 bytes
Dense matrix values:
  Row 0: 2.0 0.0 0.0 0.0 
  Row 1: 0.0 2.0 0.0 0.0 
  Row 2: 0.0 0.0 2.0 0.0 
  Row 3: 0.0 0.0 0.0 2.0 

2. Verification of Conversion Accuracy:
Conversion accuracy: Perfect (0 errors)

3. Batch Identity Creation:
Created batch of 5 identity tensors
Each tensor in batch has:
IdentityTensor[float32]
  Size: 3x3
  Scale: 1.5
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 29 bytes
  Validation: enabled
Conversion testing completed successfully

=== Testing Identity Error Handling ===

1. Invalid Size Errors:
Correctly caught zero size error
Correctly caught negative size error
Correctly caught oversized matrix error

2. Invalid Offset Errors:
Correctly caught invalid offset error
Correctly caught positive offset error for sub-diagonal

3. Edge Case Testing:
1x1 identity matrix works correctly
Out-of-bounds access returns zero

Error Handling Summary: 7/7 tests passed
All error handling tests passed

=== Testing Identity Memory Management ===

1. Memory Usage Analysis:
  10x10 matrix:
    Identity storage: 29 bytes
    Dense would need: 100 elements
    Memory efficiency: Constant vs O(n²)
  100x100 matrix:
    Identity storage: 29 bytes
    Dense would need: 10000 elements
    Memory efficiency: Constant vs O(n²)
  1000x1000 matrix:
    Identity storage: 29 bytes
    Dense would need: 1000000 elements
    Memory efficiency: Constant vs O(n²)

2. Copy Constructor Testing:
Original:
IdentityTensor[float32]
  Size: 5x5
  Scale: 3.0
  Diagonal offset: 0
  Diagonal length: 5
  Memory usage: 29 bytes
  Validation: enabled
Copy:
IdentityTensor[float32]
  Size: 5x5
  Scale: 3.0
  Diagonal offset: 0
  Diagonal length: 5
  Memory usage: 29 bytes
  Validation: enabled
Copy constructor preserves values

3. Large Identity Creation/Destruction:
Starting: Large Identity Creation/Destruction
Completed: Large Identity Creation/Destruction (100 operations)
Large identity creation/destruction completed
Memory management testing completed successfully

=====================================
=== IDENTITY TENSOR TESTS COMPLETE ===
=====================================

=== Identity Matrix Implementation Summary ===
+ Memory-efficient identity matrix representation
+ Configurable scaling factors and diagonal offsets
+ Fast element access without full matrix storage
+ Optimized mathematical operations (transpose, scaling)
+ Comprehensive error handling and validation
+ Performance benchmarking and memory analysis
+ Seamless conversion to dense format
+ Batch identity tensor creation utilities
+ Constant memory usage regardless of matrix size
+ Integration-ready for tensor library ecosystem

Performance Characteristics:
- Creation time: O(1) - constant regardless of size
- Memory usage: O(1) - ~20 bytes per identity tensor
- Element access: O(1) - direct mathematical computation
- Conversion to dense: O(n²) - only when explicitly needed

Mathematical Features:
- Standard identity matrices (I)
- Scaled identity matrices (cI)
- Super-diagonal matrices (offset > 0)
- Sub-diagonal matrices (offset < 0)
- Transpose operations (flipped offset)
- Scaling operations (multiplicative factors)

Identity Matrix Implementation (Part 1.3.4A) Complete
Ready for integration with remaining specialized tensor types
```

### File: `45_specialized_tensors_part_b.mojo` — Section 9.2

**Run:** `pixi run mojo 45_specialized_tensors_part_b.mojo`

```mojo

**File: `45_specialized_tensors_part_b.mojo`**

```mojo

from memory import UnsafePointer
from collections import List
from sys import sizeof


# Core Tensor Infrastructure - Part 1.3.4B: Diagonal Tensor Implementation
#
# This section implements memory-efficient diagonal tensor structures optimized
# for mathematical operations involving diagonal matrices and banded structures.
# Provides specialized storage patterns that exploit sparsity in diagonal matrices
# for optimal performance and memory usage.
#
# Key Design Principles:
# - Sparse storage for diagonal elements only
# - Support for multiple diagonal bands (super/sub diagonals)
# - Fast element access and mathematical operations
# - Memory usage proportional to number of stored diagonals
# - Seamless integration with dense and identity tensors
#
# Implementation Features:
# 1. Efficient diagonal-only storage representation
# 2. Multiple diagonal band support (tridiagonal, pentadiagonal, etc.)
# 3. Fast element access with automatic zero handling
# 4. Optimized diagonal extraction and manipulation
# 5. Comprehensive validation and bounds checking
#
# Mathematical Properties Exploited:
# - Diagonal matrices are highly sparse with known structure
# - Only non-zero diagonals need explicit storage
# - Matrix operations can leverage diagonal structure
# - Memory scales with number of diagonals, not matrix size

alias DEFAULT_DIAGONAL_SIZE = 100
alias MAX_DIAGONAL_SIZE = 10000
alias MAX_DIAGONAL_BANDS = 10


struct DiagonalTensor[dtype: DType]:
    """Memory-efficient diagonal tensor with sparse diagonal storage."""
    var diagonal_data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var offset: Int  # Diagonal offset (0 = main, +n = super, -n = sub)
    var diagonal_length: Int
    var _owns_data: Bool
    var _validation_enabled: Bool
    
    fn __init__(out self, size: Int, offset: Int = 0) raises:
        """Initialize diagonal tensor with specified size and offset."""
        if size <= 0:
            raise Error("Diagonal tensor size must be positive")
        if size > MAX_DIAGONAL_SIZE:
            raise Error("Diagonal tensor size exceeds maximum allowed")
        if abs(offset) >= size:
            raise Error("Diagonal offset must be smaller than matrix size")
        
        self.size = size
        self.offset = offset
        self._owns_data = True
        self._validation_enabled = True
        
        # Calculate diagonal length based on offset
        if offset >= 0:
            self.diagonal_length = max(0, size - offset)
        else:
            self.diagonal_length = max(0, size + offset)
        
        if self.diagonal_length <= 0:
            raise Error("Invalid diagonal configuration")
        
        # Allocate memory only for diagonal elements
        self.diagonal_data = UnsafePointer[Scalar[dtype]].alloc(self.diagonal_length)
        
        # Initialize to zero
        for i in range(self.diagonal_length):
            self.diagonal_data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for diagonal tensor."""
        self.size = existing.size
        self.offset = existing.offset
        self.diagonal_length = existing.diagonal_length
        self._owns_data = True
        self._validation_enabled = existing._validation_enabled
        
        # Allocate and copy diagonal data
        self.diagonal_data = UnsafePointer[Scalar[dtype]].alloc(self.diagonal_length)
        for i in range(self.diagonal_length):
            self.diagonal_data[i] = existing.diagonal_data[i]
    
    fn __del__(owned self):
        """Destructor to free allocated memory."""
        if self._owns_data:
            self.diagonal_data.free()
    
    fn get_diagonal_index(self, row: Int, col: Int) -> Int:
        """Convert matrix position to diagonal index, returns -1 if not on diagonal."""
        if col - row != self.offset:
            return -1  # Not on the target diagonal
        
        if self.offset >= 0:
            return row
        else:
            return col
    
    fn is_on_diagonal(self, row: Int, col: Int) -> Bool:
        """Check if position is on the stored diagonal."""
        return col - row == self.offset
    
    fn get_item(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element at specified matrix position."""
        if self._validation_enabled:
            if row < 0 or row >= self.size or col < 0 or col >= self.size:
                return Scalar[dtype](0)
        
        var diag_index = self.get_diagonal_index(row, col)
        if diag_index >= 0 and diag_index < self.diagonal_length:
            return self.diagonal_data[diag_index]
        else:
            return Scalar[dtype](0)
    
    fn get_item_unsafe(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element without bounds checking for performance-critical code."""
        var diag_index = self.get_diagonal_index(row, col)
        if diag_index >= 0:
            return self.diagonal_data[diag_index]
        else:
            return Scalar[dtype](0)
    
    fn set_diagonal_element(mut self, index: Int, value: Scalar[dtype]) raises:
        """Set diagonal element at specified diagonal index."""
        if self._validation_enabled:
            if index < 0 or index >= self.diagonal_length:
                raise Error("Diagonal index out of bounds")
        
        self.diagonal_data[index] = value
    
    fn get_diagonal_element(self, index: Int) -> Scalar[dtype]:
        """Get diagonal element at specified diagonal index."""
        if self._validation_enabled:
            if index < 0 or index >= self.diagonal_length:
                return Scalar[dtype](0)
        
        return self.diagonal_data[index]
    
    fn set_item(mut self, row: Int, col: Int, value: Scalar[dtype]) raises:
        """Set element at specified matrix position."""
        if self._validation_enabled:
            if row < 0 or row >= self.size or col < 0 or col >= self.size:
                raise Error("Matrix indices out of bounds")
        
        var diag_index = self.get_diagonal_index(row, col)
        if diag_index >= 0 and diag_index < self.diagonal_length:
            self.diagonal_data[diag_index] = value
        else:
            if value != Scalar[dtype](0):
                raise Error("Cannot set non-zero value outside diagonal")
    
    fn fill_diagonal(mut self, value: Scalar[dtype]):
        """Fill entire diagonal with specified value."""
        for i in range(self.diagonal_length):
            self.diagonal_data[i] = value
    
    fn scale_diagonal(mut self, factor: Scalar[dtype]):
        """Scale all diagonal elements by factor."""
        for i in range(self.diagonal_length):
            self.diagonal_data[i] *= factor
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        var base_size = 4 * sizeof[Int]() + 2 * sizeof[Bool]()
        var data_size = self.diagonal_length * sizeof[Scalar[dtype]]()
        return base_size + data_size
    
    fn get_sparsity(self) -> Float64:
        """Calculate sparsity ratio (stored elements / total elements)."""
        var total_elements = self.size * self.size
        if total_elements == 0:
            return 0.0
        return Float64(self.diagonal_length) / Float64(total_elements)
    
    fn set_validation(mut self, enabled: Bool):
        """Enable or disable bounds checking for performance tuning."""
        self._validation_enabled = enabled
    
    fn transpose(self) raises -> DiagonalTensor[dtype]:
        """Return transposed diagonal tensor (flipped offset)."""
        var transposed = DiagonalTensor[dtype](self.size, -self.offset)
        
        # Copy diagonal data
        for i in range(self.diagonal_length):
            transposed.set_diagonal_element(i, self.diagonal_data[i])
        
        return transposed
    
    fn to_dense(self) raises -> DenseTensor[dtype]:
        """Convert to dense tensor representation."""
        var shape = List[Int]()
        shape.append(self.size)
        shape.append(self.size)
        
        var dense = DenseTensor[dtype](shape)
        
        # Fill only diagonal elements
        for i in range(self.size):
            for j in range(self.size):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = self.get_item_unsafe(i, j)
                dense.set_item(indices, value)
        
        return dense
    
    fn print_info(self):
        """Print diagonal tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var size_str: String = String(self.size)
        var offset_str: String = String(self.offset)
        var diag_len_str: String = String(self.diagonal_length)
        var memory_str: String = String(self.get_memory_usage())
        var sparsity_str: String = String(self.get_sparsity() * 100.0)
        
        print("DiagonalTensor[" + dtype_str + "]")
        print("  Size: " + size_str + "x" + size_str)
        print("  Diagonal offset: " + offset_str)
        print("  Diagonal length: " + diag_len_str)
        print("  Memory usage: " + memory_str + " bytes")
        print("  Sparsity: " + sparsity_str + "% (stored/total)")
        print("  Validation: " + ("enabled" if self._validation_enabled else "disabled"))


struct MultiBandDiagonal[dtype: DType]:
    """Multi-diagonal tensor supporting multiple diagonal bands."""
    var diagonals: UnsafePointer[DiagonalTensor[dtype]]
    var offsets: UnsafePointer[Int]
    var num_diagonals: Int
    var size: Int
    var _owns_data: Bool
    
    fn __init__(out self, size: Int, offsets: List[Int]) raises:
        """Initialize multi-band diagonal tensor."""
        if size <= 0:
            raise Error("Multi-diagonal tensor size must be positive")
        if len(offsets) == 0:
            raise Error("Must specify at least one diagonal offset")
        if len(offsets) > MAX_DIAGONAL_BANDS:
            raise Error("Too many diagonal bands")
        
        self.size = size
        self.num_diagonals = len(offsets)
        self._owns_data = True
        
        # Allocate memory for diagonal tensors and offsets
        self.diagonals = UnsafePointer[DiagonalTensor[dtype]].alloc(self.num_diagonals)
        self.offsets = UnsafePointer[Int].alloc(self.num_diagonals)
        
        # Initialize each diagonal
        for i in range(self.num_diagonals):
            var offset = offsets[i]
            self.offsets[i] = offset
            self.diagonals[i] = DiagonalTensor[dtype](size, offset)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for multi-band diagonal tensor."""
        self.size = existing.size
        self.num_diagonals = existing.num_diagonals
        self._owns_data = True
        
        # Allocate and copy arrays
        self.diagonals = UnsafePointer[DiagonalTensor[dtype]].alloc(self.num_diagonals)
        self.offsets = UnsafePointer[Int].alloc(self.num_diagonals)
        
        for i in range(self.num_diagonals):
            self.offsets[i] = existing.offsets[i]
            self.diagonals[i] = existing.diagonals[i]
    
    fn __del__(owned self):
        """Destructor to free allocated memory."""
        if self._owns_data:
            self.diagonals.free()
            self.offsets.free()
    
    fn get_item(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element at specified matrix position."""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return Scalar[dtype](0)
        
        var offset = col - row
        
        # Search for matching diagonal
        for i in range(self.num_diagonals):
            if self.offsets[i] == offset:
                return self.diagonals[i].get_item(row, col)
        
        return Scalar[dtype](0)
    
    fn set_diagonal_value(mut self, diagonal_index: Int, element_index: Int, value: Scalar[dtype]) raises:
        """Set value in specified diagonal."""
        if diagonal_index < 0 or diagonal_index >= self.num_diagonals:
            raise Error("Diagonal index out of bounds")
        
        self.diagonals[diagonal_index].set_diagonal_element(element_index, value)
    
    fn get_total_memory_usage(self) -> Int:
        """Get total memory usage for all diagonals."""
        var total = sizeof[Int]() * 3 + sizeof[Bool]()  # Base struct size
        
        for i in range(self.num_diagonals):
            total += self.diagonals[i].get_memory_usage()
        
        return total
    
    fn print_info(self):
        """Print multi-diagonal tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var size_str: String = String(self.size)
        var num_diag_str: String = String(self.num_diagonals)
        var memory_str: String = String(self.get_total_memory_usage())
        
        print("MultiBandDiagonal[" + dtype_str + "]")
        print("  Size: " + size_str + "x" + size_str)
        print("  Number of diagonals: " + num_diag_str)
        print("  Total memory usage: " + memory_str + " bytes")
        
        print("  Diagonal offsets:")
        for i in range(self.num_diagonals):
            var offset_str: String = String(self.offsets[i])
            var diag_len_str: String = String(self.diagonals[i].diagonal_length)
            print("    [" + String(i) + "] offset=" + offset_str + ", length=" + diag_len_str)


struct DenseTensor[dtype: DType]:
    """Dense tensor implementation for conversion and compatibility."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1
        self._owns_data = True
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
            self.shape.append(shape[i])
            self._total_elements *= shape[i]
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for dense tensor."""
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        self.ndim = existing.ndim
        self._total_elements = existing._total_elements
        self._owns_data = True
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
    
    fn _get_linear_index(self, indices: List[Int]) -> Int:
        """Convert multi-dimensional indices to linear index."""
        var linear_index = 0
        var stride = 1
        
        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]
        
        return linear_index
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        return self.data[linear_index]
    
    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        self.data[linear_index] = value
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        return self._total_elements
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        return self._total_elements * sizeof[Scalar[dtype]]()
    
    fn print_info(self):
        """Print tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var memory_str: String = String(self.get_memory_usage())
        
        print("DenseTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)
        print("  Memory usage: " + memory_str + " bytes")


fn create_diagonal[dtype: DType](size: Int, values: List[Scalar[dtype]], offset: Int = 0) raises -> DiagonalTensor[dtype]:
    """Create diagonal tensor with specified values."""
    var tensor = DiagonalTensor[dtype](size, offset)
    
    var max_elements = min(len(values), tensor.diagonal_length)
    for i in range(max_elements):
        tensor.set_diagonal_element(i, values[i])
    
    return tensor

fn create_main_diagonal[dtype: DType](size: Int, value: Scalar[dtype] = Scalar[dtype](1)) raises -> DiagonalTensor[dtype]:
    """Create main diagonal tensor with uniform value."""
    var tensor = DiagonalTensor[dtype](size, 0)
    tensor.fill_diagonal(value)
    return tensor

fn create_super_diagonal[dtype: DType](size: Int, offset: Int, value: Scalar[dtype] = Scalar[dtype](1)) raises -> DiagonalTensor[dtype]:
    """Create super-diagonal tensor."""
    if offset <= 0:
        raise Error("Super-diagonal offset must be positive")
    
    var tensor = DiagonalTensor[dtype](size, offset)
    tensor.fill_diagonal(value)
    return tensor

fn create_sub_diagonal[dtype: DType](size: Int, offset: Int, value: Scalar[dtype] = Scalar[dtype](1)) raises -> DiagonalTensor[dtype]:
    """Create sub-diagonal tensor."""
    if offset >= 0:
        raise Error("Sub-diagonal offset must be negative")
    
    var tensor = DiagonalTensor[dtype](size, offset)
    tensor.fill_diagonal(value)
    return tensor

fn create_tridiagonal[dtype: DType](size: Int, main_val: Scalar[dtype], super_val: Scalar[dtype], sub_val: Scalar[dtype]) raises -> MultiBandDiagonal[dtype]:
    """Create tridiagonal matrix."""
    var offsets = List[Int]()
    offsets.append(-1)  # Sub-diagonal
    offsets.append(0)   # Main diagonal
    offsets.append(1)   # Super-diagonal
    
    var multi = MultiBandDiagonal[dtype](size, offsets)
    
    # Fill diagonals
    var main_diag = create_main_diagonal[dtype](size, main_val)
    var super_diag = create_super_diagonal[dtype](size, 1, super_val)
    var sub_diag = create_sub_diagonal[dtype](size, -1, sub_val)
    
    # Set values
    for i in range(main_diag.diagonal_length):
        multi.set_diagonal_value(1, i, main_val)  # Main diagonal is index 1
    
    for i in range(super_diag.diagonal_length):
        multi.set_diagonal_value(2, i, super_val)  # Super diagonal is index 2
    
    for i in range(sub_diag.diagonal_length):
        multi.set_diagonal_value(0, i, sub_val)  # Sub diagonal is index 0
    
    return multi

fn dense_to_diagonal[dtype: DType](dense: DenseTensor[dtype], offset: Int = 0, threshold: Float64 = 1e-10) raises -> DiagonalTensor[dtype]:
    """Extract diagonal from dense tensor."""
    if dense.ndim != 2:
        raise Error("Can only extract diagonal from 2D tensors")
    if dense.shape[0] != dense.shape[1]:
        raise Error("Matrix must be square for diagonal extraction")
    
    var size = dense.shape[0]
    var diagonal = DiagonalTensor[dtype](size, offset)
    
    # Extract diagonal elements
    var diag_index = 0
    for i in range(size):
        var j = i + offset
        if j >= 0 and j < size:
            var indices = List[Int]()
            indices.append(i)
            indices.append(j)
            var value = dense.get_item(indices)
            
            if abs(Float64(value)) > threshold:
                diagonal.set_diagonal_element(diag_index, value)
            
            diag_index += 1
    
    return diagonal


struct PerformanceTimer:
    """Simple performance timing utility."""
    var name: String
    
    fn __init__(out self, name: String):
        self.name = name
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
    
    fn start(self):
        """Start timing operation."""
        print("Starting: " + self.name)
    
    fn end(self, operations: Int = 1):
        """End timing and report results."""
        var ops_str: String = String(operations)
        print("Completed: " + self.name + " (" + ops_str + " operations)")

fn benchmark_diagonal_creation[dtype: DType](size: Int, iterations: Int) -> Float64:
    """Benchmark diagonal tensor creation performance."""
    var timer = PerformanceTimer("Diagonal Creation Benchmark")
    timer.start()
    
    try:
        for _ in range(iterations):
            var diagonal = create_main_diagonal[dtype](size, Scalar[dtype](1))
            var _ = diagonal.get_item(0, 0)  # Force usage
    except:
        print("Benchmark failed")
        return -1.0
    
    timer.end(iterations)
    return Float64(iterations) / 1000.0

fn benchmark_diagonal_memory_efficiency[dtype: DType](size: Int) -> Float64:
    """Compare memory usage: diagonal vs dense."""
    try:
        var diagonal = create_main_diagonal[dtype](size, Scalar[dtype](1))
        var dense = diagonal.to_dense()
        
        var diagonal_memory = diagonal.get_memory_usage()
        var dense_memory = dense.get_memory_usage()
        
        var efficiency = Float64(diagonal_memory) / Float64(dense_memory)
        
        var diagonal_str: String = String(diagonal_memory)
        var dense_str: String = String(dense_memory)
        var efficiency_str: String = String(efficiency * 100.0)
        
        print("Memory Efficiency Analysis:")
        print("  Diagonal tensor: " + diagonal_str + " bytes")
        print("  Dense equivalent: " + dense_str + " bytes")
        print("  Efficiency: " + efficiency_str + "% of dense storage")
        
        return efficiency
        
    except:
        print("Memory benchmark failed")
        return -1.0


fn test_basic_diagonal_operations():
    """Test basic diagonal tensor functionality."""
    print("=== Testing Basic Diagonal Operations ===")
    
    try:
        print("\n1. Main Diagonal Tensor Creation:")
        var main_diag = create_main_diagonal[DType.float32](4, 2.0)
        main_diag.print_info()
        
        print("\nMain diagonal matrix values (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = main_diag.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Super-diagonal Tensor:")
        var super_diag = create_super_diagonal[DType.float32](4, 1, 3.0)
        super_diag.print_info()
        
        print("Super-diagonal values:")
        for i in range(4):
            for j in range(4):
                var value = super_diag.get_item(i, j)
                if value != Scalar[DType.float32](0):
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var value_str: String = String(value)
                    print("  [" + i_str + ", " + j_str + "] = " + value_str)
        
        print("\n3. Sub-diagonal Tensor:")
        var sub_diag = create_sub_diagonal[DType.float32](4, -1, -1.5)
        sub_diag.print_info()
        
        print("Sub-diagonal values:")
        for i in range(4):
            for j in range(4):
                var value = sub_diag.get_item(i, j)
                if value != Scalar[DType.float32](0):
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var value_str: String = String(value)
                    print("  [" + i_str + ", " + j_str + "] = " + value_str)
        
        print("Basic diagonal operations completed successfully")
        
    except:
        print("Error in basic diagonal operations")

fn test_diagonal_mathematical_operations():
    """Test mathematical operations on diagonal tensors."""
    print("\n=== Testing Diagonal Mathematical Operations ===")
    
    try:
        print("\n1. Diagonal Scaling:")
        var diagonal = create_main_diagonal[DType.float32](3, 2.0)
        print("Original diagonal:")
        diagonal.print_info()
        
        diagonal.scale_diagonal(1.5)
        print("After scaling by 1.5:")
        diagonal.print_info()
        
        print("Scaled diagonal values:")
        for i in range(diagonal.diagonal_length):
            var value = diagonal.get_diagonal_element(i)
            var i_str: String = String(i)
            var value_str: String = String(value)
            print("  Diagonal[" + i_str + "] = " + value_str)
        
        print("\n2. Diagonal Transpose:")
        var original = create_super_diagonal[DType.float32](3, 1, 5.0)
        var transposed = original.transpose()
        
        print("Original super-diagonal:")
        original.print_info()
        print("Transposed (becomes sub-diagonal):")
        transposed.print_info()
        
        print("\n3. Custom Diagonal Values:")
        var values = List[Scalar[DType.float32]]()
        values.append(1.0)
        values.append(4.0)
        values.append(9.0)
        values.append(16.0)
        
        var custom_diag = create_diagonal[DType.float32](4, values, 0)
        custom_diag.print_info()
        
        print("Custom diagonal values:")
        for i in range(custom_diag.diagonal_length):
            var value = custom_diag.get_diagonal_element(i)
            var i_str: String = String(i)
            var value_str: String = String(value)
            print("  Diagonal[" + i_str + "] = " + value_str)
        
        print("Mathematical operations completed successfully")
        
    except:
        print("Error in mathematical operations")

fn test_multi_diagonal_operations():
    """Test multi-diagonal tensor functionality."""
    print("\n=== Testing Multi-Diagonal Operations ===")
    
    try:
        print("\n1. Tridiagonal Matrix Creation:")
        var tridiag = create_tridiagonal[DType.float32](4, 2.0, 1.0, -1.0)
        tridiag.print_info()
        
        print("Tridiagonal matrix values (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = tridiag.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Multi-diagonal Memory Usage:")
        var total_memory = tridiag.get_total_memory_usage()
        var total_str: String = String(total_memory)
        print("Total memory usage: " + total_str + " bytes")
        
        # Compare with dense equivalent
        var dense_elements = 4 * 4 * sizeof[Scalar[DType.float32]]()
        var dense_str: String = String(dense_elements)
        var efficiency = Float64(total_memory) / Float64(dense_elements) * 100.0
        var efficiency_str: String = String(efficiency)
        print("Dense equivalent: " + dense_str + " bytes")
        print("Memory efficiency: " + efficiency_str + "% of dense storage")
        
        print("Multi-diagonal operations completed successfully")
        
    except:
        print("Error in multi-diagonal operations")

fn test_diagonal_performance():
    """Test diagonal tensor performance characteristics."""
    print("\n=== Testing Diagonal Performance ===")
    
    try:
        print("\n1. Creation Performance:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(1000)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var iterations = 1000
            var size_str: String = String(size)
            print("Testing " + size_str + "x" + size_str + " diagonal creation...")
            
            var result = benchmark_diagonal_creation[DType.float32](size, iterations)
            var result_str: String = String(result)
            print("  Performance metric: " + result_str + " k-ops")
        
        print("\n2. Memory Efficiency Comparison:")
        for i in range(len(sizes)):
            var size = sizes[i]
            var size_str: String = String(size)
            print("Memory analysis for " + size_str + "x" + size_str + " matrix:")
            var _ = benchmark_diagonal_memory_efficiency[DType.float32](size)
        
        print("\n3. Access Pattern Performance:")
        var large_diagonal = create_main_diagonal[DType.float32](1000, 2.0)
        large_diagonal.set_validation(False)  # Disable bounds checking
        
        var timer = PerformanceTimer("Element Access (1M operations)")
        timer.start()
        
        var sum = Scalar[DType.float32](0)
        for i in range(1000):
            for j in range(1000):
                sum += large_diagonal.get_item_unsafe(i, j)
        
        timer.end(1000000)
        var sum_str: String = String(sum)
        print("Sum result: " + sum_str + " (validation)")
        
        print("\n4. Diagonal Element Access Performance:")
        var diag_timer = PerformanceTimer("Diagonal Element Access (100k operations)")
        diag_timer.start()
        
        var diag_sum = Scalar[DType.float32](0)
        for _ in range(100000):
            for i in range(large_diagonal.diagonal_length):
                diag_sum += large_diagonal.get_diagonal_element(i)
        
        diag_timer.end(100000)
        var diag_sum_str: String = String(diag_sum)
        print("Diagonal sum result: " + diag_sum_str + " (validation)")
        
        print("Performance testing completed successfully")
        
    except:
        print("Error in performance testing")

fn test_diagonal_conversions():
    """Test conversions between diagonal and other formats."""
    print("\n=== Testing Diagonal Conversions ===")
    
    try:
        print("\n1. Diagonal to Dense Conversion:")
        var values = List[Scalar[DType.float32]]()
        values.append(1.0)
        values.append(4.0)
        values.append(9.0)
        values.append(16.0)
        
        var diagonal = create_diagonal[DType.float32](4, values, 0)
        var dense = diagonal.to_dense()
        
        print("Original diagonal:")
        diagonal.print_info()
        print("Converted dense:")
        dense.print_info()
        
        print("Dense matrix values:")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = dense.get_item(indices)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Dense to Diagonal Extraction:")
        var extracted = dense_to_diagonal[DType.float32](dense, 0)
        extracted.print_info()
        
        print("Extracted diagonal values:")
        for i in range(extracted.diagonal_length):
            var value = extracted.get_diagonal_element(i)
            var i_str: String = String(i)
            var value_str: String = String(value)
            print("  Diagonal[" + i_str + "] = " + value_str)
        
        print("\n3. Conversion Accuracy Verification:")
        var errors = 0
        for i in range(diagonal.diagonal_length):
            var original_val = diagonal.get_diagonal_element(i)
            var extracted_val = extracted.get_diagonal_element(i)
            
            if original_val != extracted_val:
                errors += 1
                var i_str: String = String(i)
                var orig_str: String = String(original_val)
                var extr_str: String = String(extracted_val)
                print("  Mismatch at diagonal[" + i_str + "]: " + orig_str + " vs " + extr_str)
        
        var errors_str: String = String(errors)
        if errors == 0:
            print("Conversion accuracy: Perfect (0 errors)")
        else:
            print("Conversion errors: " + errors_str)
        
        print("Conversion testing completed successfully")
        
    except:
        print("Error in conversion testing")

fn test_diagonal_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Diagonal Error Handling ===")
    
    var test_count = 0
    var passed_count = 0
    
    print("\n1. Invalid Size Errors:")
    test_count += 1
    try:
        var _ = create_main_diagonal[DType.float32](0, 1.0)
        print("Should have failed with zero size")
    except:
        print("Correctly caught zero size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_main_diagonal[DType.float32](-5, 1.0)
        print("Should have failed with negative size")
    except:
        print("Correctly caught negative size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_main_diagonal[DType.float32](20000, 1.0)
        print("Should have failed with oversized matrix")
    except:
        print("Correctly caught oversized matrix error")
        passed_count += 1
    
    print("\n2. Invalid Offset Errors:")
    test_count += 1
    try:
        var _ = DiagonalTensor[DType.float32](5, 5)
        print("Should have failed with offset >= size")
    except:
        print("Correctly caught invalid offset error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_super_diagonal[DType.float32](3, 0, 1.0)
        print("Should have failed with zero offset for super-diagonal")
    except:
        print("Correctly caught zero offset error for super-diagonal")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_sub_diagonal[DType.float32](3, 1, 1.0)
        print("Should have failed with positive offset for sub-diagonal")
    except:
        print("Correctly caught positive offset error for sub-diagonal")
        passed_count += 1
    
    print("\n3. Bounds Checking:")
    test_count += 1
    try:
        var diagonal = create_main_diagonal[DType.float32](3, 1.0)
        diagonal.set_diagonal_element(5, 2.0)  # Out of bounds
        print("Should have failed with out-of-bounds diagonal access")
    except:
        print("Correctly caught out-of-bounds diagonal access")
        passed_count += 1
    
    test_count += 1
    try:
        var diagonal = create_main_diagonal[DType.float32](3, 1.0)
        diagonal.set_item(1, 2, 3.0)  # Off-diagonal non-zero
        print("Should have failed with off-diagonal non-zero assignment")
    except:
        print("Correctly caught off-diagonal non-zero assignment")
        passed_count += 1
    
    print("\n4. Edge Case Testing:")
    test_count += 1
    try:
        var tiny = DiagonalTensor[DType.float32](1, 0)
        var value = tiny.get_item(0, 0)
        if value == Scalar[DType.float32](0):  # Should be zero initially
            print("1x1 diagonal tensor works correctly")
            passed_count += 1
        else:
            print("1x1 diagonal tensor incorrect initial value")
    except:
        print("1x1 diagonal tensor creation failed")
    
    test_count += 1
    try:
        var diagonal = create_main_diagonal[DType.float32](3, 1.0)
        var out_of_bounds = diagonal.get_item(5, 5)
        if out_of_bounds == Scalar[DType.float32](0):
            print("Out-of-bounds access returns zero")
            passed_count += 1
        else:
            print("Out-of-bounds access incorrect value")
    except:
        print("Out-of-bounds access handling failed")
    
    var passed_str: String = String(passed_count)
    var total_str: String = String(test_count)
    print("\nError Handling Summary: " + passed_str + "/" + total_str + " tests passed")
    
    if passed_count == test_count:
        print("All error handling tests passed")
    else:
        print("Some error handling tests failed")

fn test_diagonal_memory_management():
    """Test memory management and resource cleanup."""
    print("\n=== Testing Diagonal Memory Management ===")
    
    try:
        print("\n1. Memory Usage Analysis:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(1000)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var diagonal = create_main_diagonal[DType.float32](size, 1.0)
            var memory = diagonal.get_memory_usage()
            var sparsity = diagonal.get_sparsity()
            
            var size_str: String = String(size)
            var memory_str: String = String(memory)
            var sparsity_str: String = String(sparsity * 100.0)
            
            print("  " + size_str + "x" + size_str + " matrix:")
            print("    Diagonal storage: " + memory_str + " bytes")
            print("    Sparsity: " + sparsity_str + "% stored/total")
            print("    Memory efficiency: O(n) vs O(n²) for dense")
        
        print("\n2. Copy Constructor Testing:")
        var original = create_diagonal[DType.float32](5, List[Scalar[DType.float32]](), 0)
        original.fill_diagonal(3.0)
        var copied = original
        
        print("Original:")
        original.print_info()
        print("Copy:")
        copied.print_info()
        
        # Verify independence by modifying original
        original.scale_diagonal(2.0)
        
        var orig_val = original.get_diagonal_element(0)
        var copy_val = copied.get_diagonal_element(0)
        
        if orig_val != copy_val:
            print("Copy constructor creates independent copy")
        else:
            print("Copy constructor may not be independent")
        
        print("\n3. Multi-diagonal Memory Management:")
        var offsets = List[Int]()
        offsets.append(-1)
        offsets.append(0)
        offsets.append(1)
        
        var multi = MultiBandDiagonal[DType.float32](100, offsets)
        var multi_memory = multi.get_total_memory_usage()
        var multi_memory_str: String = String(multi_memory)
        
        print("Multi-diagonal memory usage: " + multi_memory_str + " bytes")
        print("Stores 3 diagonals efficiently")
        
        print("\n4. Large Diagonal Creation/Destruction:")
        var creation_timer = PerformanceTimer("Large Diagonal Creation/Destruction")
        creation_timer.start()
        
        for _ in range(100):
            var large_diagonal = create_main_diagonal[DType.float32](1000, 1.0)
            var _ = large_diagonal.get_item(0, 0)  # Force usage
        
        creation_timer.end(100)
        print("Large diagonal creation/destruction completed")
        
        print("Memory management testing completed successfully")
        
    except:
        print("Error in memory management testing")

fn run_all_diagonal_tests():
    """Run comprehensive test suite for diagonal tensors."""
    print("=====================================")
    print("=== DIAGONAL TENSOR TEST SUITE ===")
    print("=====================================")
    
    test_basic_diagonal_operations()
    test_diagonal_mathematical_operations()
    test_multi_diagonal_operations()
    test_diagonal_performance()
    test_diagonal_conversions()
    test_diagonal_error_handling()
    test_diagonal_memory_management()
    
    print("\n=====================================")
    print("=== DIAGONAL TENSOR TESTS COMPLETE ===")
    print("=====================================")


fn main():
    """Main function to run diagonal tensor implementation tests."""
    print("=== Specialized Tensor Types - Part 1.3.4B ===")
    print("Diagonal Tensor Implementation - Complete Standalone Module")
    print("")
    
    run_all_diagonal_tests()
    
    print("\n=== Diagonal Tensor Implementation Summary ===")
    print("+ Memory-efficient diagonal tensor representation")
    print("+ Support for multiple diagonal bands (main, super, sub)")
    print("+ Sparse storage proportional to diagonal count, not matrix size")
    print("+ Fast element access with automatic zero handling")
    print("+ Multi-diagonal tensor support (tridiagonal, pentadiagonal)")
    print("+ Optimized mathematical operations (scaling, transpose)")
    print("+ Comprehensive error handling and validation")
    print("+ Performance benchmarking and memory analysis")
    print("+ Seamless conversion to/from dense format")
    print("+ Diagonal extraction from dense matrices")
    print("+ Integration-ready for tensor library ecosystem")
    print("")
    print("Performance Characteristics:")
    print("- Creation time: O(n) - linear in diagonal length")
    print("- Memory usage: O(n) - proportional to diagonal elements")
    print("- Element access: O(1) - direct diagonal computation")
    print("- Conversion to dense: O(n²) - only when explicitly needed")
    print("- Multi-diagonal: O(k*n) where k is number of diagonals")
    print("")
    print("Mathematical Features:")
    print("- Main diagonal matrices")
    print("- Super-diagonal matrices (offset > 0)")
    print("- Sub-diagonal matrices (offset < 0)")
    print("- Multi-band diagonal matrices")
    print("- Tridiagonal matrix creation utilities")
    print("- Diagonal scaling and transpose operations")
    print("- Dense matrix diagonal extraction")
    print("")
    print("Diagonal Tensor Implementation (Part 1.3.4B) Complete")
    print("Ready for integration with remaining specialized tensor types")
```

### Expected Output for `45_specialized_tensors_part_b.mojo`

```
=== Specialized Tensor Types - Part 1.3.4B ===
Diagonal Tensor Implementation - Complete Standalone Module

=====================================
=== DIAGONAL TENSOR TEST SUITE ===
=====================================
=== Testing Basic Diagonal Operations ===

1. Main Diagonal Tensor Creation:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 50 bytes
  Sparsity: 25.0% (stored/total)
  Validation: enabled

Main diagonal matrix values (4x4):
  Row 0: 2.0 0.0 0.0 0.0 
  Row 1: 0.0 2.0 0.0 0.0 
  Row 2: 0.0 0.0 2.0 0.0 
  Row 3: 0.0 0.0 0.0 2.0 

2. Super-diagonal Tensor:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: 1
  Diagonal length: 3
  Memory usage: 46 bytes
  Sparsity: 18.75% (stored/total)
  Validation: enabled
Super-diagonal values:
  [0, 1] = 3.0
  [1, 2] = 3.0
  [2, 3] = 3.0

3. Sub-diagonal Tensor:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: -1
  Diagonal length: 3
  Memory usage: 46 bytes
  Sparsity: 18.75% (stored/total)
  Validation: enabled
Sub-diagonal values:
  [1, 0] = -1.5
  [2, 1] = -1.5
  [3, 2] = -1.5
Basic diagonal operations completed successfully

=== Testing Diagonal Mathematical Operations ===

1. Diagonal Scaling:
Original diagonal:
DiagonalTensor[float32]
  Size: 3x3
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 46 bytes
  Sparsity: 33.33333333333333% (stored/total)
  Validation: enabled
After scaling by 1.5:
DiagonalTensor[float32]
  Size: 3x3
  Diagonal offset: 0
  Diagonal length: 3
  Memory usage: 46 bytes
  Sparsity: 33.33333333333333% (stored/total)
  Validation: enabled
Scaled diagonal values:
  Diagonal[0] = 3.0
  Diagonal[1] = 3.0
  Diagonal[2] = 3.0

2. Diagonal Transpose:
Original super-diagonal:
DiagonalTensor[float32]
  Size: 3x3
  Diagonal offset: 1
  Diagonal length: 2
  Memory usage: 42 bytes
  Sparsity: 22.22222222222222% (stored/total)
  Validation: enabled
Transposed (becomes sub-diagonal):
DiagonalTensor[float32]
  Size: 3x3
  Diagonal offset: -1
  Diagonal length: 2
  Memory usage: 42 bytes
  Sparsity: 22.22222222222222% (stored/total)
  Validation: enabled

3. Custom Diagonal Values:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 50 bytes
  Sparsity: 25.0% (stored/total)
  Validation: enabled
Custom diagonal values:
  Diagonal[0] = 1.0
  Diagonal[1] = 4.0
  Diagonal[2] = 9.0
  Diagonal[3] = 16.0
Mathematical operations completed successfully

=== Testing Multi-Diagonal Operations ===

1. Tridiagonal Matrix Creation:
MultiBandDiagonal[float32]
  Size: 4x4
  Number of diagonals: 3
  Total memory usage: 167 bytes
  Diagonal offsets:
    [0] offset=-1, length=3
    [1] offset=0, length=4
    [2] offset=1, length=3
Tridiagonal matrix values (4x4):
  Row 0: 2.0 1.0 0.0 0.0 
  Row 1: -1.0 2.0 1.0 0.0 
  Row 2: 0.0 -1.0 2.0 1.0 
  Row 3: 0.0 0.0 -1.0 2.0 

2. Multi-diagonal Memory Usage:
Total memory usage: 167 bytes
Dense equivalent: 64 bytes
Memory efficiency: 260.9375% of dense storage
Multi-diagonal operations completed successfully

=== Testing Diagonal Performance ===

1. Creation Performance:
Testing 10x10 diagonal creation...
Starting: Diagonal Creation Benchmark
Completed: Diagonal Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 100x100 diagonal creation...
Starting: Diagonal Creation Benchmark
Completed: Diagonal Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 1000x1000 diagonal creation...
Starting: Diagonal Creation Benchmark
Completed: Diagonal Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops

2. Memory Efficiency Comparison:
Memory analysis for 10x10 matrix:
Memory Efficiency Analysis:
  Diagonal tensor: 74 bytes
  Dense equivalent: 400 bytes
  Efficiency: 18.5% of dense storage
Memory analysis for 100x100 matrix:
Memory Efficiency Analysis:
  Diagonal tensor: 434 bytes
  Dense equivalent: 40000 bytes
  Efficiency: 1.085% of dense storage
Memory analysis for 1000x1000 matrix:
Memory Efficiency Analysis:
  Diagonal tensor: 4034 bytes
  Dense equivalent: 4000000 bytes
  Efficiency: 0.10085000000000001% of dense storage

3. Access Pattern Performance:
Starting: Element Access (1M operations)
Completed: Element Access (1M operations) (1000000 operations)
Sum result: 2000.0 (validation)

4. Diagonal Element Access Performance:
Starting: Diagonal Element Access (100k operations)
Completed: Diagonal Element Access (100k operations) (100000 operations)
Diagonal sum result: 33554432.0 (validation)
Performance testing completed successfully

=== Testing Diagonal Conversions ===

1. Diagonal to Dense Conversion:
Original diagonal:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 50 bytes
  Sparsity: 25.0% (stored/total)
  Validation: enabled
Converted dense:
DenseTensor[float32]
  Shape: [4, 4]
  Elements: 16
  Memory usage: 64 bytes
Dense matrix values:
  Row 0: 1.0 0.0 0.0 0.0 
  Row 1: 0.0 4.0 0.0 0.0 
  Row 2: 0.0 0.0 9.0 0.0 
  Row 3: 0.0 0.0 0.0 16.0 

2. Dense to Diagonal Extraction:
DiagonalTensor[float32]
  Size: 4x4
  Diagonal offset: 0
  Diagonal length: 4
  Memory usage: 50 bytes
  Sparsity: 25.0% (stored/total)
  Validation: enabled
Extracted diagonal values:
  Diagonal[0] = 1.0
  Diagonal[1] = 4.0
  Diagonal[2] = 9.0
  Diagonal[3] = 16.0

3. Conversion Accuracy Verification:
Conversion accuracy: Perfect (0 errors)
Conversion testing completed successfully

=== Testing Diagonal Error Handling ===

1. Invalid Size Errors:
Correctly caught zero size error
Correctly caught negative size error
Correctly caught oversized matrix error

2. Invalid Offset Errors:
Correctly caught invalid offset error
Correctly caught zero offset error for super-diagonal
Correctly caught positive offset error for sub-diagonal

3. Bounds Checking:
Correctly caught out-of-bounds diagonal access
Correctly caught off-diagonal non-zero assignment

4. Edge Case Testing:
1x1 diagonal tensor works correctly
Out-of-bounds access returns zero

Error Handling Summary: 10/10 tests passed
All error handling tests passed

=== Testing Diagonal Memory Management ===

1. Memory Usage Analysis:
  10x10 matrix:
    Diagonal storage: 74 bytes
    Sparsity: 10.0% stored/total
    Memory efficiency: O(n) vs O(n²) for dense
  100x100 matrix:
    Diagonal storage: 434 bytes
    Sparsity: 1.0% stored/total
    Memory efficiency: O(n) vs O(n²) for dense
  1000x1000 matrix:
    Diagonal storage: 4034 bytes
    Sparsity: 0.1% stored/total
    Memory efficiency: O(n) vs O(n²) for dense

2. Copy Constructor Testing:
Original:
DiagonalTensor[float32]
  Size: 5x5
  Diagonal offset: 0
  Diagonal length: 5
  Memory usage: 54 bytes
  Sparsity: 20.0% (stored/total)
  Validation: enabled
Copy:
DiagonalTensor[float32]
  Size: 5x5
  Diagonal offset: 0
  Diagonal length: 5
  Memory usage: 54 bytes
  Sparsity: 20.0% (stored/total)
  Validation: enabled
Copy constructor creates independent copy

3. Multi-diagonal Memory Management:
Multi-diagonal memory usage: 1319 bytes
Stores 3 diagonals efficiently

4. Large Diagonal Creation/Destruction:
Starting: Large Diagonal Creation/Destruction
Completed: Large Diagonal Creation/Destruction (100 operations)
Large diagonal creation/destruction completed
Memory management testing completed successfully

=====================================
=== DIAGONAL TENSOR TESTS COMPLETE ===
=====================================

=== Diagonal Tensor Implementation Summary ===
+ Memory-efficient diagonal tensor representation
+ Support for multiple diagonal bands (main, super, sub)
+ Sparse storage proportional to diagonal count, not matrix size
+ Fast element access with automatic zero handling
+ Multi-diagonal tensor support (tridiagonal, pentadiagonal)
+ Optimized mathematical operations (scaling, transpose)
+ Comprehensive error handling and validation
+ Performance benchmarking and memory analysis
+ Seamless conversion to/from dense format
+ Diagonal extraction from dense matrices
+ Integration-ready for tensor library ecosystem

Performance Characteristics:
- Creation time: O(n) - linear in diagonal length
- Memory usage: O(n) - proportional to diagonal elements
- Element access: O(1) - direct diagonal computation
- Conversion to dense: O(n²) - only when explicitly needed
- Multi-diagonal: O(k*n) where k is number of diagonals

Mathematical Features:
- Main diagonal matrices
- Super-diagonal matrices (offset > 0)
- Sub-diagonal matrices (offset < 0)
- Multi-band diagonal matrices
- Tridiagonal matrix creation utilities
- Diagonal scaling and transpose operations
- Dense matrix diagonal extraction

Diagonal Tensor Implementation (Part 1.3.4B) Complete
Ready for integration with remaining specialized tensor types
```

### File: `45_specialized_tensors_part_c.mojo` — Section 9.3

**Run:** `pixi run mojo 45_specialized_tensors_part_c.mojo`

```mojo

from memory import UnsafePointer
from collections import List
from sys import sizeof


# Core Tensor Infrastructure - Part 1.3.4C: Sparse Tensor Implementation
#
# This section implements memory-efficient sparse tensor structures optimized
# for mathematical operations involving large matrices with predominantly zero
# elements. Provides specialized storage formats that exploit sparsity patterns
# for optimal performance and memory usage.
#
# Key Design Principles:
# - Memory usage proportional to non-zero elements only
# - Support for COO (Coordinate) format for flexible construction
# - Fast sparse matrix operations and conversions
# - Automatic sparsity detection and optimization
# - Integration with dense tensor operations
#
# Implementation Features:
# 1. COO (Coordinate) format for flexible sparse representation
# 2. Automatic zero element detection and removal
# 3. Sparse matrix arithmetic and element manipulation
# 4. Memory-efficient storage with dynamic capacity management
# 5. Seamless conversion to dense format when needed

alias DEFAULT_SPARSE_CAPACITY = 1000
alias MAX_SPARSE_ELEMENTS = 100000
alias DEFAULT_SPARSITY_THRESHOLD = 1e-10
alias MAX_SPARSE_SIZE = 5000


struct SparseElement[dtype: DType]:
    """Single sparse tensor element with coordinates and value."""
    var row: Int
    var col: Int
    var value: Scalar[dtype]
    
    fn __init__(out self, row: Int, col: Int, value: Scalar[dtype]):
        self.row = row
        self.col = col
        self.value = value
    
    fn __copyinit__(out self, existing: Self):
        self.row = existing.row
        self.col = existing.col
        self.value = existing.value
    
    fn is_zero(self, threshold: Float64 = DEFAULT_SPARSITY_THRESHOLD) -> Bool:
        """Check if element value is effectively zero."""
        return abs(Float64(self.value)) <= threshold
    
    fn print_info(self):
        """Print sparse element information."""
        var row_str: String = String(self.row)
        var col_str: String = String(self.col)
        var value_str: String = String(self.value)
        print("  [" + row_str + ", " + col_str + "] = " + value_str)


struct SparseTensorCOO[dtype: DType]:
    """Sparse tensor in Coordinate (COO) format."""
    var elements: UnsafePointer[SparseElement[dtype]]
    var num_elements: Int
    var capacity: Int
    var shape: List[Int]
    var _owns_data: Bool
    var _validation_enabled: Bool
    
    fn __init__(out self, shape: List[Int], capacity: Int = DEFAULT_SPARSE_CAPACITY) raises:
        if len(shape) != 2:
            raise Error("Sparse tensor currently supports only 2D matrices")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
            if shape[i] > MAX_SPARSE_SIZE:
                raise Error("Dimension size exceeds maximum allowed")
        
        if capacity <= 0 or capacity > MAX_SPARSE_ELEMENTS:
            raise Error("Invalid capacity")
        
        self.shape = List[Int]()
        for i in range(len(shape)):
            self.shape.append(shape[i])
        
        self.capacity = capacity
        self.num_elements = 0
        self._owns_data = True
        self._validation_enabled = True
        
        self.elements = UnsafePointer[SparseElement[dtype]].alloc(capacity)
    
    fn __copyinit__(out self, existing: Self):
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        
        self.capacity = existing.capacity
        self.num_elements = existing.num_elements
        self._owns_data = True
        self._validation_enabled = existing._validation_enabled
        
        self.elements = UnsafePointer[SparseElement[dtype]].alloc(self.capacity)
        for i in range(self.num_elements):
            self.elements[i] = existing.elements[i]
    
    fn __del__(owned self):
        if self._owns_data:
            self.elements.free()
    
    fn _find_element_index(self, row: Int, col: Int) -> Int:
        """Find index of element at (row, col), returns -1 if not found."""
        for i in range(self.num_elements):
            if self.elements[i].row == row and self.elements[i].col == col:
                return i
        return -1
    
    fn _is_valid_position(self, row: Int, col: Int) -> Bool:
        """Check if position is within matrix bounds."""
        return row >= 0 and row < self.shape[0] and col >= 0 and col < self.shape[1]
    
    fn add_element(mut self, row: Int, col: Int, value: Scalar[dtype]) raises:
        """Add or update sparse element at specified position."""
        if self._validation_enabled:
            if not self._is_valid_position(row, col):
                raise Error("Element coordinates out of bounds")
        
        # Check for zero value
        var element = SparseElement[dtype](row, col, value)
        if element.is_zero():
            self.remove_element(row, col)
            return
        
        # Check if element already exists
        var existing_index = self._find_element_index(row, col)
        if existing_index >= 0:
            self.elements[existing_index].value = value
            return
        
        # Add new element
        if self.num_elements >= self.capacity:
            raise Error("Sparse tensor capacity exceeded")
        
        self.elements[self.num_elements] = element
        self.num_elements += 1
    
    fn remove_element(mut self, row: Int, col: Int):
        """Remove element at specified position."""
        var index = self._find_element_index(row, col)
        if index >= 0:
            for i in range(index, self.num_elements - 1):
                self.elements[i] = self.elements[i + 1]
            self.num_elements -= 1
    
    fn get_item(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element value at specified position."""
        if self._validation_enabled:
            if not self._is_valid_position(row, col):
                return Scalar[dtype](0)
        
        var index = self._find_element_index(row, col)
        if index >= 0:
            return self.elements[index].value
        else:
            return Scalar[dtype](0)
    
    fn set_item(mut self, row: Int, col: Int, value: Scalar[dtype]) raises:
        """Set element value at specified position."""
        self.add_element(row, col, value)
    
    fn get_sparsity(self) -> Float64:
        """Calculate sparsity ratio (non-zero elements / total elements)."""
        var total_elements = self.shape[0] * self.shape[1]
        if total_elements == 0:
            return 0.0
        return Float64(self.num_elements) / Float64(total_elements)
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        var base_size = 4 * sizeof[Int]() + 2 * sizeof[Bool]()
        var elements_size = self.capacity * sizeof[SparseElement[dtype]]()
        var shape_size = len(self.shape) * sizeof[Int]()
        return base_size + elements_size + shape_size
    
    fn compress_storage(mut self):
        """Remove zero elements and optimize storage."""
        var write_index = 0
        
        for read_index in range(self.num_elements):
            var element = self.elements[read_index]
            if not element.is_zero():
                if write_index != read_index:
                    self.elements[write_index] = element
                write_index += 1
        
        self.num_elements = write_index
    
    fn set_validation(mut self, enabled: Bool):
        """Enable or disable bounds checking."""
        self._validation_enabled = enabled
    
    fn to_dense(self) raises -> DenseTensor[dtype]:
        """Convert to dense tensor representation."""
        var dense = DenseTensor[dtype](self.shape)
        
        for i in range(self.num_elements):
            var element = self.elements[i]
            var indices = List[Int]()
            indices.append(element.row)
            indices.append(element.col)
            dense.set_item(indices, element.value)
        
        return dense
    
    fn print_info(self):
        """Print sparse tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var rows_str: String = String(self.shape[0])
        var cols_str: String = String(self.shape[1])
        var elements_str: String = String(self.num_elements)
        var capacity_str: String = String(self.capacity)
        var sparsity = self.get_sparsity()
        var sparsity_str: String = String(sparsity * 100.0)
        var memory_str: String = String(self.get_memory_usage())
        
        print("SparseTensorCOO[" + dtype_str + "]")
        print("  Shape: " + rows_str + "x" + cols_str)
        print("  Non-zero elements: " + elements_str + "/" + capacity_str)
        print("  Sparsity: " + sparsity_str + "% (non-zero/total)")
        print("  Memory usage: " + memory_str + " bytes")
        print("  Validation: " + ("enabled" if self._validation_enabled else "disabled"))


struct DenseTensor[dtype: DType]:
    """Dense tensor implementation for conversion and compatibility."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1
        self._owns_data = True
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
            self.shape.append(shape[i])
            self._total_elements *= shape[i]
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        self.ndim = existing.ndim
        self._total_elements = existing._total_elements
        self._owns_data = True
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
    
    fn _get_linear_index(self, indices: List[Int]) -> Int:
        """Convert multi-dimensional indices to linear index."""
        var linear_index = 0
        var stride = 1
        
        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]
        
        return linear_index
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        return self.data[linear_index]
    
    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        self.data[linear_index] = value
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        return self._total_elements
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        return self._total_elements * sizeof[Scalar[dtype]]()
    
    fn print_info(self):
        """Print tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var memory_str: String = String(self.get_memory_usage())
        
        print("DenseTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)
        print("  Memory usage: " + memory_str + " bytes")


fn create_sparse_coo[dtype: DType](shape: List[Int], capacity: Int = DEFAULT_SPARSE_CAPACITY) raises -> SparseTensorCOO[dtype]:
    """Create empty COO sparse tensor."""
    return SparseTensorCOO[dtype](shape, capacity)

fn create_sparse_from_dense[dtype: DType](dense: DenseTensor[dtype], threshold: Float64 = DEFAULT_SPARSITY_THRESHOLD) raises -> SparseTensorCOO[dtype]:
    """Convert dense tensor to COO sparse format."""
    if dense.ndim != 2:
        raise Error("Can only convert 2D tensors to sparse format")
    
    var sparse = create_sparse_coo[dtype](dense.shape)
    
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            var indices = List[Int]()
            indices.append(i)
            indices.append(j)
            var value = dense.get_item(indices)
            
            if abs(Float64(value)) > threshold:
                sparse.add_element(i, j, value)
    
    return sparse

fn create_sparse_identity[dtype: DType](size: Int, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> SparseTensorCOO[dtype]:
    """Create sparse identity matrix in COO format."""
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)
    
    var sparse = create_sparse_coo[dtype](shape, size)
    
    for i in range(size):
        sparse.add_element(i, i, scale)
    
    return sparse

fn create_sparse_diagonal[dtype: DType](size: Int, values: List[Scalar[dtype]], offset: Int = 0) raises -> SparseTensorCOO[dtype]:
    """Create sparse diagonal matrix in COO format."""
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)
    
    var sparse = create_sparse_coo[dtype](shape, len(values))
    
    for i in range(len(values) if len(values) < size else size):
        var row = i
        var col = i + offset
        
        if col >= 0 and col < size:
            sparse.add_element(row, col, values[i])
    
    return sparse


struct PerformanceTimer:
    """Simple performance timing utility."""
    var name: String
    
    fn __init__(out self, name: String):
        self.name = name
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
    
    fn start(self):
        """Start timing operation."""
        print("Starting: " + self.name)
    
    fn end(self, operations: Int = 1):
        """End timing and report results."""
        var ops_str: String = String(operations)
        print("Completed: " + self.name + " (" + ops_str + " operations)")

fn benchmark_sparse_creation[dtype: DType](size: Int, sparsity: Float64, iterations: Int) -> Float64:
    """Benchmark sparse tensor creation performance."""
    var timer = PerformanceTimer("Sparse Creation Benchmark")
    timer.start()
    
    try:
        for _ in range(iterations):
            var shape = List[Int]()
            shape.append(size)
            shape.append(size)
            
            var sparse = create_sparse_coo[dtype](shape)
            
            var num_elements = Int(Float64(size * size) * sparsity)
            for i in range(num_elements if num_elements < size else size):
                sparse.add_element(i, i, Scalar[dtype](1))
            
            var _ = sparse.get_item(0, 0)
    except:
        print("Benchmark failed")
        return -1.0
    
    timer.end(iterations)
    return Float64(iterations) / 1000.0

fn benchmark_sparse_memory_efficiency[dtype: DType](size: Int, sparsity: Float64) -> Float64:
    """Compare memory usage: sparse vs dense."""
    try:
        var shape = List[Int]()
        shape.append(size)
        shape.append(size)
        
        var sparse = create_sparse_coo[dtype](shape)
        var num_elements = Int(Float64(size * size) * sparsity)
        
        for i in range(num_elements if num_elements < size else size):
            sparse.add_element(i, i, Scalar[dtype](1))
        
        var dense = sparse.to_dense()
        
        var sparse_memory = sparse.get_memory_usage()
        var dense_memory = dense.get_memory_usage()
        
        var efficiency = Float64(sparse_memory) / Float64(dense_memory)
        
        var sparse_str: String = String(sparse_memory)
        var dense_str: String = String(dense_memory)
        var efficiency_str: String = String(efficiency * 100.0)
        var sparsity_str: String = String(sparsity * 100.0)
        
        print("Memory Efficiency Analysis (Sparsity: " + sparsity_str + "%):")
        print("  Sparse tensor: " + sparse_str + " bytes")
        print("  Dense equivalent: " + dense_str + " bytes")
        print("  Efficiency: " + efficiency_str + "% of dense storage")
        
        return efficiency
        
    except:
        print("Memory benchmark failed")
        return -1.0


fn test_basic_sparse_operations():
    """Test basic sparse tensor functionality."""
    print("=== Testing Basic Sparse Operations ===")
    
    try:
        print("\n1. COO Sparse Tensor Creation:")
        var shape = List[Int]()
        shape.append(4)
        shape.append(4)
        
        var sparse = create_sparse_coo[DType.float32](shape, 10)
        sparse.print_info()
        
        print("\n2. Adding Sparse Elements:")
        sparse.add_element(0, 0, 1.0)
        sparse.add_element(1, 1, 2.0)
        sparse.add_element(2, 3, 3.5)
        sparse.add_element(3, 2, -1.5)
        
        sparse.print_info()
        
        print("Sparse tensor elements:")
        for i in range(sparse.num_elements):
            var element = sparse.elements[i]
            element.print_info()
        
        print("\n3. Sparse Matrix Display (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = sparse.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("Basic sparse operations completed successfully")
        
    except:
        print("Error in basic sparse operations")

fn test_sparse_mathematical_operations():
    """Test mathematical operations on sparse tensors."""
    print("\n=== Testing Sparse Mathematical Operations ===")
    
    try:
        print("\n1. Sparse Element Manipulation:")
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        
        var sparse = create_sparse_coo[DType.float32](shape)
        
        sparse.add_element(0, 0, 5.0)
        sparse.add_element(1, 1, 10.0)
        sparse.add_element(2, 2, 15.0)
        sparse.add_element(0, 2, 3.0)
        
        print("Before compression:")
        sparse.print_info()
        
        sparse.add_element(1, 0, 0.0)  # Add zero element
        
        print("After adding zero element:")
        sparse.print_info()
        
        print("\n2. Sparse Storage Compression:")
        sparse.add_element(2, 0, 1e-12)  # Nearly zero
        sparse.add_element(2, 1, 1e-8)   # Small but kept
        
        print("Before compression:")
        sparse.print_info()
        
        sparse.compress_storage()
        
        print("After compression:")
        sparse.print_info()
        
        print("Mathematical operations completed successfully")
        
    except:
        print("Error in mathematical operations")

fn test_sparse_conversions():
    """Test conversions between sparse and dense formats."""
    print("\n=== Testing Sparse Conversions ===")
    
    try:
        print("\n1. Sparse to Dense Conversion:")
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        
        var sparse = create_sparse_coo[DType.float32](shape)
        sparse.add_element(0, 0, 1.0)
        sparse.add_element(1, 1, 4.0)
        sparse.add_element(2, 2, 9.0)
        sparse.add_element(0, 2, 2.0)
        
        print("Original sparse tensor:")
        sparse.print_info()
        
        var dense = sparse.to_dense()
        print("Converted dense tensor:")
        dense.print_info()
        
        print("Dense matrix values:")
        for i in range(3):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = dense.get_item(indices)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Dense to Sparse Conversion:")
        var sparse_from_dense = create_sparse_from_dense[DType.float32](dense)
        print("Converted back to sparse:")
        sparse_from_dense.print_info()
        
        print("Round-trip sparse elements:")
        for i in range(sparse_from_dense.num_elements):
            var element = sparse_from_dense.elements[i]
            element.print_info()
        
        print("Conversion testing completed successfully")
        
    except:
        print("Error in conversion testing")

fn test_sparse_performance():
    """Test sparse tensor performance characteristics."""
    print("\n=== Testing Sparse Performance ===")
    
    print("\n1. Creation Performance with Different Sparsity:")
    var sparsity_levels = List[Float64]()
    sparsity_levels.append(0.01)
    sparsity_levels.append(0.05)
    sparsity_levels.append(0.10)
    
    for i in range(len(sparsity_levels)):
        var sparsity = sparsity_levels[i]
        var iterations = 100
        var sparsity_str: String = String(sparsity * 100.0)
        print("Testing " + sparsity_str + "% sparsity creation...")
        
        var result = benchmark_sparse_creation[DType.float32](50, sparsity, iterations)
        var result_str: String = String(result)
        print("  Performance metric: " + result_str + " k-ops")
    
    print("\n2. Memory Efficiency Analysis:")
    var sizes = List[Int]()
    sizes.append(50)
    sizes.append(100)
    sizes.append(200)
    
    for i in range(len(sizes)):
        var size = sizes[i]
        var size_str: String = String(size)
        print("Memory analysis for " + size_str + "x" + size_str + " matrix:")
        var _ = benchmark_sparse_memory_efficiency[DType.float32](size, 0.05)
    
    print("Performance testing completed successfully")

fn test_sparse_specialized_matrices():
    """Test creation of specialized sparse matrices."""
    print("\n=== Testing Specialized Sparse Matrices ===")
    
    try:
        print("\n1. Sparse Identity Matrix:")
        var identity = create_sparse_identity[DType.float32](4, 2.0)
        identity.print_info()
        
        print("Identity matrix values:")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = identity.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Sparse Diagonal Matrix:")
        var diag_values = List[Scalar[DType.float32]]()
        diag_values.append(1.0)
        diag_values.append(4.0)
        diag_values.append(9.0)
        diag_values.append(16.0)
        
        var diagonal = create_sparse_diagonal[DType.float32](4, diag_values, 0)
        diagonal.print_info()
        
        print("Diagonal matrix values:")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = diagonal.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("Specialized matrix testing completed successfully")
        
    except:
        print("Error in specialized matrix testing")

fn test_sparse_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Sparse Error Handling ===")
    
    var test_count = 0
    var passed_count = 0
    
    print("\n1. Invalid Shape Errors:")
    test_count += 1
    try:
        var shape = List[Int]()
        shape.append(0)
        shape.append(5)
        var _ = create_sparse_coo[DType.float32](shape)
        print("Should have failed with zero dimension")
    except:
        print("Correctly caught zero dimension error")
        passed_count += 1
    
    test_count += 1
    try:
        var shape = List[Int]()
        shape.append(5)
        shape.append(-3)
        var _ = create_sparse_coo[DType.float32](shape)
        print("Should have failed with negative dimension")
    except:
        print("Correctly caught negative dimension error")
        passed_count += 1
    
    print("\n2. Capacity Overflow:")
    test_count += 1
    try:
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        var sparse = create_sparse_coo[DType.float32](shape, 2)
        
        sparse.add_element(0, 0, 1.0)
        sparse.add_element(1, 1, 2.0)
        sparse.add_element(2, 2, 3.0)  # Should exceed capacity
        print("Should have failed with capacity overflow")
    except:
        print("Correctly caught capacity overflow error")
        passed_count += 1
    
    print("\n3. Bounds Checking:")
    test_count += 1
    try:
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        var sparse = create_sparse_coo[DType.float32](shape)
        sparse.add_element(5, 1, 1.0)  # Out of bounds
        print("Should have failed with out-of-bounds access")
    except:
        print("Correctly caught out-of-bounds access")
        passed_count += 1
    
    var passed_str: String = String(passed_count)
    var total_str: String = String(test_count)
    print("\nError Handling Summary: " + passed_str + "/" + total_str + " tests passed")
    
    if passed_count == test_count:
        print("All error handling tests passed")
    else:
        print("Some error handling tests failed")

fn test_sparse_memory_management():
    """Test memory management and resource cleanup."""
    print("\n=== Testing Sparse Memory Management ===")
    
    try:
        print("\n1. Memory Usage Analysis:")
        var sparsity_levels = List[Float64]()
        sparsity_levels.append(0.01)
        sparsity_levels.append(0.05)
        sparsity_levels.append(0.10)
        
        for i in range(len(sparsity_levels)):
            var sparsity = sparsity_levels[i]
            var shape = List[Int]()
            shape.append(100)
            shape.append(100)
            
            var sparse = create_sparse_coo[DType.float32](shape, 1000)
            var num_elements = Int(Float64(100 * 100) * sparsity)
            
            for j in range(num_elements if num_elements < 100 else 100):
                sparse.add_element(j, j, Scalar[DType.float32](j + 1))
            
            var memory = sparse.get_memory_usage()
            var actual_sparsity = sparse.get_sparsity()
            
            var sparsity_str: String = String(sparsity * 100.0)
            var memory_str: String = String(memory)
            var actual_str: String = String(actual_sparsity * 100.0)
            
            print("  Sparsity " + sparsity_str + "%:")
            print("    Memory usage: " + memory_str + " bytes")
            print("    Actual sparsity: " + actual_str + "%")
        
        print("\n2. Copy Constructor Testing:")
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        
        var original = create_sparse_coo[DType.float32](shape)
        original.add_element(0, 0, 1.0)
        original.add_element(1, 1, 4.0)
        original.add_element(2, 2, 9.0)
        
        var copied = original
        
        print("Original:")
        original.print_info()
        print("Copy:")
        copied.print_info()
        
        original.add_element(0, 1, 2.0)
        
        var orig_count = original.num_elements
        var copy_count = copied.num_elements
        
        if orig_count != copy_count:
            print("Copy constructor creates independent copy")
        else:
            print("Copy constructor may share data")
        
        print("\n3. Large Sparse Creation/Destruction:")
        var creation_timer = PerformanceTimer("Large Sparse Creation/Destruction")
        creation_timer.start()
        
        for _ in range(50):
            var large_shape = List[Int]()
            large_shape.append(500)
            large_shape.append(500)
            var large_sparse = create_sparse_coo[DType.float32](large_shape, 100)
            
            for j in range(50):
                large_sparse.add_element(j, j, Scalar[DType.float32](1))
            
            var _ = large_sparse.get_item(0, 0)
        
        creation_timer.end(50)
        print("Large sparse creation/destruction completed")
        
        print("Memory management testing completed successfully")
        
    except:
        print("Error in memory management testing")

fn run_all_sparse_tests():
    """Run comprehensive test suite for sparse tensors."""
    print("=====================================")
    print("=== SPARSE TENSOR TEST SUITE ===")
    print("=====================================")
    
    test_basic_sparse_operations()
    test_sparse_mathematical_operations()
    test_sparse_conversions()
    test_sparse_performance()
    test_sparse_specialized_matrices()
    test_sparse_error_handling()
    test_sparse_memory_management()
    
    print("\n=====================================")
    print("=== SPARSE TENSOR TESTS COMPLETE ===")
    print("=====================================")


fn main():
    """Main function to run sparse tensor implementation tests."""
    print("=== Specialized Tensor Types - Part 1.3.4C ===")
    print("Sparse Tensor Implementation - Complete Standalone Module")
    print("")
    
    run_all_sparse_tests()
    
    print("\n=== Sparse Tensor Implementation Summary ===")
    print("+ Memory-efficient sparse tensor representation")
    print("+ COO (Coordinate) format for flexible sparse construction")
    print("+ Sparse storage proportional to non-zero elements only")
    print("+ Fast element access with automatic zero handling")
    print("+ Specialized sparse matrix creation (identity, diagonal)")
    print("+ Comprehensive sparsity analysis and optimization")
    print("+ Performance benchmarking and memory analysis")
    print("+ Seamless conversion to/from dense format")
    print("+ Integration-ready for tensor library ecosystem")
    print("")
    print("Performance Characteristics:")
    print("- Memory usage: O(nnz) - proportional to non-zero elements")
    print("- Element access: O(nnz) - linear search in elements")
    print("- Conversion to dense: O(mn) - where m,n are matrix dimensions")
    print("- Storage compression: O(nnz) - removes zero elements")
    print("")
    print("Sparse Features Supported:")
    print("- COO (Coordinate) format for flexible construction")
    print("- Automatic zero element detection and removal")
    print("- Sparse matrix element manipulation")
    print("- Dense-to-sparse conversion with threshold")
    print("- Specialized sparse matrix creation utilities")
    print("")
    print("Mathematical Features:")
    print("- Sparse matrix element manipulation")
    print("- Automatic zero element compression")
    print("- Sparse matrix storage optimization")
    print("- Dense-to-sparse conversion with threshold")
    print("- Specialized sparse matrix creation utilities")
    print("")
    print("Sparse Tensor Implementation (Part 1.3.4C) Complete")
    print("Ready for integration with remaining specialized tensor types")
```

### Expected Output for `45_specialized_tensors_part_c.mojo`

```
=== Specialized Tensor Types - Part 1.3.4C ===
Sparse Tensor Implementation - Complete Standalone Module

=====================================
=== SPARSE TENSOR TEST SUITE ===
=====================================
=== Testing Basic Sparse Operations ===

1. COO Sparse Tensor Creation:
SparseTensorCOO[float32]
  Shape: 4x4
  Non-zero elements: 0/10
  Sparsity: 0.0% (non-zero/total)
  Memory usage: 290 bytes
  Validation: enabled

2. Adding Sparse Elements:
SparseTensorCOO[float32]
  Shape: 4x4
  Non-zero elements: 4/10
  Sparsity: 25.0% (non-zero/total)
  Memory usage: 290 bytes
  Validation: enabled
Sparse tensor elements:
  [0, 0] = 1.0
  [1, 1] = 2.0
  [2, 3] = 3.5
  [3, 2] = -1.5

3. Sparse Matrix Display (4x4):
  Row 0: 1.0 0.0 0.0 0.0 
  Row 1: 0.0 2.0 0.0 0.0 
  Row 2: 0.0 0.0 0.0 3.5 
  Row 3: 0.0 0.0 -1.5 0.0 
Basic sparse operations completed successfully

=== Testing Sparse Mathematical Operations ===

1. Sparse Element Manipulation:
Before compression:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 4/1000
  Sparsity: 44.44444444444444% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
After adding zero element:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 4/1000
  Sparsity: 44.44444444444444% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled

2. Sparse Storage Compression:
Before compression:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 5/1000
  Sparsity: 55.55555555555556% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
After compression:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 5/1000
  Sparsity: 55.55555555555556% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
Mathematical operations completed successfully

=== Testing Sparse Conversions ===

1. Sparse to Dense Conversion:
Original sparse tensor:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 4/1000
  Sparsity: 44.44444444444444% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
Converted dense tensor:
DenseTensor[float32]
  Shape: [3, 3]
  Elements: 9
  Memory usage: 36 bytes
Dense matrix values:
  Row 0: 1.0 0.0 2.0 
  Row 1: 0.0 4.0 0.0 
  Row 2: 0.0 0.0 9.0 

2. Dense to Sparse Conversion:
Converted back to sparse:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 4/1000
  Sparsity: 44.44444444444444% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
Round-trip sparse elements:
  [0, 0] = 1.0
  [0, 2] = 2.0
  [1, 1] = 4.0
  [2, 2] = 9.0
Conversion testing completed successfully

=== Testing Sparse Performance ===

1. Creation Performance with Different Sparsity:
Testing 1.0% sparsity creation...
Starting: Sparse Creation Benchmark
Completed: Sparse Creation Benchmark (100 operations)
  Performance metric: 0.1 k-ops
Testing 5.0% sparsity creation...
Starting: Sparse Creation Benchmark
Completed: Sparse Creation Benchmark (100 operations)
  Performance metric: 0.1 k-ops
Testing 10.0% sparsity creation...
Starting: Sparse Creation Benchmark
Completed: Sparse Creation Benchmark (100 operations)
  Performance metric: 0.1 k-ops

2. Memory Efficiency Analysis:
Memory analysis for 50x50 matrix:
Memory Efficiency Analysis (Sparsity: 5.0%):
  Sparse tensor: 24050 bytes
  Dense equivalent: 10000 bytes
  Efficiency: 240.49999999999997% of dense storage
Memory analysis for 100x100 matrix:
Memory Efficiency Analysis (Sparsity: 5.0%):
  Sparse tensor: 24050 bytes
  Dense equivalent: 40000 bytes
  Efficiency: 60.12499999999999% of dense storage
Memory analysis for 200x200 matrix:
Memory Efficiency Analysis (Sparsity: 5.0%):
  Sparse tensor: 24050 bytes
  Dense equivalent: 160000 bytes
  Efficiency: 15.031249999999998% of dense storage
Performance testing completed successfully

=== Testing Specialized Sparse Matrices ===

1. Sparse Identity Matrix:
SparseTensorCOO[float32]
  Shape: 4x4
  Non-zero elements: 4/4
  Sparsity: 25.0% (non-zero/total)
  Memory usage: 146 bytes
  Validation: enabled
Identity matrix values:
  Row 0: 2.0 0.0 0.0 0.0 
  Row 1: 0.0 2.0 0.0 0.0 
  Row 2: 0.0 0.0 2.0 0.0 
  Row 3: 0.0 0.0 0.0 2.0 

2. Sparse Diagonal Matrix:
SparseTensorCOO[float32]
  Shape: 4x4
  Non-zero elements: 4/4
  Sparsity: 25.0% (non-zero/total)
  Memory usage: 146 bytes
  Validation: enabled
Diagonal matrix values:
  Row 0: 1.0 0.0 0.0 0.0 
  Row 1: 0.0 4.0 0.0 0.0 
  Row 2: 0.0 0.0 9.0 0.0 
  Row 3: 0.0 0.0 0.0 16.0 
Specialized matrix testing completed successfully

=== Testing Sparse Error Handling ===

1. Invalid Shape Errors:
Correctly caught zero dimension error
Correctly caught negative dimension error

2. Capacity Overflow:
Correctly caught capacity overflow error

3. Bounds Checking:
Correctly caught out-of-bounds access

Error Handling Summary: 4/4 tests passed
All error handling tests passed

=== Testing Sparse Memory Management ===

1. Memory Usage Analysis:
  Sparsity 1.0%:
    Memory usage: 24050 bytes
    Actual sparsity: 1.0%
  Sparsity 5.0%:
    Memory usage: 24050 bytes
    Actual sparsity: 1.0%
  Sparsity 10.0%:
    Memory usage: 24050 bytes
    Actual sparsity: 1.0%

2. Copy Constructor Testing:
Original:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 3/1000
  Sparsity: 33.33333333333333% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
Copy:
SparseTensorCOO[float32]
  Shape: 3x3
  Non-zero elements: 3/1000
  Sparsity: 33.33333333333333% (non-zero/total)
  Memory usage: 24050 bytes
  Validation: enabled
Copy constructor creates independent copy

3. Large Sparse Creation/Destruction:
Starting: Large Sparse Creation/Destruction
Completed: Large Sparse Creation/Destruction (50 operations)
Large sparse creation/destruction completed
Memory management testing completed successfully

=====================================
=== SPARSE TENSOR TESTS COMPLETE ===
=====================================

=== Sparse Tensor Implementation Summary ===
+ Memory-efficient sparse tensor representation
+ COO (Coordinate) format for flexible sparse construction
+ Sparse storage proportional to non-zero elements only
+ Fast element access with automatic zero handling
+ Specialized sparse matrix creation (identity, diagonal)
+ Comprehensive sparsity analysis and optimization
+ Performance benchmarking and memory analysis
+ Seamless conversion to/from dense format
+ Integration-ready for tensor library ecosystem

Performance Characteristics:
- Memory usage: O(nnz) - proportional to non-zero elements
- Element access: O(nnz) - linear search in elements
- Conversion to dense: O(mn) - where m,n are matrix dimensions
- Storage compression: O(nnz) - removes zero elements

Sparse Features Supported:
- COO (Coordinate) format for flexible construction
- Automatic zero element detection and removal
- Sparse matrix element manipulation
- Dense-to-sparse conversion with threshold
- Specialized sparse matrix creation utilities

Mathematical Features:
- Sparse matrix element manipulation
- Automatic zero element compression
- Sparse matrix storage optimization
- Dense-to-sparse conversion with threshold
- Specialized sparse matrix creation utilities

Sparse Tensor Implementation (Part 1.3.4C) Complete
Ready for integration with remaining specialized tensor types
```

### File: `45_specialized_tensors_part_d.mojo` — Section 9.4

**Run:** `pixi run mojo 45_specialized_tensors_part_d.mojo`

```mojo

from memory import UnsafePointer
from collections import List
from sys import sizeof


# Core Tensor Infrastructure - Part 1.3.4D: Triangular Tensor Implementation
#
# This section implements memory-efficient triangular tensor structures optimized
# for mathematical operations involving upper and lower triangular matrices.
# Provides specialized storage patterns that exploit the triangular structure
# for optimal performance and memory usage.
#
# Key Design Principles:
# - Memory-efficient storage for triangular matrices (n(n+1)/2 elements)
# - Support for both upper and lower triangular formats
# - Fast element access with automatic zero handling for non-stored regions
# - Integration with linear algebra operations
# - Seamless conversion to dense format when needed
#
# Implementation Features:
# 1. Packed triangular storage for memory efficiency
# 2. Upper and lower triangular matrix support
# 3. Fast element access with triangular indexing
# 4. Optimized triangular matrix operations
# 5. Comprehensive validation and bounds checking
#
# Mathematical Properties Exploited:
# - Triangular matrices have zeros in half the matrix
# - Only n(n+1)/2 elements need storage instead of n²
# - Matrix operations can leverage triangular structure
# - Forward/backward substitution optimizations

alias MAX_TRIANGULAR_SIZE = 5000
alias DEFAULT_TRIANGULAR_SIZE = 100


struct TriangularTensor[dtype: DType]:
    """Memory-efficient triangular matrix implementation with packed storage."""
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var is_upper: Bool
    var _storage_size: Int
    var _owns_data: Bool
    var _validation_enabled: Bool
    
    fn __init__(out self, size: Int, is_upper: Bool = True) raises:
        """Initialize triangular tensor with specified size and orientation."""
        if size <= 0:
            raise Error("Triangular matrix size must be positive")
        if size > MAX_TRIANGULAR_SIZE:
            raise Error("Triangular matrix size exceeds maximum allowed")
        
        self.size = size
        self.is_upper = is_upper
        self._owns_data = True
        self._validation_enabled = True
        
        # Triangular storage: n(n+1)/2 elements
        self._storage_size = (size * (size + 1)) // 2
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._storage_size)
        
        # Initialize to zero
        for i in range(self._storage_size):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for triangular tensor."""
        self.size = existing.size
        self.is_upper = existing.is_upper
        self._storage_size = existing._storage_size
        self._owns_data = True
        self._validation_enabled = existing._validation_enabled
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._storage_size)
        for i in range(self._storage_size):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        """Destructor to free allocated memory."""
        if self._owns_data:
            self.data.free()
    
    fn _get_storage_index(self, row: Int, col: Int) -> Int:
        """Get storage index for triangular element, returns -1 if not stored."""
        if self.is_upper:
            # Upper triangular: store elements where col >= row
            if col >= row:
                return (row * (2 * self.size - row - 1)) // 2 + (col - row)
            else:
                return -1  # Element is zero (not stored)
        else:
            # Lower triangular: store elements where row >= col
            if row >= col:
                return (row * (row + 1)) // 2 + col
            else:
                return -1  # Element is zero (not stored)
    
    fn _is_in_triangle(self, row: Int, col: Int) -> Bool:
        """Check if position is within the stored triangular region."""
        if self.is_upper:
            return col >= row
        else:
            return row >= col
    
    fn _is_valid_position(self, row: Int, col: Int) -> Bool:
        """Check if position is within matrix bounds."""
        return row >= 0 and row < self.size and col >= 0 and col < self.size
    
    fn get_item(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element at specified position."""
        if self._validation_enabled:
            if not self._is_valid_position(row, col):
                return Scalar[dtype](0)
        
        var storage_index = self._get_storage_index(row, col)
        if storage_index >= 0:
            return self.data[storage_index]
        else:
            return Scalar[dtype](0)
    
    fn get_item_unsafe(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element without bounds checking for performance-critical code."""
        var storage_index = self._get_storage_index(row, col)
        if storage_index >= 0:
            return self.data[storage_index]
        else:
            return Scalar[dtype](0)
    
    fn set_item(mut self, row: Int, col: Int, value: Scalar[dtype]) raises:
        """Set element at specified position."""
        if self._validation_enabled:
            if not self._is_valid_position(row, col):
                raise Error("Matrix indices out of bounds")
        
        var storage_index = self._get_storage_index(row, col)
        if storage_index >= 0:
            self.data[storage_index] = value
        else:
            if value != Scalar[dtype](0):
                raise Error("Cannot set non-zero value in triangular zero region")
    
    fn set_item_unsafe(mut self, row: Int, col: Int, value: Scalar[dtype]):
        """Set element without bounds checking for performance-critical code."""
        var storage_index = self._get_storage_index(row, col)
        if storage_index >= 0:
            self.data[storage_index] = value
    
    fn get_diagonal_element(self, index: Int) -> Scalar[dtype]:
        """Get diagonal element at specified index."""
        if self._validation_enabled:
            if index < 0 or index >= self.size:
                return Scalar[dtype](0)
        
        return self.get_item(index, index)
    
    fn set_diagonal_element(mut self, index: Int, value: Scalar[dtype]) raises:
        """Set diagonal element at specified index."""
        if self._validation_enabled:
            if index < 0 or index >= self.size:
                raise Error("Diagonal index out of bounds")
        
        self.set_item(index, index, value)
    
    fn fill_diagonal(mut self, value: Scalar[dtype]):
        """Fill diagonal with specified value."""
        for i in range(self.size):
            var _ = self.get_item(i, i)  # Get storage index
            var storage_index = self._get_storage_index(i, i)
            if storage_index >= 0:
                self.data[storage_index] = value
    
    fn scale_triangle(mut self, factor: Scalar[dtype]):
        """Scale all stored triangular elements by factor."""
        for i in range(self._storage_size):
            self.data[i] *= factor
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        var base_size = 4 * sizeof[Int]() + 2 * sizeof[Bool]()
        var data_size = self._storage_size * sizeof[Scalar[dtype]]()
        return base_size + data_size
    
    fn get_storage_efficiency(self) -> Float64:
        """Calculate storage efficiency compared to dense matrix."""
        var triangular_storage = self._storage_size
        var dense_storage = self.size * self.size
        return Float64(triangular_storage) / Float64(dense_storage)
    
    fn set_validation(mut self, enabled: Bool):
        """Enable or disable bounds checking for performance tuning."""
        self._validation_enabled = enabled
    
    fn transpose(self) raises -> TriangularTensor[dtype]:
        """Return transposed triangular tensor (flips upper/lower)."""
        var transposed = TriangularTensor[dtype](self.size, not self.is_upper)
        
        # Copy elements with transposed indices
        for i in range(self.size):
            for j in range(self.size):
                var value = self.get_item_unsafe(i, j)
                if value != Scalar[dtype](0):
                    transposed.set_item_unsafe(j, i, value)
        
        return transposed
    
    fn to_dense(self) raises -> DenseTensor[dtype]:
        """Convert to dense tensor representation."""
        var shape = List[Int]()
        shape.append(self.size)
        shape.append(self.size)
        
        var dense = DenseTensor[dtype](shape)
        
        for i in range(self.size):
            for j in range(self.size):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = self.get_item_unsafe(i, j)
                dense.set_item(indices, value)
        
        return dense
    
    fn print_info(self):
        """Print triangular tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var size_str: String = String(self.size)
        var storage_str: String = String(self._storage_size)
        var memory_str: String = String(self.get_memory_usage())
        var efficiency = self.get_storage_efficiency()
        var efficiency_str: String = String(efficiency * 100.0)
        var triangle_type = "Upper" if self.is_upper else "Lower"
        
        print("TriangularTensor[" + dtype_str + "] (" + triangle_type + ")")
        print("  Size: " + size_str + "x" + size_str)
        print("  Storage elements: " + storage_str)
        print("  Memory usage: " + memory_str + " bytes")
        print("  Storage efficiency: " + efficiency_str + "% of dense")
        print("  Validation: " + ("enabled" if self._validation_enabled else "disabled"))


struct DenseTensor[dtype: DType]:
    """Dense tensor implementation for conversion and compatibility."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1
        self._owns_data = True
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
            self.shape.append(shape[i])
            self._total_elements *= shape[i]
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for dense tensor."""
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        self.ndim = existing.ndim
        self._total_elements = existing._total_elements
        self._owns_data = True
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
    
    fn _get_linear_index(self, indices: List[Int]) -> Int:
        """Convert multi-dimensional indices to linear index."""
        var linear_index = 0
        var stride = 1
        
        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]
        
        return linear_index
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        return self.data[linear_index]
    
    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element at specified indices."""
        var linear_index = self._get_linear_index(indices)
        self.data[linear_index] = value
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        return self._total_elements
    
    fn get_memory_usage(self) -> Int:
        """Get memory usage in bytes."""
        return self._total_elements * sizeof[Scalar[dtype]]()
    
    fn print_info(self):
        """Print tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var memory_str: String = String(self.get_memory_usage())
        
        print("DenseTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)
        print("  Memory usage: " + memory_str + " bytes")


fn create_upper_triangular[dtype: DType](size: Int) raises -> TriangularTensor[dtype]:
    """Create upper triangular tensor."""
    return TriangularTensor[dtype](size, True)

fn create_lower_triangular[dtype: DType](size: Int) raises -> TriangularTensor[dtype]:
    """Create lower triangular tensor."""
    return TriangularTensor[dtype](size, False)

fn create_triangular_identity[dtype: DType](size: Int, is_upper: Bool = True, scale: Scalar[dtype] = Scalar[dtype](1)) raises -> TriangularTensor[dtype]:
    """Create triangular identity matrix."""
    var triangular = TriangularTensor[dtype](size, is_upper)
    triangular.fill_diagonal(scale)
    return triangular

fn create_triangular_from_values[dtype: DType](size: Int, values: List[Scalar[dtype]], is_upper: Bool = True) raises -> TriangularTensor[dtype]:
    """Create triangular tensor from list of values (row-major order)."""
    var triangular = TriangularTensor[dtype](size, is_upper)
    
    var value_index = 0
    for i in range(size):
        for j in range(size):
            if triangular._is_in_triangle(i, j) and value_index < len(values):
                triangular.set_item_unsafe(i, j, values[value_index])
                value_index += 1
    
    return triangular

fn dense_to_triangular[dtype: DType](dense: DenseTensor[dtype], is_upper: Bool = True, threshold: Float64 = 1e-10) raises -> TriangularTensor[dtype]:
    """Extract triangular part from dense tensor."""
    if dense.ndim != 2:
        raise Error("Can only extract triangular part from 2D tensors")
    if dense.shape[0] != dense.shape[1]:
        raise Error("Matrix must be square for triangular extraction")
    
    var size = dense.shape[0]
    var triangular = TriangularTensor[dtype](size, is_upper)
    
    # Extract triangular elements
    for i in range(size):
        for j in range(size):
            if triangular._is_in_triangle(i, j):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = dense.get_item(indices)
                
                if abs(Float64(value)) > threshold:
                    triangular.set_item_unsafe(i, j, value)
    
    return triangular


struct PerformanceTimer:
    """Simple performance timing utility."""
    var name: String
    
    fn __init__(out self, name: String):
        self.name = name
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
    
    fn start(self):
        """Start timing operation."""
        print("Starting: " + self.name)
    
    fn end(self, operations: Int = 1):
        """End timing and report results."""
        var ops_str: String = String(operations)
        print("Completed: " + self.name + " (" + ops_str + " operations)")

fn benchmark_triangular_creation[dtype: DType](size: Int, iterations: Int) -> Float64:
    """Benchmark triangular tensor creation performance."""
    var timer = PerformanceTimer("Triangular Creation Benchmark")
    timer.start()
    
    try:
        for _ in range(iterations):
            var triangular = create_upper_triangular[dtype](size)
            var _ = triangular.get_item(0, 0)
    except:
        print("Benchmark failed")
        return -1.0
    
    timer.end(iterations)
    return Float64(iterations) / 1000.0

fn benchmark_triangular_memory_efficiency[dtype: DType](size: Int) -> Float64:
    """Compare memory usage: triangular vs dense."""
    try:
        var triangular = create_upper_triangular[dtype](size)
        var dense = triangular.to_dense()
        
        var triangular_memory = triangular.get_memory_usage()
        var dense_memory = dense.get_memory_usage()
        
        var efficiency = Float64(triangular_memory) / Float64(dense_memory)
        
        var triangular_str: String = String(triangular_memory)
        var dense_str: String = String(dense_memory)
        var efficiency_str: String = String(efficiency * 100.0)
        
        print("Memory Efficiency Analysis:")
        print("  Triangular tensor: " + triangular_str + " bytes")
        print("  Dense equivalent: " + dense_str + " bytes")
        print("  Efficiency: " + efficiency_str + "% of dense storage")
        
        return efficiency
        
    except:
        print("Memory benchmark failed")
        return -1.0


fn test_basic_triangular_operations():
    """Test basic triangular tensor functionality."""
    print("=== Testing Basic Triangular Operations ===")
    
    try:
        print("\n1. Upper Triangular Tensor Creation:")
        var upper = create_upper_triangular[DType.float32](4)
        upper.print_info()
        
        # Fill with some values
        upper.set_item(0, 0, 1.0)
        upper.set_item(0, 1, 2.0)
        upper.set_item(0, 2, 3.0)
        upper.set_item(0, 3, 4.0)
        upper.set_item(1, 1, 5.0)
        upper.set_item(1, 2, 6.0)
        upper.set_item(1, 3, 7.0)
        upper.set_item(2, 2, 8.0)
        upper.set_item(2, 3, 9.0)
        upper.set_item(3, 3, 10.0)
        
        print("\nUpper triangular matrix values (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = upper.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Lower Triangular Tensor Creation:")
        var lower = create_lower_triangular[DType.float32](4)
        lower.print_info()
        
        # Fill with some values
        lower.set_item(0, 0, 10.0)
        lower.set_item(1, 0, 9.0)
        lower.set_item(1, 1, 8.0)
        lower.set_item(2, 0, 7.0)
        lower.set_item(2, 1, 6.0)
        lower.set_item(2, 2, 5.0)
        lower.set_item(3, 0, 4.0)
        lower.set_item(3, 1, 3.0)
        lower.set_item(3, 2, 2.0)
        lower.set_item(3, 3, 1.0)
        
        print("\nLower triangular matrix values (4x4):")
        for i in range(4):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(4):
                var value = lower.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n3. Storage Efficiency:")
        var upper_efficiency = upper.get_storage_efficiency()
        var lower_efficiency = lower.get_storage_efficiency()
        var upper_eff_str: String = String(upper_efficiency * 100.0)
        var lower_eff_str: String = String(lower_efficiency * 100.0)
        
        print("Upper triangular storage efficiency: " + upper_eff_str + "%")
        print("Lower triangular storage efficiency: " + lower_eff_str + "%")
        
        print("Basic triangular operations completed successfully")
        
    except:
        print("Error in basic triangular operations")

fn test_triangular_mathematical_operations():
    """Test mathematical operations on triangular tensors."""
    print("\n=== Testing Triangular Mathematical Operations ===")
    
    try:
        print("\n1. Triangular Identity Creation:")
        var upper_identity = create_triangular_identity[DType.float32](3, True, 2.0)
        var lower_identity = create_triangular_identity[DType.float32](3, False, 3.0)
        
        print("Upper triangular identity:")
        upper_identity.print_info()
        
        print("Lower triangular identity:")
        lower_identity.print_info()
        
        print("\n2. Diagonal Element Access:")
        for i in range(3):
            var upper_diag = upper_identity.get_diagonal_element(i)
            var lower_diag = lower_identity.get_diagonal_element(i)
            var i_str: String = String(i)
            var upper_str: String = String(upper_diag)
            var lower_str: String = String(lower_diag)
            print("  Diagonal[" + i_str + "]: upper=" + upper_str + ", lower=" + lower_str)
        
        print("\n3. Triangular Scaling:")
        var scaled_upper = create_upper_triangular[DType.float32](3)
        scaled_upper.fill_diagonal(1.0)
        scaled_upper.set_item(0, 1, 2.0)
        scaled_upper.set_item(0, 2, 3.0)
        scaled_upper.set_item(1, 2, 4.0)
        
        print("Before scaling:")
        for i in range(3):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(3):
                var value = scaled_upper.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        scaled_upper.scale_triangle(2.0)
        
        print("After scaling by 2.0:")
        for i in range(3):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(3):
                var value = scaled_upper.get_item(i, j)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n4. Triangular Transpose:")
        var original = create_upper_triangular[DType.float32](3)
        original.set_item(0, 0, 1.0)
        original.set_item(0, 1, 2.0)
        original.set_item(0, 2, 3.0)
        original.set_item(1, 1, 4.0)
        original.set_item(1, 2, 5.0)
        original.set_item(2, 2, 6.0)
        
        var transposed = original.transpose()
        
        print("Original upper triangular:")
        original.print_info()
        print("Transposed (becomes lower):")
        transposed.print_info()
        
        print("Mathematical operations completed successfully")
        
    except:
        print("Error in mathematical operations")

fn test_triangular_conversions():
    """Test conversions between triangular and dense formats."""
    print("\n=== Testing Triangular Conversions ===")
    
    try:
        print("\n1. Triangular to Dense Conversion:")
        var values = List[Scalar[DType.float32]]()
        values.append(1.0)
        values.append(2.0)
        values.append(3.0)
        values.append(4.0)
        values.append(5.0)
        values.append(6.0)
        
        var triangular = create_triangular_from_values[DType.float32](3, values, True)
        
        print("Original triangular:")
        triangular.print_info()
        
        var dense = triangular.to_dense()
        print("Converted dense:")
        dense.print_info()
        
        print("Dense matrix values:")
        for i in range(3):
            var i_str: String = String(i)
            print("  Row " + i_str + ": ", end="")
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = dense.get_item(indices)
                var value_str: String = String(value)
                print(value_str + " ", end="")
            print("")
        
        print("\n2. Dense to Triangular Extraction:")
        var extracted_upper = dense_to_triangular[DType.float32](dense, True)
        var extracted_lower = dense_to_triangular[DType.float32](dense, False)
        
        print("Extracted upper triangular:")
        extracted_upper.print_info()
        print("Extracted lower triangular:")
        extracted_lower.print_info()
        
        print("\n3. Conversion Accuracy Verification:")
        var errors = 0
        for i in range(3):
            for j in range(3):
                if triangular._is_in_triangle(i, j):
                    var original_val = triangular.get_item(i, j)
                    var extracted_val = extracted_upper.get_item(i, j)
                    
                    if original_val != extracted_val:
                        errors += 1
                        var i_str: String = String(i)
                        var j_str: String = String(j)
                        var orig_str: String = String(original_val)
                        var extr_str: String = String(extracted_val)
                        print("  Mismatch at [" + i_str + ", " + j_str + "]: " + orig_str + " vs " + extr_str)
        
        var errors_str: String = String(errors)
        if errors == 0:
            print("Conversion accuracy: Perfect (0 errors)")
        else:
            print("Conversion errors: " + errors_str)
        
        print("Conversion testing completed successfully")
        
    except:
        print("Error in conversion testing")

fn test_triangular_performance():
    """Test triangular tensor performance characteristics."""
    print("\n=== Testing Triangular Performance ===")
    
    try:
        print("\n1. Creation Performance:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(500)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var iterations = 1000
            var size_str: String = String(size)
            print("Testing " + size_str + "x" + size_str + " triangular creation...")
            
            var result = benchmark_triangular_creation[DType.float32](size, iterations)
            var result_str: String = String(result)
            print("  Performance metric: " + result_str + " k-ops")
        
        print("\n2. Memory Efficiency Comparison:")
        for i in range(len(sizes)):
            var size = sizes[i]
            var size_str: String = String(size)
            print("Memory analysis for " + size_str + "x" + size_str + " matrix:")
            var _ = benchmark_triangular_memory_efficiency[DType.float32](size)
        
        print("\n3. Access Pattern Performance:")
        var large_triangular = create_upper_triangular[DType.float32](1000)
        large_triangular.set_validation(False)  # Disable bounds checking
        
        # Fill diagonal
        for i in range(1000):
            large_triangular.set_item_unsafe(i, i, Scalar[DType.float32](i + 1))
        
        var timer = PerformanceTimer("Triangular Element Access (1M operations)")
        timer.start()
        
        var sum = Scalar[DType.float32](0)
        for _ in range(1000):
            for i in range(1000):
                sum += large_triangular.get_item_unsafe(i, i)
        
        timer.end(1000000)
        var sum_str: String = String(sum)
        print("Sum result: " + sum_str + " (validation)")
        
        print("\n4. Storage Efficiency Analysis:")
        for i in range(len(sizes)):
            var size = sizes[i]
            var triangular = create_upper_triangular[DType.float32](size)
            var efficiency = triangular.get_storage_efficiency()
            var memory_savings = Float64(1.0) - efficiency
            memory_savings *= 100.0
            
            var size_str: String = String(size)
            var efficiency_str: String = String(efficiency * 100.0)
            var savings_str: String = String(memory_savings)
            
            print("  " + size_str + "x" + size_str + " matrix:")
            print("    Storage efficiency: " + efficiency_str + "% of dense")
            print("    Memory savings: " + savings_str + "%")
        
        print("Performance testing completed successfully")
        
    except:
        print("Error in performance testing")

fn test_triangular_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Triangular Error Handling ===")
    
    var test_count = 0
    var passed_count = 0
    
    print("\n1. Invalid Size Errors:")
    test_count += 1
    try:
        var _ = create_upper_triangular[DType.float32](0)
        print("Should have failed with zero size")
    except:
        print("Correctly caught zero size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_upper_triangular[DType.float32](-5)
        print("Should have failed with negative size")
    except:
        print("Correctly caught negative size error")
        passed_count += 1
    
    test_count += 1
    try:
        var _ = create_upper_triangular[DType.float32](10000)
        print("Should have failed with oversized matrix")
    except:
        print("Correctly caught oversized matrix error")
        passed_count += 1
    
    print("\n2. Invalid Element Assignment:")
    test_count += 1
    try:
        var upper = create_upper_triangular[DType.float32](3)
        upper.set_item(2, 0, 5.0)  # Lower triangle in upper matrix
        print("Should have failed with invalid triangle assignment")
    except:
        print("Correctly caught invalid triangle assignment")
        passed_count += 1
    
    test_count += 1
    try:
        var lower = create_lower_triangular[DType.float32](3)
        lower.set_item(0, 2, 5.0)  # Upper triangle in lower matrix
        print("Should have failed with invalid triangle assignment")
    except:
        print("Correctly caught invalid triangle assignment")
        passed_count += 1
    
    print("\n3. Bounds Checking:")
    test_count += 1
    try:
        var triangular = create_upper_triangular[DType.float32](3)
        triangular.set_diagonal_element(5, 2.0)  # Out of bounds
        print("Should have failed with out-of-bounds diagonal access")
    except:
        print("Correctly caught out-of-bounds diagonal access")
        passed_count += 1
    
    print("\n4. Edge Case Testing:")
    test_count += 1
    try:
        var tiny = create_upper_triangular[DType.float32](1)
        tiny.set_item(0, 0, 5.0)
        var value = tiny.get_item(0, 0)
        if value == Scalar[DType.float32](5.0):
            print("1x1 triangular matrix works correctly")
            passed_count += 1
        else:
            print("1x1 triangular matrix incorrect value")
    except:
        print("1x1 triangular matrix creation failed")
    
    test_count += 1
    try:
        var triangular = create_upper_triangular[DType.float32](3)
        var out_of_bounds = triangular.get_item(5, 5)
        if out_of_bounds == Scalar[DType.float32](0):
            print("Out-of-bounds access returns zero")
            passed_count += 1
        else:
            print("Out-of-bounds access incorrect value")
    except:
        print("Out-of-bounds access handling failed")
    
    var passed_str: String = String(passed_count)
    var total_str: String = String(test_count)
    print("\nError Handling Summary: " + passed_str + "/" + total_str + " tests passed")
    
    if passed_count == test_count:
        print("All error handling tests passed")
    else:
        print("Some error handling tests failed")

fn test_triangular_memory_management():
    """Test memory management and resource cleanup."""
    print("\n=== Testing Triangular Memory Management ===")
    
    try:
        print("\n1. Memory Usage Analysis:")
        var sizes = List[Int]()
        sizes.append(10)
        sizes.append(100)
        sizes.append(500)
        
        for i in range(len(sizes)):
            var size = sizes[i]
            var triangular = create_upper_triangular[DType.float32](size)
            var memory = triangular.get_memory_usage()
            var efficiency = triangular.get_storage_efficiency()
            var storage_elements = triangular._storage_size
            
            var size_str: String = String(size)
            var memory_str: String = String(memory)
            var efficiency_str: String = String(efficiency * 100.0)
            var storage_str: String = String(storage_elements)
            var total_elements = size * size
            var total_str: String = String(total_elements)
            
            print("  " + size_str + "x" + size_str + " matrix:")
            print("    Triangular storage: " + memory_str + " bytes")
            print("    Storage elements: " + storage_str + "/" + total_str)
            print("    Memory efficiency: " + efficiency_str + "% of dense")
        
        print("\n2. Copy Constructor Testing:")
        var original = create_triangular_identity[DType.float32](5, True, 3.0)
        var copied = original
        
        print("Original:")
        original.print_info()
        print("Copy:")
        copied.print_info()
        
        # Verify independence by modifying original
        original.scale_triangle(2.0)
        
        var orig_val = original.get_diagonal_element(0)
        var copy_val = copied.get_diagonal_element(0)
        
        if orig_val != copy_val:
            print("Copy constructor creates independent copy")
        else:
            print("Copy constructor may not be independent")
        
        print("\n3. Large Triangular Creation/Destruction:")
        var creation_timer = PerformanceTimer("Large Triangular Creation/Destruction")
        creation_timer.start()
        
        for _ in range(100):
            var large_triangular = create_upper_triangular[DType.float32](1000)
            var _ = large_triangular.get_item(0, 0)
        
        creation_timer.end(100)
        print("Large triangular creation/destruction completed")
        
        print("Memory management testing completed successfully")
        
    except:
        print("Error in memory management testing")

fn run_all_triangular_tests():
    """Run comprehensive test suite for triangular tensors."""
    print("=====================================")
    print("=== TRIANGULAR TENSOR TEST SUITE ===")
    print("=====================================")
    
    test_basic_triangular_operations()
    test_triangular_mathematical_operations()
    test_triangular_conversions()
    test_triangular_performance()
    test_triangular_error_handling()
    test_triangular_memory_management()
    
    print("\n=====================================")
    print("=== TRIANGULAR TENSOR TESTS COMPLETE ===")
    print("=====================================")


fn print_specialized_tensor_ecosystem_summary():
    """Print comprehensive summary of all specialized tensor types."""
    print("\n" + "=" * 60)
    print("=== SPECIALIZED TENSOR ECOSYSTEM COMPLETE ===")
    print("=" * 60)
    
    print("\nPart 1.3.4 - Specialized Tensor Types Implementation Summary:")
    print("")
    
    print("A. Identity Tensors (Part 1.3.4A):")
    print("   + Memory-efficient O(1) storage regardless of matrix size")
    print("   + Configurable scaling factors and diagonal offsets")
    print("   + Fast mathematical operations (transpose, scaling)")
    print("   + Support for super-diagonal and sub-diagonal variants")
    print("   + Constant ~20 bytes memory usage per identity tensor")
    print("")
    
    print("B. Diagonal Tensors (Part 1.3.4B):")
    print("   + Memory-efficient O(n) storage for diagonal elements")
    print("   + Support for multiple diagonal bands (main, super, sub)")
    print("   + Multi-diagonal tensor support (tridiagonal, pentadiagonal)")
    print("   + Optimized diagonal extraction and manipulation")
    print("   + Memory usage proportional to diagonal count, not matrix size")
    print("")
    
    print("C. Sparse Tensors (Part 1.3.4C):")
    print("   + Memory-efficient O(nnz) storage for non-zero elements")
    print("   + COO (Coordinate) format for flexible sparse construction")
    print("   + Automatic zero element detection and removal")
    print("   + Specialized sparse matrix creation (identity, diagonal)")
    print("   + Dense-to-sparse conversion with configurable thresholds")
    print("")
    
    print("D. Triangular Tensors (Part 1.3.4D):")
    print("   + Memory-efficient O(n²/2) storage for triangular matrices")
    print("   + Support for both upper and lower triangular formats")
    print("   + Packed triangular storage saving ~50% memory vs dense")
    print("   + Fast triangular operations and linear algebra support")
    print("   + Seamless conversion to/from dense format")
    print("")
    
    print("Integration Features:")
    print("+ Unified factory function interfaces across all types")
    print("+ Consistent error handling and validation frameworks")
    print("+ Performance benchmarking and memory analysis tools")
    print("+ Seamless conversions between specialized and dense formats")
    print("+ Comprehensive testing suites for all implementations")
    print("+ Memory management with automatic resource cleanup")
    print("")
    
    print("Performance Characteristics Summary:")
    print("- Identity Tensors:   O(1) memory,   O(1) access")
    print("- Diagonal Tensors:   O(n) memory,   O(1) access")
    print("- Sparse Tensors:     O(nnz) memory, O(nnz) access")
    print("- Triangular Tensors: O(n²/2) memory, O(1) access")
    print("")
    
    print("Mathematical Operations Supported:")
    print("+ Element access and modification with bounds checking")
    print("+ Matrix transpose operations")
    print("+ Scaling and diagonal manipulation")
    print("+ Format conversions and type casting")
    print("+ Specialized matrix creation utilities")
    print("+ Memory compression and optimization")
    print("")
    
    print("Integration Readiness:")
    print("+ All specialized tensor types are standalone and complete")
    print("+ Ready for integration with broader tensor library ecosystem")
    print("+ Compatible with existing tensor infrastructure")
    print("+ Extensible framework for additional specialized types")
    print("+ Comprehensive documentation and testing coverage")
    print("")
    
    print("Next Integration Steps:")
    print("1. Integrate with Device Abstraction Layer (Part 1.4)")
    print("2. Add GPU acceleration support for specialized operations")
    print("3. Implement specialized linear algebra routines")
    print("4. Add automatic format selection based on sparsity patterns")
    print("5. Extend to support additional specialized matrix types")
    print("")
    
    print("=" * 60)
    print("=== SPECIALIZED TENSOR IMPLEMENTATION COMPLETE ===")
    print("=" * 60)


fn main():
    """Main function to run triangular tensor implementation tests."""
    print("=== Specialized Tensor Types - Part 1.3.4D ===")
    print("Triangular Tensor Implementation - Complete Standalone Module")
    print("")
    
    run_all_triangular_tests()
    
    print("\n=== Triangular Tensor Implementation Summary ===")
    print("+ Memory-efficient triangular matrix representation")
    print("+ Support for both upper and lower triangular formats")
    print("+ Packed storage saving ~50% memory compared to dense")
    print("+ Fast element access with triangular indexing")
    print("+ Optimized triangular matrix operations")
    print("+ Comprehensive error handling and validation")
    print("+ Performance benchmarking and memory analysis")
    print("+ Seamless conversion to/from dense format")
    print("+ Integration-ready for tensor library ecosystem")
    print("")
    print("Performance Characteristics:")
    print("- Memory usage: O(n²/2) - half of dense storage")
    print("- Element access: O(1) - direct triangular indexing")
    print("- Storage efficiency: ~50% of dense matrix storage")
    print("- Conversion to dense: O(n²) - only when explicitly needed")
    print("")
    print("Mathematical Features:")
    print("- Upper and lower triangular matrices")
    print("- Triangular identity matrix creation")
    print("- Diagonal element access and manipulation")
    print("- Triangular matrix transpose operations")
    print("- Scaling operations for all triangular elements")
    print("- Dense matrix triangular extraction")
    print("")
    print("Triangular Tensor Implementation (Part 1.3.4D) Complete")
    print("")
    
    # Print comprehensive ecosystem summary
    print_specialized_tensor_ecosystem_summary()
```

### Expected Output for `45_specialized_tensors_part_d.mojo`

```
=== Specialized Tensor Types - Part 1.3.4D ===
Triangular Tensor Implementation - Complete Standalone Module

=====================================
=== TRIANGULAR TENSOR TEST SUITE ===
=====================================
=== Testing Basic Triangular Operations ===

1. Upper Triangular Tensor Creation:
TriangularTensor[float32] (Upper)
  Size: 4x4
  Storage elements: 10
  Memory usage: 74 bytes
  Storage efficiency: 62.5% of dense
  Validation: enabled

Upper triangular matrix values (4x4):
  Row 0: 1.0 2.0 3.0 5.0 
  Row 1: 0.0 5.0 6.0 8.0 
  Row 2: 0.0 0.0 8.0 10.0 
  Row 3: 0.0 0.0 0.0 10.0 

2. Lower Triangular Tensor Creation:
TriangularTensor[float32] (Lower)
  Size: 4x4
  Storage elements: 10
  Memory usage: 74 bytes
  Storage efficiency: 62.5% of dense
  Validation: enabled

Lower triangular matrix values (4x4):
  Row 0: 10.0 0.0 0.0 0.0 
  Row 1: 9.0 8.0 0.0 0.0 
  Row 2: 7.0 6.0 5.0 0.0 
  Row 3: 4.0 3.0 2.0 1.0 

3. Storage Efficiency:
Upper triangular storage efficiency: 62.5%
Lower triangular storage efficiency: 62.5%
Basic triangular operations completed successfully

=== Testing Triangular Mathematical Operations ===

1. Triangular Identity Creation:
Upper triangular identity:
TriangularTensor[float32] (Upper)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled
Lower triangular identity:
TriangularTensor[float32] (Lower)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled

2. Diagonal Element Access:
  Diagonal[0]: upper=2.0, lower=3.0
  Diagonal[1]: upper=2.0, lower=3.0
  Diagonal[2]: upper=2.0, lower=3.0

3. Triangular Scaling:
Before scaling:
  Row 0: 1.0 2.0 3.0 
  Row 1: 0.0 3.0 4.0 
  Row 2: 0.0 0.0 4.0 
After scaling by 2.0:
  Row 0: 2.0 4.0 6.0 
  Row 1: 0.0 6.0 8.0 
  Row 2: 0.0 0.0 8.0 

4. Triangular Transpose:
Original upper triangular:
TriangularTensor[float32] (Upper)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled
Transposed (becomes lower):
TriangularTensor[float32] (Lower)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled
Mathematical operations completed successfully

=== Testing Triangular Conversions ===

1. Triangular to Dense Conversion:
Original triangular:
TriangularTensor[float32] (Upper)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled
Converted dense:
DenseTensor[float32]
  Shape: [3, 3]
  Elements: 9
  Memory usage: 36 bytes
Dense matrix values:
  Row 0: 1.0 2.0 4.0 
  Row 1: 0.0 4.0 6.0 
  Row 2: 0.0 0.0 6.0 

2. Dense to Triangular Extraction:
Extracted upper triangular:
TriangularTensor[float32] (Upper)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled
Extracted lower triangular:
TriangularTensor[float32] (Lower)
  Size: 3x3
  Storage elements: 6
  Memory usage: 58 bytes
  Storage efficiency: 66.66666666666666% of dense
  Validation: enabled

3. Conversion Accuracy Verification:
Conversion accuracy: Perfect (0 errors)
Conversion testing completed successfully

=== Testing Triangular Performance ===

1. Creation Performance:
Testing 10x10 triangular creation...
Starting: Triangular Creation Benchmark
Completed: Triangular Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 100x100 triangular creation...
Starting: Triangular Creation Benchmark
Completed: Triangular Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops
Testing 500x500 triangular creation...
Starting: Triangular Creation Benchmark
Completed: Triangular Creation Benchmark (1000 operations)
  Performance metric: 1.0 k-ops

2. Memory Efficiency Comparison:
Memory analysis for 10x10 matrix:
Memory Efficiency Analysis:
  Triangular tensor: 254 bytes
  Dense equivalent: 400 bytes
  Efficiency: 63.5% of dense storage
Memory analysis for 100x100 matrix:
Memory Efficiency Analysis:
  Triangular tensor: 20234 bytes
  Dense equivalent: 40000 bytes
  Efficiency: 50.585% of dense storage
Memory analysis for 500x500 matrix:
Memory Efficiency Analysis:
  Triangular tensor: 501034 bytes
  Dense equivalent: 1000000 bytes
  Efficiency: 50.1034% of dense storage

3. Access Pattern Performance:
Starting: Triangular Element Access (1M operations)
Completed: Triangular Element Access (1M operations) (1000000 operations)
Sum result: 499993470.0 (validation)

4. Storage Efficiency Analysis:
  10x10 matrix:
    Storage efficiency: 55.00000000000001% of dense
    Memory savings: 44.99999999999999%
  100x100 matrix:
    Storage efficiency: 50.5% of dense
    Memory savings: 49.5%
  500x500 matrix:
    Storage efficiency: 50.1% of dense
    Memory savings: 49.9%
Performance testing completed successfully

=== Testing Triangular Error Handling ===

1. Invalid Size Errors:
Correctly caught zero size error
Correctly caught negative size error
Correctly caught oversized matrix error

2. Invalid Element Assignment:
Correctly caught invalid triangle assignment
Correctly caught invalid triangle assignment

3. Bounds Checking:
Correctly caught out-of-bounds diagonal access

4. Edge Case Testing:
1x1 triangular matrix works correctly
Out-of-bounds access returns zero

Error Handling Summary: 8/8 tests passed
All error handling tests passed

=== Testing Triangular Memory Management ===

1. Memory Usage Analysis:
  10x10 matrix:
    Triangular storage: 254 bytes
    Storage elements: 55/100
    Memory efficiency: 55.00000000000001% of dense
  100x100 matrix:
    Triangular storage: 20234 bytes
    Storage elements: 5050/10000
    Memory efficiency: 50.5% of dense
  500x500 matrix:
    Triangular storage: 501034 bytes
    Storage elements: 125250/250000
    Memory efficiency: 50.1% of dense

2. Copy Constructor Testing:
Original:
TriangularTensor[float32] (Upper)
  Size: 5x5
  Storage elements: 15
  Memory usage: 94 bytes
  Storage efficiency: 60.0% of dense
  Validation: enabled
Copy:
TriangularTensor[float32] (Upper)
  Size: 5x5
  Storage elements: 15
  Memory usage: 94 bytes
  Storage efficiency: 60.0% of dense
  Validation: enabled
Copy constructor creates independent copy

3. Large Triangular Creation/Destruction:
Starting: Large Triangular Creation/Destruction
Completed: Large Triangular Creation/Destruction (100 operations)
Large triangular creation/destruction completed
Memory management testing completed successfully

=====================================
=== TRIANGULAR TENSOR TESTS COMPLETE ===
=====================================

=== Triangular Tensor Implementation Summary ===
+ Memory-efficient triangular matrix representation
+ Support for both upper and lower triangular formats
+ Packed storage saving ~50% memory compared to dense
+ Fast element access with triangular indexing
+ Optimized triangular matrix operations
+ Comprehensive error handling and validation
+ Performance benchmarking and memory analysis
+ Seamless conversion to/from dense format
+ Integration-ready for tensor library ecosystem

Performance Characteristics:
- Memory usage: O(n²/2) - half of dense storage
- Element access: O(1) - direct triangular indexing
- Storage efficiency: ~50% of dense matrix storage
- Conversion to dense: O(n²) - only when explicitly needed

Mathematical Features:
- Upper and lower triangular matrices
- Triangular identity matrix creation
- Diagonal element access and manipulation
- Triangular matrix transpose operations
- Scaling operations for all triangular elements
- Dense matrix triangular extraction

Triangular Tensor Implementation (Part 1.3.4D) Complete


============================================================
=== SPECIALIZED TENSOR ECOSYSTEM COMPLETE ===
============================================================

Part 1.3.4 - Specialized Tensor Types Implementation Summary:

A. Identity Tensors (Part 1.3.4A):
   + Memory-efficient O(1) storage regardless of matrix size
   + Configurable scaling factors and diagonal offsets
   + Fast mathematical operations (transpose, scaling)
   + Support for super-diagonal and sub-diagonal variants
   + Constant ~20 bytes memory usage per identity tensor

B. Diagonal Tensors (Part 1.3.4B):
   + Memory-efficient O(n) storage for diagonal elements
   + Support for multiple diagonal bands (main, super, sub)
   + Multi-diagonal tensor support (tridiagonal, pentadiagonal)
   + Optimized diagonal extraction and manipulation
   + Memory usage proportional to diagonal count, not matrix size

C. Sparse Tensors (Part 1.3.4C):
   + Memory-efficient O(nnz) storage for non-zero elements
   + COO (Coordinate) format for flexible sparse construction
   + Automatic zero element detection and removal
   + Specialized sparse matrix creation (identity, diagonal)
   + Dense-to-sparse conversion with configurable thresholds

D. Triangular Tensors (Part 1.3.4D):
   + Memory-efficient O(n²/2) storage for triangular matrices
   + Support for both upper and lower triangular formats
   + Packed triangular storage saving ~50% memory vs dense
   + Fast triangular operations and linear algebra support
   + Seamless conversion to/from dense format

Integration Features:
+ Unified factory function interfaces across all types
+ Consistent error handling and validation frameworks
+ Performance benchmarking and memory analysis tools
+ Seamless conversions between specialized and dense formats
+ Comprehensive testing suites for all implementations
+ Memory management with automatic resource cleanup

Performance Characteristics Summary:
- Identity Tensors:   O(1) memory,   O(1) access
- Diagonal Tensors:   O(n) memory,   O(1) access
- Sparse Tensors:     O(nnz) memory, O(nnz) access
- Triangular Tensors: O(n²/2) memory, O(1) access

Mathematical Operations Supported:
+ Element access and modification with bounds checking
+ Matrix transpose operations
+ Scaling and diagonal manipulation
+ Format conversions and type casting
+ Specialized matrix creation utilities
+ Memory compression and optimization

Integration Readiness:
+ All specialized tensor types are standalone and complete
+ Ready for integration with broader tensor library ecosystem
+ Compatible with existing tensor infrastructure
+ Extensible framework for additional specialized types
+ Comprehensive documentation and testing coverage

Next Integration Steps:
1. Integrate with Device Abstraction Layer (Part 1.4)
2. Add GPU acceleration support for specialized operations
3. Implement specialized linear algebra routines
4. Add automatic format selection based on sparsity patterns
5. Extend to support additional specialized matrix types

============================================================
=== SPECIALIZED TENSOR IMPLEMENTATION COMPLETE ===
============================================================
```

## Chapter Summary

`IdentityTensor` needs no data buffer at all: three scalars and a bool answer every possible query through a single `col - row == offset` comparison, which is why its memory usage is a flat, size-independent 29 bytes whether the matrix it represents is 10×10 or 1000×1000 — the number describes the rule, not the (never-materialized) matrix. `DiagonalTensor` gives that up for real, size-proportional storage — `34 + diagonal_length × sizeof[Scalar]` bytes — in exchange for direct O(1) access to any diagonal position by index, a trade that pays off only once you look past a single diagonal: wrapping three of them in a `MultiBandDiagonal` to build a tridiagonal matrix pays a fixed ~34-byte overhead *per band*, which this chapter traced to a concrete, verified case (a 4×4 tridiagonal matrix, 167 bytes of sparse storage against 64 bytes dense — 2.6× larger, not smaller). `SparseTensorCOO` drops the diagonal's structural guarantee entirely, storing arbitrary `(row, col, value)` triples at the cost of linear-time lookup, and its own memory accounting compounds the same fixed-overhead lesson: `get_memory_usage()` is driven by `capacity` (the reservation), not `num_elements` (the tenancy), which is why a small sparse matrix at the default `capacity=1000` can cost 240% of its dense equivalent, and why `compress_storage()` — which only rearranges live elements within an already-allocated buffer — cannot fix that. `TriangularTensor` closed the chapter with packed storage's genuine, verifiable win (efficiency provably approaching 50% of dense as size grows, with a correct lower-triangular formula to prove the idea works) sitting right next to a genuine, hand-traceable bug: the upper-triangular packing formula's base-offset term has a sign error that silently aliases distinct `(row, col)` positions onto the same storage slot, an error this chapter derived from first principles, then confirmed against the source's own captured output down to the exact corrupted digit.

## Self-Check Questions

1. `IdentityTensor.get_memory_usage()` returns the same 29 bytes for a `size=5` and a `size=5,000,000` tensor. Explain concretely why this is correct behavior rather than a bug, and name the one thing this struct genuinely cannot do that a `DiagonalTensor` or `DenseTensor` can.
2. A `DiagonalTensor` at `offset=3` on a `size=5` matrix has `diagonal_length = 2`. Using `get_memory_usage() = 34 + diagonal_length × sizeof[Scalar[dtype]]()`, compute its memory usage for `dtype=DType.float32`, and compute the dense equivalent (`5×5×4` bytes). Which is smaller, and by roughly what factor?
3. `SparseTensorCOO` is constructed with an explicit `capacity=50` instead of the default `1000`, for a `20×20` matrix holding 8 nonzero elements. Using `get_memory_usage() = 34 + capacity × 24 + shape_bytes` (16 bytes for a 2-element shape list), compute its reported memory usage, and compare it to the `20×20` dense equivalent (`20×20×4` bytes). Does the smaller capacity change which one wins, compared to the default-capacity case traced in Worked Example 9.3.2's COMMON TRAP?
4. Using the *correct* upper-triangular base formula derived in Worked Example 9.4.1 (`base(r) = r×(2×size - r + 1) // 2`), compute the correct storage index for `(row=2, col=3)` on a `size=4` matrix, and compare it to the index the actual (buggy) code computes for the same position. Do they collide with any other valid `(row, col)` pair under the buggy formula?
5. `TriangularTensor`'s lower-triangular branch (`(row*(row+1))//2 + col`) has no aliasing bug. Explain, in your own words, why counting "elements in all rows before row `r`" gives a different — and in this case correct — formula for the lower-triangular case than for the upper-triangular one.

## Where We Go Next

Chapter 10 (`part1/05-device-abstraction-layer.md`) moves from tensor *shape* to tensor *location* — the question of which physical device (CPU, GPU) a tensor's data actually lives on, and how operations get dispatched to the right one. Every specialized structure in this chapter (`IdentityTensor`'s rule-only representation, `DiagonalTensor`'s compact buffer, `SparseTensorCOO`'s coordinate list, `TriangularTensor`'s packed storage) has so far assumed its data — where it has any — lives in ordinary host memory; Chapter 10 is where that assumption starts to be made explicit and, eventually, optional.

## Worked Solutions

**1.** It's correct because `get_memory_usage()`'s formula — `3 × sizeof[Int]() + sizeof[Scalar[dtype]]() + sizeof[Bool]()` — never references `size` at all; the struct's only fields are `size`, `scale`, `offset`, and `_validation_enabled`, and none of those fields' own sizes change as the matrix they describe gets larger. What `IdentityTensor` genuinely cannot do is represent a matrix with even one entry that breaks the "constant `scale` on one diagonal" pattern — there is no `set_item` to write an arbitrary value into an arbitrary position, because there is no per-element storage for such a value to live in. Both `DiagonalTensor` (for arbitrary values restricted to a diagonal) and `DenseTensor` (for arbitrary values anywhere) can do this; `IdentityTensor` trades that flexibility for its constant-memory guarantee.

**2.** Memory usage: `34 + 2 × 4 = 42` bytes. Dense equivalent: `5 × 5 × 4 = 100` bytes. The diagonal representation is smaller, by a factor of `100 / 42 ≈ 2.38×` — a real, if modest, saving, unlike the tridiagonal case in Worked Example 9.2.3 where three bands' combined fixed overhead pushed the sparse representation *past* dense. A single diagonal band, even a short one, still comes out ahead here because there's only one 34-byte overhead to amortize, not three.

**3.** Memory usage: `34 + 50 × 24 + 16 = 34 + 1200 + 16 = 1250` bytes. Dense equivalent: `20 × 20 × 4 = 1600` bytes. `1250 / 1600 = 78.125%` — smaller than dense, unlike the default-capacity case. Yes, the smaller capacity changes the outcome: at the default `capacity=1000`, the same 8-element, `20×20` sparse tensor would report `34 + 1000×24 + 16 = 24050` bytes — over 15× larger than the `1600`-byte dense equivalent. The crossover point traced in the COMMON TRAP moves directly with `capacity`: reserving only as much room as you actually expect to use (50, not 1000, for 8 elements) is what makes COO sparse storage competitive at small sizes; the default capacity is tuned for cases with far more nonzero elements than this one has.

**4.** Correct base for row 2, size 4: `base(2) = 2×(2×4 - 2 + 1)//2 = 2×7//2 = 7`. Correct index for `(2,3)`: `7 + (3-2) = 8`. The buggy code's base for row 2, size 4 (from Worked Example 9.4.1's table) is `5`, giving buggy index `5 + (3-2) = 6`. Checking whether index 6 collides with any other position under the buggy formula: row 3's buggy base is `6`, and `(3,3)` (the only valid position in row 3) computes to `base(3) + (3-3) = 6 + 0 = 6` — the same slot. So yes: under the buggy formula, `(2,3)` and `(3,3)` alias to index 6, exactly the third collision identified in Worked Example 9.4.2's trace of the `4×4` test.

**5.** Counting "elements in all rows before row `r`" gives a *different* formula for the two cases because the two triangles grow in opposite directions as `row` increases. In a lower triangular matrix, row `r` stores `r+1` elements (columns `0` through `r`) — row count grows as `row` grows, so the running total before row `r` is the increasing sum `0+1+2+...+r = r(r+1)/2`, a clean formula with no reference to the matrix's overall `size` at all. In an upper triangular matrix, row `r` stores `size - r` elements (columns `r` through `size-1`) — row count *shrinks* as `row` grows, so the running total before row `r` is `size + (size-1) + ... + (size-r+1)`, a sum that does depend on `size`, and is correspondingly easier to get a sign wrong in (as the actual code does). The lower-triangular formula's independence from `size` is exactly what makes it simpler to write correctly, and the upper-triangular formula's dependence on `size` is exactly where the bug traced in this chapter lives.
