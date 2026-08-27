# Chapter 7: Memory Layout Design — Strides, Views, Alignment, and Broadcasting

> "Chapter 6 gave the tensor a shape and a stride formula it trusted blindly, in one fixed layout. This chapter asks the three questions a shape alone can't answer: is this memory laid out the way you think it is, can you look at *part* of it without copying *all* of it, and what happens when two tensors of different shapes need to meet in the same arithmetic operation?"

**What you will understand by the end of this chapter:**

- Strides for row-major *and* column-major layouts, computed by hand from a general formula that Chapter 6 only showed you in its row-major special case, plus the index↔offset conversion in both directions — and the one direction that silently breaks for non-row-major strides
- `TensorView`: a zero-copy slice, transpose, or reshape that reads and writes through an existing tensor's buffer, built on the `RefCount` and `_owns_data` machinery Chapter 6 introduced but never used
- Alignment and padding: why a tensor's *logical* shape and its *physical* footprint in memory can differ, and how that padding connects back to Chapter 5's SIMD widths and Chapter 4's GPU warp size
- Broadcasting: the exact rule two shapes must satisfy to be combined in one operation, why the result is computed with stride-0 dimensions instead of a real copy, and a close reading of where this book's own broadcasting code quietly does less than its comments claim

**What you need to know first:**

- Chapter 6 (`TensorShape`, `TensorStrides`, and the row-major stride formula `stride[ndim-1]=1`, `stride[i]=stride[i+1]×shape[i+1]`)
- Chapter 3 (the memory-bus / bus-utilization argument, and the AoS-vs-SoA reasoning built on it)
- Chapter 4 (memory coalescing, warp size, and the broadcasting example from Section 4.5)
- Chapter 5 (SIMD width and the "main loop plus remainder" shape)

<a id="part-121-stride-calculation-system"></a>
## 7.1 Strides Revisited: One Formula, Many Layouts `[FOUNDATIONAL]`

### Intuition

Picture two ways of numbering apartments in a building with 3 floors and 4 units per floor. The first scheme numbers unit-by-unit within a floor before moving to the next floor: floor 0's units are 0, 1, 2, 3; floor 1's are 4, 5, 6, 7; and so on — the *unit number* varies fastest. The second scheme numbers floor-by-floor within a unit position before moving to the next unit: unit-position 0 across all floors is numbered 0, 1, 2; unit-position 1 across all floors is 3, 4, 5; and so on — the *floor number* varies fastest.

Both schemes describe the exact same building. Neither is "more correct." But if you're handed a number and asked "which apartment is this," you need to know which scheme produced it — and that's exactly what a stride array records. The first scheme is what this book has been calling row-major (C-style): the *rightmost* index varies fastest. The second is column-major (Fortran-style, and the convention LAPACK and many numerical linear-algebra libraries still use): the *leftmost* index varies fastest. Chapter 6 only ever showed you the first one. This section generalizes the formula so it can produce either.

### Background

```mojo
struct MemoryLayout(Copyable, Movable):
    alias ROW_MAJOR = 0
    alias COLUMN_MAJOR = 1
    alias CUSTOM = 2
    var layout_type: Int
```

`StrideInfo` wraps a shape, a layout, and the strides that layout implies, plus a cached `is_contiguous` flag and the total byte count — the same "compute once, cache forever" discipline Chapter 6's `TensorShape._size` established:

```mojo
struct StrideInfo:
    var strides: UnsafePointer[Int]
    var shape: UnsafePointer[Int]
    var ndim: Int
    var layout: MemoryLayout
    var is_contiguous: Bool
    var element_size: Int
    var total_bytes: Int
```

The two stride algorithms are mirror images of each other:

| Layout | Fastest-varying index | Stride formula | `[2,3,4]` example |
|---|---|---|---|
| Row-major (C) | rightmost (last) | `strides[ndim-1]=1`; for `i` from `ndim-2` down to `0`: `strides[i]=strides[i+1]×shape[i+1]` | `[12, 4, 1]` |
| Column-major (Fortran) | leftmost (first) | `strides[0]=1`; for `i` from `1` to `ndim-1`: `strides[i]=strides[i-1]×shape[i-1]` | `[1, 2, 6]` |

Contiguity is checked the same way in reverse for each layout: row-major expects each stride to equal the product of every *later* dimension's size (checked scanning right-to-left); column-major expects each stride to equal the product of every *earlier* dimension's size (checked scanning left-to-right). A custom stride pattern is never considered contiguous by this code, even if it happens to match one of the two formulas above — `is_stride_pattern_contiguous` only checks the `ROW_MAJOR` and `COLUMN_MAJOR` branches.

Two free functions convert between a coordinate list and a flat memory offset:

```mojo
fn index_to_offset(indices: List[Int], strides: List[Int]) -> Int:
    var offset = 0
    for i in range(len(indices)):
        offset += indices[i] * strides[i]
    return offset
```

This is the same dot-product Chapter 6.2 already used. Its inverse, `offset_to_indices`, is where this chapter's first `[COMMON TRAP]` lives — see below.

### Worked Example 7.1.1 — Both layouts, two shapes

For shape `[3, 4]`: row-major gives `strides[1]=1`, `strides[0]=strides[1]×shape[1]=1×4=4` → `[4, 1]`. Column-major gives `strides[0]=1`, `strides[1]=strides[0]×shape[0]=1×3=3` → `[1, 3]`.

For shape `[2, 3, 4]` (the shape Chapter 6 used throughout): row-major gives the familiar `[12, 4, 1]`. Column-major works forward instead of backward: `strides[0]=1`; `strides[1]=strides[0]×shape[0]=1×2=2`; `strides[2]=strides[1]×shape[1]=2×3=6` → `[1, 2, 6]`. Both results match this section's own test output exactly.

### Worked Example 7.1.2 — Index↔offset round trip

Using shape `[3,4]`'s row-major strides `[4,1]`: index `[1,2]` → offset `1×4 + 2×1 = 6`. Index `[2,3]` → offset `2×4 + 3×1 = 11`. Running these two offsets back through `offset_to_indices` recovers the original coordinates: for offset `11`, `indices[0] = 11 // 4 = 2`, remainder `11 % 4 = 3`; `indices[1] = 3 // 1 = 3`, remainder `0` → `[2, 3]`. Correct.

```
[COMMON TRAP]  offset_to_indices only inverts row-major (descending-stride) layouts

offset_to_indices decodes a flat offset the same way you'd decode a mixed-radix
number: divide by the largest "place value" first, keep the remainder, divide by
the next place value, and so on. That only produces a valid coordinate when the
strides are visited in DECREASING order — which row-major strides always are
(stride[0] is the largest, stride[ndim-1]=1 is the smallest).

Feed it column-major strides instead and it silently produces nonsense. Take
shape [3,4], column-major strides [1,3], and a real, in-bounds offset of 7
(valid: index [1,2] gives 1×1 + 2×3 = 7). The function computes:

    indices[0] = 7 // strides[0] = 7 // 1 = 7      <-- shape[0] is only 3!
    remainder  = 7 % 1 = 0
    indices[1] = 0 // strides[1] = 0 // 3 = 0

Result: [7, 0] — an index whose first coordinate is more than double the
dimension's actual size, silently returned with no bounds check and no error.
The function's docstring calls it "the inverse operation of index_to_offset,"
which is only true for the one stride ordering it was written against.
```

### Worked Example 7.1.3 — `StridedTensor`'s memory dump, row-major vs. column-major

`StridedTensor` composes `StrideInfo` with a data buffer, the same composition pattern `Tensor[dtype]` used in Chapter 6.3. Build two `[2,3]` tensors — one row-major, one column-major — and set the same four logical values into each: `tensor[0,0]=1.0`, `tensor[0,1]=2.0`, `tensor[1,0]=3.0`, `tensor[1,1]=4.0` (leaving `[1,2]` untouched at its zero-initialized value in both).

Row-major strides are `[3,1]`. `[0,0]→offset 0`, `[0,1]→offset 1`, `[1,0]→offset 3`, `[1,1]→offset 4`. Column-major strides are `[1,2]`. `[0,0]→offset 0`, `[0,1]→offset 2`, `[1,0]→offset 1`, `[1,1]→offset 3`. Laying out six memory slots for each:

```
Row-major (strides [3,1]):        Column-major (strides [1,2]):
 data[0] = 1.0   ([0,0])           data[0] = 1.0   ([0,0])
 data[1] = 2.0   ([0,1])           data[1] = 3.0   ([1,0])
 data[2] = 0.0   (never set)       data[2] = 2.0   ([0,1])
 data[3] = 3.0   ([1,0])           data[3] = 4.0   ([1,1])
 data[4] = 4.0   ([1,1])           data[4] = 0.0   (never set)
 data[5] = 0.0   (never set)       data[5] = 0.0   (never set)
```

Same four logical values, same tensor shape, two completely different physical byte sequences — the entire point of this section: shape describes *meaning*, strides describe *memory*.

<a id="part-122-view-and-slicing-infrastructure"></a>
## 7.2 Views: Reading a Tensor Without Copying It `[FOUNDATIONAL]`

### Intuition

Chapter 3's bus-utilization argument was about bytes *moved*: every byte that crosses the memory bus costs bandwidth, whether or not the computation needed it. A **view** takes that argument to its logical extreme — it moves *zero* bytes. Instead of copying a slice of a tensor into a new buffer, a view is a small record of *where* to look in the existing buffer: a starting offset, a shape, and a set of strides that might not match the original tensor's own strides at all. Reading through a view costs exactly the same as reading the original data, because it *is* the original data, examined through a different lens.

Chapter 6.3 declared a field for exactly this — `_owns_data: Bool` — but every struct in that chapter always set it to `True`. This section is where `_owns_data = False` finally does something.

### Background

A slice needs three numbers per dimension — start, end, step — plus normalization logic for Python-style negative indices and open-ended bounds (`SliceSpec`, `MultiSliceSpec`). Safe sharing of one buffer across multiple views needs a way to know when the last view is gone (`RefCount`, a hand-rolled reference counter: its `__copyinit__` increments a shared counter, its `__del__` decrements it). And the view itself needs the data pointer, its own shape and strides, and an offset into the shared buffer:

```mojo
struct TensorView[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var offset: Int
    var ndim: Int
    var ref_count: RefCount
    var is_contiguous: Bool
```

Slicing computes a *new* offset and stride pair without touching `data` at all:

```mojo
fn slice(self, slice_specs: MultiSliceSpec) -> TensorView[dtype]:
    ...
    for i in range(self.ndim):
        var spec = normalized_specs.get_slice(i)
        if spec.is_valid:
            new_shape.append(spec.length())
            new_strides.append(self.strides[i] * spec.step)
            new_offset += spec.start * self.strides[i]
    return TensorView[dtype](self.data, new_shape, new_strides, new_offset)
```

`transpose` swaps two dimensions' shape *and* stride entries — no data moves, only the map describing how to read it changes. `reshape` is more restrictive: it only succeeds when the view is already contiguous, because a reshape assumes it can lay new strides out fresh in row-major order, which only makes sense if the existing data really is one unbroken run.

### Worked Example 7.2.1 — Slice normalization

`SliceSpec` normalizes Python-style slice syntax against a concrete dimension size. For a dimension of size `10`:

`[1:5:1]` needs no negative-index handling: `start=1, end=5, length=(5-1+1-1)//1=4`. `[::2]` (both bounds unspecified) resolves to the full range: `start=0, end=10, length=(10-0+2-1)//2=5`. `[-3:]` resolves the negative start: `-3+10=7`, then `start=7, end=10, length=(10-7+1-1)//1=3`. All three match this section's test output exactly, and `get_indices()` walking `[1:5:1]` produces `[1,2,3,4]`; walking `[::2]` produces `[0,2,4,6,8]`.

### Worked Example 7.2.2 — A row slice and a column slice of the same tensor

Take a `[4,5]` tensor filled with `tensor[i,j] = i×5+j` (row-major strides `[5,1]`, so this is literally the flat sequence `0..19`). Slicing rows `[1:3, :]` normalizes dim 0 to `start=1,end=3,length=2` and dim 1 to the full range `length=5`; the new offset is `1×5 = 5`, and the new strides stay `[5,1]` (still contiguous — a whole-row slice never breaks contiguity). Reading `slice[0,0]` through `slice[1,4]` recovers `5.0` through `14.0`, exactly the original rows 1 and 2.

Slicing columns `[:, 1:4:2]` (columns 1 and 3, skipping every other) instead: dim 0 stays full (`length=4`), dim 1 normalizes to `start=1,end=4,step=2,length=(4-1+2-1)//2=2`. New offset is `1×1=1` (dim 1's contribution only — dim 0's start is `0`). New strides are `[5×1, 1×2] = [5,2]` — no longer contiguous, since consecutive elements of this view are two floats apart in memory, not one. Reading it back gives `slice[i,j] = tensor[i, 1+2j]`: `slice[1,0]=tensor[1,1]=6.0`, `slice[1,1]=tensor[1,3]=8.0`, and so on through row 3.

```
[COMMON TRAP]  the source's own captured output disagrees with the hand-derived
value at exactly one cell: slice[0,0]

Every other cell in the column-slice trace above matches the by-hand derivation
exactly (slice[0,1]=3.0, slice[1,0]=6.0, slice[1,1]=8.0, ... slice[3,1]=18.0).
slice[0,0], by the same formula, should read tensor[0,1] = 1.0. The book's own
recorded run instead prints a garbage float (6.209e-42 — a classic bit pattern
for uninitialized or stale memory). This isn't a mistake in the derivation above;
it's a real, reproducible anomaly in the demo's own output, of the same kind
seen in Worked Example 7.2.3 below. The lesson isn't "the math is wrong
somewhere" — it's that pointer-based code can look completely correct on paper
and still read a stale or uninitialized byte at a specific cell, and the only
way to know for certain is to run it and compare against your hand trace, cell
by cell, not just spot-check a few.
```

### Worked Example 7.2.3 — When "should be filled" isn't "is filled"

A related anomaly shows up even earlier, in the chapter's very first view demo. A `[3,4]` tensor is filled with `tensor[i,j] = i×4+j+1` (so `[0,0]=1.0`, `[0,1]=2.0`, `[1,0]=5.0`, `[1,1]=6.0`), then a full view is created and read back. The recorded output for `view[0,0]`, `view[0,1]`, `view[1,0]`, `view[1,1]` is `1.7510376`, `6.209e-42`, `7.977031e+17`, `9.550975e-06` — none of which are the filled values at all. Whatever the exact cause (this book won't reverse-engineer it here), the practical discipline this section is teaching survives the anomaly either way: derive the expected value by hand from the shape, strides, and offset; then check the *actual* value the program reports; and treat any mismatch as a lead to chase, never as a typo in your own arithmetic until you've ruled out the code.

### Worked Example 7.2.4 — Transpose and reshape change the map, not the memory

A `[2,3]` tensor filled with `tensor[i,j] = i×10+j` has row-major strides `[3,1]`. `transpose(0,1)` swaps both the shape and stride entries for axes 0 and 1, producing a view of shape `[3,2]` with strides `[1,3]` — no longer contiguous, since a step along the new axis 0 (stride `1`) is smaller than a step along the new axis 1 (stride `3`). Reading it back: `transposed[0,1] = tensor[1,0] = 10.0`; `transposed[2,1] = tensor[1,2] = 12.0` — every value is exactly the original tensor read with its two coordinates swapped, at zero data-movement cost.

`reshape([6])` on the *original* (contiguous) view succeeds and produces a flat `[6]` view with stride `[1]`, reading out the tensor in row-major order: `0.0, 0.0(from [0,2]... wait — actually [0,2] was never set, stays 0), 2.0, 10.0, 11.0, 12.0` reflects the same underlying buffer, one dimension collapsed into one. Had you tried to reshape the *transposed* (non-contiguous) view instead, `reshape` would refuse and hand back the original view unchanged — the `if ... or not self.is_contiguous: return self` guard exists precisely because a reshape's row-major stride assumption only holds for contiguous memory.

### Worked Example 7.2.5 — A three-way slice, and mutation through a view

For a `[3,4,2]` tensor filled with `tensor[i,j,k] = i×100+j×10+k`, a compound slice `[1:3, 0:4:2, :]` (rows 1–2, every other column, all of the last axis) produces a view of shape `[2,2,2]`, strides `[8,4,1]`, offset `8` (from `1×stride[0]=1×8=8` — this tensor's row-major strides are `[8,2,1]`... wait, more precisely: original strides for `[3,4,2]` are `[8,2,1]`; the column step of `2` scales that middle stride to `2×2=4`, giving the view's final `[8,4,1]`). Reading it back recovers `slice[0,0,0]=tensor[1,0,0]=100.0`, `slice[1,1,1]=tensor[2,2,1]=221.0`, and every value in between, matching the source's own test output cell for cell.

Finally, the point of all of this: a view sharing a buffer with its parent means a write through the view *is* a write to the parent. Slicing the first row of a `[2,3]` tensor and setting `slice_view[0,1] = 99.0` changes `tensor[0,1]` too — checking `tensor.get_item([0,1])` afterward reports `99.0`, not the original `1.0`. Chapter 3's argument said moving fewer bytes is faster; this section's argument is that a view doesn't just move fewer bytes, it doesn't create a second copy of the *truth* either — there's exactly one buffer, and every view is a different way of addressing the same one.

## 7.3 Alignment and Padding: What SIMD and the GPU Actually Want `[FOUNDATIONAL]`

### Intuition

Chapter 5 established that a SIMD register processes a fixed width of elements at once (4, 8, or 16 lanes), and Chapter 4 established that a GPU warp moves a fixed number of threads' memory requests together (32 threads, ideally hitting one contiguous run of bytes). Both hardware features share an assumption this chapter's earlier sections never had to think about: that a buffer *starts* and is *organized* on a boundary the hardware expects — 16, 32, or 64 bytes for SIMD register widths; 128 bytes for a GPU's coalesced-access granularity. A tensor whose innermost dimension doesn't divide evenly into that width leaves the hardware two choices: process a partial, wasted vector at the end (exactly Chapter 5.2's remainder problem), or the tensor library pads the buffer up front so every vector load is full. This section is about the second option.

### Background

`AlignmentSpec` names the common targets:

```mojo
struct AlignmentSpec(Copyable, Movable):
    alias ALIGN_SIMD_128 = 16    # SSE
    alias ALIGN_SIMD_256 = 32    # AVX
    alias ALIGN_SIMD_512 = 64    # AVX-512
    alias ALIGN_CACHE_LINE = 64
    alias ALIGN_GPU_COALESCE = 128
```

Padding is a straightforward remainder calculation: `padding = alignment - (size % alignment)`, or `0` when `size` already divides evenly. `get_optimal_alignment_for_arch` picks a default per target: CPU gets `32` bytes (AVX), GPU gets `128` bytes (coalescing granularity), Mixed workloads get `64` bytes (a plain cache line, splitting the difference).

`SIMDTensorLayout` applies this at the *dimension* level rather than the whole-buffer level: only the innermost dimension gets padded to a SIMD-width multiple (in elements, not bytes, here — `simd_width=8` means 8 *elements*), because that's the dimension a vectorized loop actually strides across.

### Worked Example 7.3.1 — Padding a handful of sizes to 32-byte alignment

With the CPU default of 32 bytes: size `100` → `100 % 32 = 4`, padding `= 32-4 = 28`, aligned size `128`. Size `127` → `127 % 32 = 31`, padding `1`, aligned `128`. Size `200` → `200 % 32 = 8`, padding `24`, aligned `224`. Size `255` → `255 % 32 = 31`, padding `1`, aligned `256`. All four match this section's own printed table exactly.

### Worked Example 7.3.2 — SIMD-padding a `[3, 7, 11]` tensor at width 8

Only the innermost dimension (size `11`) gets padded: `11 % 8 = 3`, so it pads up to `11 + (8-3) = 16`, a padding of `5` elements. The aligned stride for that dimension stays `1` (unit stride is unaffected by padding); the running stride for dimension 1 becomes `16` (the *padded* size of dimension 2, not the logical `11`); dimension 0's stride becomes `16×7=112`. Total padded size: `112×3=336` elements. Memory efficiency is useful elements over padded elements: `(3×7×11)/(3×7×16) = 231/336 = 0.6875 = 68.75%` — over 31% of the allocated buffer is padding that exists purely so the innermost loop's SIMD vectors never run off the end of real data. This is the same trade Chapter 5.2 flagged from the other direction: pad the *buffer* (this chapter) so the *loop* never needs a remainder branch (Chapter 5).

```
[COMMON TRAP]  the alignment label you request and the padding you actually get
can be two unrelated numbers

create_cpu_optimized_tensor, create_gpu_optimized_tensor, and
create_custom_aligned_tensor request byte alignments of 32, 128, and (say) 128
bytes respectively — three different AlignmentSpec values, printed as three
different "Target" and "Alignment" labels. But AlignedTensor's constructor
builds its SIMDTensorLayout with a hardcoded simd_width of 8 elements,
regardless of which AlignmentSpec was passed in:

    self.simd_layout = SIMDTensorLayout(shape, 8, alignment_spec)

For a [2,3] tensor, all three variants pad the same way (innermost dim 3 pads
to 8, same as Worked Example 7.3.2's logic at smaller scale) and report the
identical "Padding: 40 bytes" and "Memory efficiency: 37.5%" — because the
number that actually determines the padding (simd_width=8) never changed. The
AlignmentSpec value is real, correctly computed, and printed — it's just
descriptive metadata here, not an input the padding math uses.
```

### Worked Example 7.3.3 — A coalescing check that only checks one thing

`GPUCoalescingOptimizer.analyze_coalescing_efficiency` computes efficiency as `ideal_bytes / bytes_per_warp`, where `bytes_per_warp = element_size × innermost_stride × warp_size`. For shape `[4,32]` with strides `[32,1]` (innermost stride `1`, warp size `32`, element size `4`): `bytes_per_warp = 4×1×32=128`, `ideal_bytes=4×32=128`, efficiency `= 128/128 = 100%`. For shape `[4,17]` with strides `[17,1]` — a shape whose innermost *size* (`17`) does **not** divide evenly into the warp size (`32`) — the same formula gives exactly the same answer: innermost *stride* is still `1`, so `bytes_per_warp` is still `4×1×32=128`, and efficiency is still `100%`.

```
[COMMON TRAP]  a metric that checks unit stride, not warp alignment, agreeing
with a comment that claims otherwise

The test data for shape [4,17] is commented "# Not warp-aligned" — true of the
shape (17 doesn't divide 32) but not something analyze_coalescing_efficiency
actually detects; it only checks whether the innermost stride is 1, which it is
in both cases, so both report 100% efficient. The one function that *does*
notice the size mismatch, recommend_optimizations, checks innermost_size <
warp_size — but only inside a branch gated on efficiency < coalescing_threshold
(0.8). Since the efficiency check above already reported 100%, that branch
never runs, and recommend_optimizations for the [4,17] case returns an empty
list — no recommendations printed at all, despite the shape genuinely being
sub-optimal for a real GPU. Reading the comment told you the intent; reading
the actual arithmetic is what tells you whether the code delivers it.
```

<a id="part-125-broadcasting-layout-preparation"></a>
## 7.4 Broadcasting: Combining Shapes That Don't Match `[FOUNDATIONAL]`

### Intuition

Chapter 4.5 broadcast a `[2]` bias vector across a `[2,2]` matrix without ever writing a `[2,2]`-sized bias buffer — every thread just read the same `2`-element array, indexed by its column only. Broadcasting is that trick generalized into a rule: two shapes can be combined element-wise whenever, dimension by dimension (aligned from the *right*, the same direction Chapter 6's row-major strides grow from), the sizes either match exactly or one of them is `1`. Wherever a `1` meets a larger size, that dimension is *virtually* stretched — every logical position along it reads the same single physical value, which is implemented, not by copying that value into a full-size buffer, but by giving that dimension a stride of `0`. A stride-`0` dimension means "moving along this axis doesn't change the memory address at all" — the cheapest possible way to make one value answer for many logical positions.

Note: an earlier, less general `calculate_broadcast_strides` appears in Section 7.1's own source file, comparing two tensors' existing strides directly. This section's version is the one this book treats as canonical — it works from a target *shape* rather than a second tensor's strides, and is the version NumPy-style broadcasting is normally described by.

### Background

```mojo
struct BroadcastRule(Copyable, Movable):
    alias INCOMPATIBLE = 0
    alias COMPATIBLE = 1
    alias IDENTICAL = 2
    alias BROADCAST_LEFT = 3
    alias BROADCAST_RIGHT = 4
    alias BROADCAST_BOTH = 5
```

`analyze_broadcast_compatibility` walks both shapes from the rightmost dimension, treating a missing dimension (when one shape has fewer axes than the other) as size `1`. Any pair of dimensions that differ *and* neither equals `1` makes the whole shape pair `INCOMPATIBLE`. Otherwise the result shape takes the larger of each dimension pair, and the rule records which side(s) needed broadcasting.

`calculate_broadcast_strides` then converts an original shape's own strides into broadcast strides against a target shape: matching dimensions keep their original stride; a size-`1` dimension broadcasting up gets stride `0`; a dimension missing entirely (the shape has fewer axes than the target) also gets stride `0`.

### Worked Example 7.4.1 — Three compatibility checks

`[3,4]` and `[4]`: aligning from the right, position 0 compares `4` vs `4` (equal); position 1 compares `3` vs the *missing* second dimension of `[4]`, treated as `1` — since the missing side is `1`, this is `BROADCAST_RIGHT`. Result shape: `[3,4]`.

`[3,4]` and `[1]` (a scalar-shaped tensor): position 0 compares `4` vs `1` — `1` broadcasts, right side again; position 1 compares `3` vs the missing dimension (`1`) — right again. Still only the right side ever needs broadcasting, so this is also `BROADCAST_RIGHT`.

`[3,4]` and `[3,5]`: position 0 compares `4` vs `5` — neither is `1`, and they don't match. `INCOMPATIBLE`, immediately, with no result shape computed.

### Worked Example 7.4.2 — Computing the actual broadcast strides

Original shape `[1,4]` with strides `[4,1]`, broadcasting to target shape `[3,4]`: aligning from the right, position 0 (target's last dim, `4`) matches the original's last dim (`4`) exactly, so it keeps the original stride, `1`. Position 1 (target's first dim, `3`) compares against the original's first dim, `1` — a broadcast, so the stride becomes `0`. Reversing back to normal order: `[0, 1]` — moving along axis 0 of the broadcast result never advances the pointer at all (every "row" reads the same 4 values); moving along axis 1 behaves exactly as it did originally.

### Worked Example 7.4.3 — A `BROADCAST_BOTH` case, traced all the way to a cost estimate

Shapes `[2,1,4]` and `[3,4]`. Aligning from the right: position 0, `4` vs `4`, equal. Position 1, `1` (shape A's middle dim) vs `3` (shape B's first dim) — A's side is `1`, so the left tensor broadcasts here. Position 2, `2` (shape A's first dim) vs the missing dimension of B (treated as `1`) — B's side is `1` here, so the right tensor broadcasts. Both sides needed broadcasting somewhere → `BROADCAST_BOTH`, result shape `[2,3,4]` (the element-wise maximum at each aligned position, reversed back to normal order).

Feeding this result into `create_broadcast_spec`'s cost estimate produces a chain of numbers that only makes sense once you notice what the factory function *doesn't* do:

```
[COMMON TRAP]  a cost formula with unreachable branches, because its factory
never populates the field they depend on

BroadcastSpec.calculate_memory_cost adds a penalty for every *registered*
input tensor smaller than the result: `memory_cost += result_elements // 4`
per small input. get_optimization_hints similarly inspects `self.input_shapes`
to decide between "scalar_broadcast", "single_broadcast", and "multi_broadcast".
Both read from the spec's `input_shapes` list — which is only ever populated by
calling `spec.add_input(shape, strides)`.

create_broadcast_spec(shape_a, shape_b), the convenience factory used in every
demo in this section, never calls add_input. It only sets the compatibility
rule and the result shape, then calls calculate_memory_cost and
get_optimization_hints immediately — against an empty input_shapes list.

The visible consequence, for [2,1,4] + [3,4] → result [2,3,4] (24 elements):
memory_cost's per-input loop never executes, so memory_cost stays exactly
result_elements = 24 — printed as "Memory cost: 24 elements", which looks like
a real number but is actually just the base cost with zero penalty terms
applied. get_optimization_hints' loop is equally empty, so total_broadcasts
stays 0 (never reaching its "== 1" branch) and has_scalar_broadcast stays
False — falling through to the catch-all "multi_broadcast", printed as the
"Optimization hint," regardless of what the actual input shapes were.

BroadcastOptimizer.analyze_broadcast_efficiency does apply one real penalty —
a flat ×0.8 whenever requires_broadcasting() is true — giving 80.0% here. But
its second penalty loop, over spec.input_shapes, is unreachable for the same
reason, so the final numbers (80.0% efficiency, "tiled" strategy, an estimated
cost of `24 + Int(24×0.2) = 24+4 = 28` units) are all real arithmetic, faithfully
computed — just from a spec that's missing the very inputs its own docstrings
describe using.
```

### Worked Example 7.4.4 — `BroadcastTensor` in practice

A `[2,3]` `BroadcastTensor` checking `can_broadcast_with([4,2,3])`: aligning from the right, `3` vs `3` (equal), `2` vs `2` (equal), then the tensor's *missing* third dimension (treated as `1`) vs `4` — the tensor's side is `1`, so it broadcasts, and the answer is `Yes`, with result shape `[4,2,3]`. A `[1]`-shaped scalar tensor checking against `[2,3]` similarly broadcasts on both remaining dimensions and also answers `Yes`. Once `prepare_for_broadcast` has been called, `get_item`/`set_item` switch from the tensor's own `strides` array to its `broadcast_strides` array — the same stride-0 mechanism traced in Worked Example 7.4.2, now wired into actual element access.

## 7.5 Complete Runnable Code

The four sections above are drawn from four independent, runnable Mojo files. Each is reproduced here in full, exactly as written, together with its own recorded output.

### File: `37_stride_calculation.mojo` — Section 7.1

**Run:** `pixi run mojo 37_stride_calculation.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias DEFAULT_ALIGNMENT = 64  # 64-byte alignment for SIMD operations

struct MemoryLayout(Copyable, Movable):
    """Memory layout specification for tensor stride calculation."""
    alias ROW_MAJOR = 0     # C-style: rightmost dimension varies fastest
    alias COLUMN_MAJOR = 1  # Fortran-style: leftmost dimension varies fastest
    alias CUSTOM = 2        # User-defined stride pattern

    var layout_type: Int

    fn __init__(out self, layout_type: Int = Self.ROW_MAJOR):
        self.layout_type = layout_type

    fn is_row_major(self) -> Bool:
        return self.layout_type == Self.ROW_MAJOR

    fn is_column_major(self) -> Bool:
        return self.layout_type == Self.COLUMN_MAJOR

    fn is_custom(self) -> Bool:
        return self.layout_type == Self.CUSTOM

    fn name(self) -> String:
        if self.layout_type == Self.ROW_MAJOR:
            return "ROW_MAJOR"
        elif self.layout_type == Self.COLUMN_MAJOR:
            return "COLUMN_MAJOR"
        else:
            return "CUSTOM"

struct StrideInfo:
    """Comprehensive stride information for tensors."""
    var strides: UnsafePointer[Int]
    var shape: UnsafePointer[Int]
    var ndim: Int
    var layout: MemoryLayout
    var is_contiguous: Bool
    var element_size: Int
    var total_bytes: Int

    fn __init__(out self, shape: List[Int], layout: MemoryLayout = MemoryLayout(), element_size: Int = 4):
        self.ndim = len(shape)
        self.layout = layout
        self.element_size = element_size
        self.is_contiguous = False  # Initialize before using in methods

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            total_elements *= shape[i]

        self.total_bytes = total_elements * element_size

        if layout.is_row_major():
            self._compute_row_major_strides()
        elif layout.is_column_major():
            self._compute_column_major_strides()
        else:
            self._compute_row_major_strides()

        self.is_contiguous = self._check_contiguity()

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.layout = existing.layout
        self.is_contiguous = existing.is_contiguous
        self.element_size = existing.element_size
        self.total_bytes = existing.total_bytes

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.strides[i] = existing.strides[i]

    fn __del__(owned self):
        self.shape.free()
        self.strides.free()

    fn _compute_row_major_strides(self):
        if self.ndim == 0:
            return
        self.strides[self.ndim - 1] = 1
        for i in range(self.ndim - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.shape[i + 1]

    fn _compute_column_major_strides(self):
        if self.ndim == 0:
            return
        self.strides[0] = 1
        for i in range(1, self.ndim):
            self.strides[i] = self.strides[i - 1] * self.shape[i - 1]

    fn _check_contiguity(self) -> Bool:
        if self.ndim <= 1:
            return True

        if self.layout.is_row_major():
            var expected_stride = 1
            for i in range(self.ndim - 1, -1, -1):
                if self.strides[i] != expected_stride:
                    return False
                expected_stride *= self.shape[i]
            return True

        elif self.layout.is_column_major():
            var expected_stride = 1
            for i in range(self.ndim):
                if self.strides[i] != expected_stride:
                    return False
                expected_stride *= self.shape[i]
            return True

        return False

    fn get_stride(self, axis: Int) -> Int:
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.strides[axis]

    fn get_shape(self, axis: Int) -> Int:
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.shape[axis]

fn compute_strides(shape: List[Int], layout: MemoryLayout = MemoryLayout()) -> List[Int]:
    """
    Example:
        shape=[2, 3, 4], row-major -> strides=[12, 4, 1].
        shape=[2, 3, 4], column-major -> strides=[1, 2, 6].
    """
    var ndim = len(shape)
    var strides = List[Int]()

    if ndim == 0:
        return strides

    for _ in range(ndim):
        strides.append(0)

    if layout.is_row_major():
        strides[ndim - 1] = 1
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

    elif layout.is_column_major():
        strides[0] = 1
        for i in range(1, ndim):
            strides[i] = strides[i - 1] * shape[i - 1]

    return strides

fn compute_custom_strides(shape: List[Int], custom_strides: List[Int]) -> Bool:
    if len(shape) != len(custom_strides):
        return False

    for i in range(len(custom_strides)):
        if custom_strides[i] <= 0:
            return False

    return True

fn index_to_offset(indices: List[Int], strides: List[Int]) -> Int:
    """Formula: offset = sum(index[i] * stride[i] for i in range(ndim))."""
    if len(indices) != len(strides):
        return -1  # Error indicator

    var offset = 0
    for i in range(len(indices)):
        offset += indices[i] * strides[i]

    return offset

fn offset_to_indices(offset: Int, shape: List[Int], strides: List[Int]) -> List[Int]:
    """Note: This is the inverse operation of index_to_offset."""
    var indices = List[Int]()
    var remaining_offset = offset

    if len(shape) != len(strides):
        return indices

    var ndim = len(shape)
    for _ in range(ndim):
        indices.append(0)

    for i in range(ndim):
        if strides[i] > 0:
            indices[i] = remaining_offset // strides[i]
            remaining_offset = remaining_offset % strides[i]

    return indices

fn calculate_broadcast_strides(shape_a: List[Int], shape_b: List[Int],
                             strides_a: List[Int], strides_b: List[Int]) -> List[Int]:
    var len_a = len(shape_a)
    var len_b = len(shape_b)
    var max_len = len_a if len_a > len_b else len_b

    var result_strides = List[Int]()

    for i in range(max_len):
        var dim_a = 1
        var dim_b = 1
        var stride_a = 0
        var stride_b = 0

        if i < len_a:
            dim_a = shape_a[len_a - 1 - i]
            stride_a = strides_a[len_a - 1 - i]
        if i < len_b:
            dim_b = shape_b[len_b - 1 - i]
            stride_b = strides_b[len_b - 1 - i]

        var result_stride = 0
        if dim_a == dim_b:
            result_stride = stride_a if stride_a > stride_b else stride_b
        elif dim_a == 1:
            result_stride = stride_b
        elif dim_b == 1:
            result_stride = stride_a

        result_strides.append(result_stride)

    var final_strides = List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        final_strides.append(result_strides[i])

    return final_strides

fn is_stride_pattern_contiguous(shape: List[Int], strides: List[Int], layout: MemoryLayout) -> Bool:
    if len(shape) != len(strides):
        return False

    if len(shape) <= 1:
        return True

    if layout.is_row_major():
        var expected_stride = 1
        for i in range(len(shape) - 1, -1, -1):
            if strides[i] != expected_stride:
                return False
            expected_stride *= shape[i]
        return True

    elif layout.is_column_major():
        var expected_stride = 1
        for i in range(len(shape)):
            if strides[i] != expected_stride:
                return False
            expected_stride *= shape[i]
        return True

    return False

fn optimize_stride_pattern(shape: List[Int], access_pattern: List[Int]) -> List[Int]:
    var ndim = len(shape)
    var optimized_strides = List[Int]()

    if ndim == 0:
        return optimized_strides

    return compute_strides(shape, MemoryLayout(MemoryLayout.ROW_MAJOR))

struct StridedTensor[dtype: DType]:
    """
    Enhanced tensor with comprehensive stride support.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var stride_info: StrideInfo
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int], layout: MemoryLayout = MemoryLayout()) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")

        self.stride_info = StrideInfo(shape, layout, 4)  # Assume 4 bytes for now
        self._owns_data = True

        var total_elements = 1
        for i in range(len(shape)):
            total_elements *= shape[i]

        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)

        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)

    fn __copyinit__(out self, existing: Self):
        self.stride_info = existing.stride_info
        self._owns_data = True

        var total_elements = 1
        for i in range(self.stride_info.ndim):
            total_elements *= self.stride_info.get_shape(i)

        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)

        for i in range(total_elements):
            self.data[i] = existing.data[i]

    fn __del__(owned self):
        if self._owns_data:
            self.data.free()

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var strides = List[Int]()
        for i in range(self.stride_info.ndim):
            strides.append(self.stride_info.get_stride(i))

        var offset = index_to_offset(indices, strides)
        return self.data[offset]

    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        var strides = List[Int]()
        for i in range(self.stride_info.ndim):
            strides.append(self.stride_info.get_stride(i))

        var offset = index_to_offset(indices, strides)
        self.data[offset] = value

    fn get_layout(self) -> MemoryLayout:
        return self.stride_info.layout

    fn is_contiguous(self) -> Bool:
        return self.stride_info.is_contiguous

    fn numel(self) -> Int:
        var total = 1
        for i in range(self.stride_info.ndim):
            total *= self.stride_info.get_shape(i)
        return total

    fn ndim(self) -> Int:
        return self.stride_info.ndim

    fn fill(self, value: Scalar[dtype]):
        var total_elements = self.numel()
        for i in range(total_elements):
            self.data[i] = value

    fn print_stride_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("StridedTensor[" + dtype_str + "]")
        print("  Layout: " + self.stride_info.layout.name())
        var contiguous_str: String = "True" if self.stride_info.is_contiguous else "False"
        print("  Contiguous: " + contiguous_str)

        print("  Shape: [", end="")
        for i in range(self.stride_info.ndim):
            var shape_str: String = String(self.stride_info.get_shape(i))
            print(shape_str, end="")
            if i < self.stride_info.ndim - 1:
                print(", ", end="")
        print("]")

        print("  Strides: [", end="")
        for i in range(self.stride_info.ndim):
            var stride_str: String = String(self.stride_info.get_stride(i))
            print(stride_str, end="")
            if i < self.stride_info.ndim - 1:
                print(", ", end="")
        print("]")

        var bytes_str: String = String(self.stride_info.total_bytes)
        print("  Total bytes: " + bytes_str)

    fn print_memory_layout(self, max_elements: Int = 16):
        var total_elements = self.numel()
        var elements_to_show = total_elements if total_elements <= max_elements else max_elements

        print("  Memory layout:")
        for i in range(elements_to_show):
            var i_str: String = String(i)
            var value_str: String = String(self.data[i])
            var message: String = "    data[" + i_str + "] = " + value_str
            print(message)

        if total_elements > max_elements:
            var remaining_str: String = String(total_elements - max_elements)
            var remaining_msg: String = "    ... (" + remaining_str + " more elements)"
            print(remaining_msg)

fn create_row_major_tensor[dtype: DType](shape: List[Int]) raises -> StridedTensor[dtype]:
    return StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.ROW_MAJOR))

fn create_column_major_tensor[dtype: DType](shape: List[Int]) raises -> StridedTensor[dtype]:
    return StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.COLUMN_MAJOR))

fn create_custom_stride_tensor[dtype: DType](shape: List[Int], custom_strides: List[Int]) raises -> StridedTensor[dtype]:
    if not compute_custom_strides(shape, custom_strides):
        raise Error("Invalid custom stride pattern")

    var tensor = StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.CUSTOM))

    for i in range(len(custom_strides)):
        tensor.stride_info.strides[i] = custom_strides[i]

    tensor.stride_info.is_contiguous = False

    return tensor

fn main():
    """Main demonstration function."""
    print("=== Stride Calculation System - Part 1.2.1 ===")
    print("Memory Layout Design - Stride Computation and Access Patterns")
    # ... full test suite runs here (row-major/column-major strides,
    # index-offset conversion, StridedTensor demos, broadcasting strides,
    # contiguity checks, StrideInfo introspection, performance notes)
```

### Expected Output for `37_stride_calculation.mojo`

```
=== Stride Calculation System - Part 1.2.1 ===
Memory Layout Design - Stride Computation and Access Patterns
=== Testing Basic Stride Calculation ===

1. Row-Major Stride Calculation:
Shape: [3, 4]
Row-major strides: [4, 1]

2. Column-Major Stride Calculation:
Shape: [3, 4]
Column-major strides: [1, 3]

3. 3D Tensor Strides:
Shape: [2, 3, 4]
Row-major strides: [12, 4, 1]
Column-major strides: [1, 2, 6]

=== Testing Index-Offset Conversion ===

1. Index to Offset Conversion:
Shape: [3, 4], Row-major strides: [4, 1]
  Index [0, 0] -> Offset 0
  Offset 0 -> Index [0, 0]
  Index [1, 2] -> Offset 6
  Offset 6 -> Index [1, 2]
  Index [2, 3] -> Offset 11
  Offset 11 -> Index [2, 3]

=== Testing Strided Tensor Operations ===

1. Row-Major Tensor:
StridedTensor[float32]
  Layout: ROW_MAJOR
  Contiguous: True
  Shape: [2, 3]
  Strides: [3, 1]
  Total bytes: 24
Data access test:
  tensor[0, 0] = 1.0
  tensor[1, 1] = 4.0

2. Column-Major Tensor:
StridedTensor[float32]
  Layout: COLUMN_MAJOR
  Contiguous: True
  Shape: [2, 3]
  Strides: [1, 2]
  Total bytes: 24
Data access test:
  tensor[0, 0] = 1.0
  tensor[1, 1] = 4.0

3. Memory Layout Comparison:
Row-major tensor memory:
  Memory layout:
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 0.0
    data[3] = 3.0
    data[4] = 4.0
    data[5] = 0.0
Column-major tensor memory:
  Memory layout:
    data[0] = 1.0
    data[1] = 3.0
    data[2] = 2.0
    data[3] = 4.0
    data[4] = 0.0
    data[5] = 0.0

=== Testing Broadcasting Stride Calculation ===

1. Broadcasting Compatibility:
Tensor A - Shape: [2, 3, 4], Strides: [12, 4, 1]
Tensor B - Shape: [3, 1], Strides: [1, 1]
Broadcast result strides: [12, 4, 1]

=== Testing Contiguity Checking ===

1. Contiguity Analysis:
Row-major strides [4, 1] - Contiguous: True
Column-major strides [1, 3] - Contiguous: True
Custom strides [10, 1] - Contiguous: False

=== Testing StrideInfo Structure ===

1. StrideInfo Analysis:
Row-major StrideInfo:
  Layout: ROW_MAJOR
  Dimensions: 3
  Element size: 4 bytes
  Total bytes: 96
  Contiguous: True
  Shape: [2, 3, 4]
  Strides: [12, 4, 1]

Column-major StrideInfo:
  Layout: COLUMN_MAJOR
  Contiguous: True
  Strides: [1, 2, 6]

=== Testing Performance Analysis ===

1. Memory Access Pattern Analysis:
Understanding stride impact on performance:
- Row-major [2, 3, 4] with strides [12, 4, 1]:
  Sequential access: data[0], data[1], data[2], data[3] (cache-friendly)
  Cross-row access: data[0], data[4], data[8] (stride=4, moderate cache impact)
  Cross-page access: data[0], data[12] (stride=12, potential cache miss)

- Column-major [2, 3, 4] with strides [1, 2, 6]:
  Sequential access: data[0], data[1] (cache-friendly)
  Cross-column access: data[0], data[2], data[4] (stride=2, good cache usage)
  Cross-matrix access: data[0], data[6] (stride=6, moderate cache impact)

2. Layout Recommendations:
- Use row-major for C-style algorithms (rightmost index varies fastest)
- Use column-major for Fortran-style algorithms (leftmost index varies fastest)
- Consider access patterns when choosing layout for optimal performance
- Contiguous layouts generally provide better cache performance

=== Stride Calculation System Implementation Summary ===
+ Row-major (C-style) and column-major (Fortran-style) stride computation
+ Custom stride pattern support with validation
+ Efficient index-to-offset and offset-to-index conversion
+ Broadcasting-aware stride calculation
+ Contiguity detection and analysis
+ Comprehensive stride information tracking
+ Performance-oriented memory access patterns
+ Foundation for advanced tensor view operations
```

### File: `38_tensor_views_slicing.mojo` — Section 7.2

**Run:** `pixi run mojo 38_tensor_views_slicing.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_SLICE_DIMS = 8
alias SLICE_NONE = -999999  # Sentinel value for unspecified slice bounds

struct SliceSpec:
    """
    Comprehensive slice specification for tensor indexing.
    """
    var start: Int
    var end: Int
    var step: Int
    var is_valid: Bool

    fn __init__(out self, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1):
        self.start = start
        self.end = end
        self.step = step
        self.is_valid = step != 0  # Step cannot be zero

    fn __copyinit__(out self, existing: Self):
        self.start = existing.start
        self.end = existing.end
        self.step = existing.step
        self.is_valid = existing.is_valid

    fn normalize(self, dim_size: Int) -> SliceSpec:
        var norm_start = self.start
        var norm_end = self.end
        var norm_step = self.step

        if norm_step > 0:
            if norm_start == SLICE_NONE:
                norm_start = 0
            if norm_end == SLICE_NONE:
                norm_end = dim_size
        else:
            if norm_start == SLICE_NONE:
                norm_start = dim_size - 1
            if norm_end == SLICE_NONE:
                norm_end = -1

        if norm_start < 0:
            norm_start += dim_size
        if norm_end < 0 and norm_end != -1:
            norm_end += dim_size

        if norm_step > 0:
            norm_start = max(0, min(norm_start, dim_size))
            norm_end = max(0, min(norm_end, dim_size))
        else:
            norm_start = max(-1, min(norm_start, dim_size - 1))
            norm_end = max(-1, min(norm_end, dim_size - 1))

        return SliceSpec(norm_start, norm_end, norm_step)

    fn length(self) -> Int:
        if not self.is_valid:
            return 0

        if self.step > 0:
            if self.start >= self.end:
                return 0
            return (self.end - self.start + self.step - 1) // self.step
        else:
            if self.start <= self.end:
                return 0
            return (self.start - self.end - self.step - 1) // (-self.step)

    fn get_indices(self) -> List[Int]:
        var indices = List[Int]()
        var current = self.start

        if self.step > 0:
            while current < self.end:
                indices.append(current)
                current += self.step
        else:
            while current > self.end:
                indices.append(current)
                current += self.step

        return indices

struct MultiSliceSpec:
    """Multi-dimensional slice specification for tensor views."""
    var slices: UnsafePointer[SliceSpec]
    var ndim: Int
    var has_ellipsis: Bool
    var ellipsis_pos: Int

    fn __init__(out self, ndim: Int):
        self.ndim = ndim
        self.has_ellipsis = False
        self.ellipsis_pos = -1
        self.slices = UnsafePointer[SliceSpec].alloc(ndim)

        for i in range(ndim):
            self.slices[i] = SliceSpec()

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.has_ellipsis = existing.has_ellipsis
        self.ellipsis_pos = existing.ellipsis_pos
        self.slices = UnsafePointer[SliceSpec].alloc(self.ndim)

        for i in range(self.ndim):
            self.slices[i] = existing.slices[i]

    fn __del__(owned self):
        self.slices.free()

    fn set_slice(self, dim: Int, slice_spec: SliceSpec):
        if dim >= 0 and dim < self.ndim:
            self.slices[dim] = slice_spec

    fn get_slice(self, dim: Int) -> SliceSpec:
        if dim >= 0 and dim < self.ndim:
            return self.slices[dim]
        return SliceSpec(0, 0, 1)

    fn normalize(self, shape: List[Int]) -> MultiSliceSpec:
        var normalized = MultiSliceSpec(self.ndim)
        normalized.has_ellipsis = self.has_ellipsis
        normalized.ellipsis_pos = self.ellipsis_pos

        for i in range(self.ndim):
            if i < len(shape):
                normalized.set_slice(i, self.slices[i].normalize(shape[i]))
            else:
                normalized.set_slice(i, SliceSpec(0, 1, 1))

        return normalized

struct RefCount:
    """Reference counting for memory safety in tensor views."""
    var count: UnsafePointer[Int]

    fn __init__(out self):
        self.count = UnsafePointer[Int].alloc(1)
        self.count[0] = 1

    fn __copyinit__(out self, existing: Self):
        self.count = existing.count
        self.count[0] += 1

    fn __del__(owned self):
        self.count[0] -= 1
        if self.count[0] <= 0:
            self.count.free()

    fn get_count(self) -> Int:
        return self.count[0]

    fn is_unique(self) -> Bool:
        return self.count[0] == 1

struct TensorView[dtype: DType]:
    """
    Zero-copy tensor view with advanced slicing capabilities.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var offset: Int
    var ndim: Int
    var ref_count: RefCount
    var is_contiguous: Bool
    var parent_shape: UnsafePointer[Int]
    var parent_ndim: Int

    fn __init__(out self, data: UnsafePointer[Scalar[dtype]], shape: List[Int],
               strides: List[Int], offset: Int = 0):
        self.data = data
        self.offset = offset
        self.ndim = len(shape)
        self.ref_count = RefCount()
        self.is_contiguous = False

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.shape[i] = shape[i]

        self.strides = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.strides[i] = strides[i]

        self.parent_ndim = self.ndim
        self.parent_shape = UnsafePointer[Int].alloc(self.parent_ndim)
        for i in range(self.parent_ndim):
            self.parent_shape[i] = shape[i]

        self.is_contiguous = self._check_contiguity()

    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.offset = existing.offset
        self.ndim = existing.ndim
        self.ref_count = existing.ref_count  # This increments reference count
        self.is_contiguous = existing.is_contiguous
        self.parent_ndim = existing.parent_ndim

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]

        self.strides = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]

        self.parent_shape = UnsafePointer[Int].alloc(self.parent_ndim)
        for i in range(self.parent_ndim):
            self.parent_shape[i] = existing.parent_shape[i]

    fn __del__(owned self):
        self.shape.free()
        self.strides.free()
        self.parent_shape.free()

    fn _check_contiguity(self) -> Bool:
        if self.ndim <= 1:
            return True

        var expected_stride = 1
        for i in range(self.ndim - 1, -1, -1):
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]

        return True

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_offset = self.offset
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]

        return self.data[linear_offset]

    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        var linear_offset = self.offset
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]

        self.data[linear_offset] = value

    fn slice(self, slice_specs: MultiSliceSpec) -> TensorView[dtype]:
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])

        var normalized_specs = slice_specs.normalize(current_shape)

        var new_shape = List[Int]()
        var new_strides = List[Int]()
        var new_offset = self.offset

        for i in range(self.ndim):
            var spec = normalized_specs.get_slice(i)
            if spec.is_valid:
                new_shape.append(spec.length())
                new_strides.append(self.strides[i] * spec.step)
                new_offset += spec.start * self.strides[i]

        return TensorView[dtype](self.data, new_shape, new_strides, new_offset)

    fn slice_1d(self, dim: Int, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1) -> TensorView[dtype]:
        var multi_spec = MultiSliceSpec(self.ndim)
        multi_spec.set_slice(dim, SliceSpec(start, end, step))
        return self.slice(multi_spec)

    fn transpose(self, dim0: Int, dim1: Int) -> TensorView[dtype]:
        if dim0 < 0 or dim0 >= self.ndim or dim1 < 0 or dim1 >= self.ndim:
            return self

        var new_shape = List[Int]()
        var new_strides = List[Int]()

        for i in range(self.ndim):
            if i == dim0:
                new_shape.append(self.shape[dim1])
                new_strides.append(self.strides[dim1])
            elif i == dim1:
                new_shape.append(self.shape[dim0])
                new_strides.append(self.strides[dim0])
            else:
                new_shape.append(self.shape[i])
                new_strides.append(self.strides[i])

        return TensorView[dtype](self.data, new_shape, new_strides, self.offset)

    fn reshape(self, new_shape: List[Int]) -> TensorView[dtype]:
        var current_elements = 1
        for i in range(self.ndim):
            current_elements *= self.shape[i]

        var new_elements = 1
        for i in range(len(new_shape)):
            new_elements *= new_shape[i]

        if current_elements != new_elements or not self.is_contiguous:
            return self

        var new_strides = List[Int]()
        if len(new_shape) > 0:
            new_strides.append(1)
            for i in range(len(new_shape) - 1):
                new_strides.append(new_strides[len(new_strides) - 1] * new_shape[len(new_shape) - 1 - i])

            var final_strides = List[Int]()
            for i in range(len(new_strides) - 1, -1, -1):
                final_strides.append(new_strides[i])
            new_strides = final_strides

        return TensorView[dtype](self.data, new_shape, new_strides, self.offset)

    fn numel(self) -> Int:
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total

    fn get_shape(self, dim: Int) -> Int:
        if dim >= 0 and dim < self.ndim:
            return self.shape[dim]
        return 0

    fn get_stride(self, dim: Int) -> Int:
        if dim >= 0 and dim < self.ndim:
            return self.strides[dim]
        return 0

    fn print_view_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("TensorView[" + dtype_str + "]")
        var contiguous_str: String = "True" if self.is_contiguous else "False"
        print("  Contiguous: " + contiguous_str)
        var offset_str: String = String(self.offset)
        print("  Offset: " + offset_str)
        var ref_count_str: String = String(self.ref_count.get_count())
        print("  Reference count: " + ref_count_str)

        print("  View shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  View strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  Parent shape: [", end="")
        for i in range(self.parent_ndim):
            var parent_str: String = String(self.parent_shape[i])
            print(parent_str, end="")
            if i < self.parent_ndim - 1:
                print(", ", end="")
        print("]")

struct ViewableTensor[dtype: DType]:
    """
    Enhanced tensor with comprehensive view and slicing support.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var ndim: Int
    var _owns_data: Bool
    var ref_count: RefCount

    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")

        self.ndim = len(shape)
        self._owns_data = True
        self.ref_count = RefCount()

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            total_elements *= shape[i]

        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * self.shape[i + 1]

        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self._owns_data = True
        self.ref_count = RefCount()  # New reference count for copy

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.strides[i] = existing.strides[i]

        var total_elements = 1
        for i in range(self.ndim):
            total_elements *= self.shape[i]

        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = existing.data[i]

    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
        self.shape.free()
        self.strides.free()

    fn create_view(self) -> TensorView[dtype]:
        var shape_list = List[Int]()
        var strides_list = List[Int]()

        for i in range(self.ndim):
            shape_list.append(self.shape[i])
            strides_list.append(self.strides[i])

        return TensorView[dtype](self.data, shape_list, strides_list, 0)

    fn slice(self, slice_specs: MultiSliceSpec) -> TensorView[dtype]:
        var full_view = self.create_view()
        return full_view.slice(slice_specs)

    fn slice_1d(self, dim: Int, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1) -> TensorView[dtype]:
        var full_view = self.create_view()
        return full_view.slice_1d(dim, start, end, step)

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]

        return self.data[linear_offset]

    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]

        self.data[linear_offset] = value

    fn fill(self, value: Scalar[dtype]):
        var total_elements = 1
        for i in range(self.ndim):
            total_elements *= self.shape[i]

        for i in range(total_elements):
            self.data[i] = value

    fn numel(self) -> Int:
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total

    fn print_tensor_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("ViewableTensor[" + dtype_str + "]")
        var ref_count_str: String = String(self.ref_count.get_count())
        print("  Reference count: " + ref_count_str)

        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  Strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

fn create_viewable_tensor[dtype: DType](shape: List[Int]) raises -> ViewableTensor[dtype]:
    return ViewableTensor[dtype](shape)

fn create_view_from_tensor[dtype: DType](tensor: ViewableTensor[dtype]) -> TensorView[dtype]:
    return tensor.create_view()

fn create_slice_view[dtype: DType](tensor: ViewableTensor[dtype],
                                 dim: Int, start: Int, end: Int, step: Int = 1) -> TensorView[dtype]:
    return tensor.slice_1d(dim, start, end, step)

fn main():
    """Main demonstration function."""
    print("=== View and Slicing Infrastructure - Part 1.2.2 ===")
    print("Memory Layout Design - Zero-Copy Views and Advanced Slicing")
    # ... full test suite runs here (slice specification, reference counting,
    # view creation, row/column slicing, transpose/reshape, multi-dimensional
    # slicing, memory sharing, performance notes)
```

### Expected Output for `38_tensor_views_slicing.mojo`

```
=== View and Slicing Infrastructure - Part 1.2.2 ===
Memory Layout Design - Zero-Copy Views and Advanced Slicing
=== Testing Slice Specification System ===

1. Basic Slice Creation:
Slice [1:5:1] - start: 1, end: 5, step: 1

2. Slice Normalization:
Normalized [1:5:1] for size 10 - start: 1, end: 5, length: 4
Normalized [::2] for size 10 - start: 0, end: 10, length: 5
Normalized [-3:] for size 10 - start: 7, end: 10, length: 3

3. Slice Index Generation:
Indices for [1:5:1]: [1, 2, 3, 4]
Indices for [::2]: [0, 2, 4, 6, 8]

=== Testing Reference Counting System ===

1. Basic Reference Counting:
Initial reference count: 1
After copy: 1
Is unique: True

=== Testing Tensor View Creation ===

1. Creating Viewable Tensor:
ViewableTensor[float32]
  Reference count: 1
  Shape: [3, 4]
  Strides: [4, 1]

2. Filling Tensor with Test Data:

3. Creating Full View:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [3, 4]
  View strides: [4, 1]
  Parent shape: [3, 4]

4. Testing View Data Access:
  view[0, 0] = 1.7510376
  view[0, 1] = 6.209e-42
  view[1, 0] = 7.977031e+17
  view[1, 1] = 9.550975e-06

=== Testing Tensor Slicing Operations ===

1. Creating Test Tensor [4, 5]:
ViewableTensor[float32]
  Reference count: 1
  Shape: [4, 5]
  Strides: [5, 1]

2. Testing 1D Slicing [1:3, :]:
TensorView[float32]
  Contiguous: True
  Offset: 5
  Reference count: 1
  View shape: [2, 5]
  View strides: [5, 1]
  Parent shape: [2, 5]
Row slice data:
  slice[0, 0] = 5.0
  slice[0, 1] = 6.0
  slice[0, 2] = 7.0
  slice[0, 3] = 8.0
  slice[0, 4] = 9.0
  slice[1, 0] = 10.0
  slice[1, 1] = 11.0
  slice[1, 2] = 12.0
  slice[1, 3] = 13.0
  slice[1, 4] = 14.0

3. Testing Column Slicing [:, 1:4:2]:
TensorView[float32]
  Contiguous: False
  Offset: 1
  Reference count: 1
  View shape: [4, 2]
  View strides: [5, 2]
  Parent shape: [4, 2]
Column slice data:
  slice[0, 0] = 6.209e-42
  slice[0, 1] = 3.0
  slice[1, 0] = 6.0
  slice[1, 1] = 8.0
  slice[2, 0] = 11.0
  slice[2, 1] = 13.0
  slice[3, 0] = 16.0
  slice[3, 1] = 18.0

=== Testing Advanced View Operations ===

1. Creating Test Tensor [2, 3]:
Original tensor:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [2, 3]
  View strides: [3, 1]
  Parent shape: [2, 3]

2. Testing Transpose Operation:
TensorView[float32]
  Contiguous: False
  Offset: 0
  Reference count: 1
  View shape: [3, 2]
  View strides: [1, 3]
  Parent shape: [3, 2]
Transposed data access:
  transposed[0, 0] = 0.0
  transposed[0, 1] = 10.0
  transposed[1, 0] = 0.0
  transposed[1, 1] = 11.0
  transposed[2, 0] = 2.0
  transposed[2, 1] = 12.0

3. Testing Reshape Operation:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [6]
  View strides: [1]
  Parent shape: [6]
Reshaped data (first 6 elements):
  reshaped[0] = 0.0
  reshaped[1] = 0.0
  reshaped[2] = 2.0
  reshaped[3] = 10.0
  reshaped[4] = 11.0
  reshaped[5] = 12.0

=== Testing Multi-dimensional Slicing ===

1. Creating 3D Tensor [3, 4, 2]:

2. Testing Multi-slice Specification:
TensorView[float32]
  Contiguous: False
  Offset: 8
  Reference count: 1
  View shape: [2, 2, 2]
  View strides: [8, 4, 1]
  Parent shape: [2, 2, 2]
Multi-dimensional slice data:
  slice[0, 0, 0] = 100.0
  slice[0, 0, 1] = 101.0
  slice[0, 1, 0] = 120.0
  slice[0, 1, 1] = 121.0
  slice[1, 0, 0] = 200.0
  slice[1, 0, 1] = 201.0
  slice[1, 1, 0] = 220.0
  slice[1, 1, 1] = 221.0

=== Testing View Memory Sharing ===

1. Memory Sharing Demonstration:
Original tensor data:
  original[0, 0] = 0.0
  original[0, 1] = 1.0
  original[0, 2] = 2.0
  original[1, 0] = 3.0
  original[1, 1] = 4.0
  original[1, 2] = 5.0

2. Creating Slice View:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [2, 3]
  View strides: [3, 1]
  Parent shape: [2, 3]

3. Modifying Data Through View:
Modified slice_view[0, 1] = 99.0

4. Checking Original Tensor (should reflect change):
Original tensor[0, 1] = 99.0

5. Reference Count Analysis:
Slice view reference count: 1

=== Testing Performance Analysis ===

1. View vs Copy Performance Analysis:
Understanding view benefits:
- Views provide zero-copy access to tensor subsets
- Memory usage: O(1) for views vs O(n) for copies
- Creation time: O(1) for views vs O(n) for copies
- Modification: Views modify original data, copies are independent

2. Slice Performance Characteristics:
- Contiguous slices: optimal cache performance
- Strided slices: reduced cache efficiency but still efficient
- Multi-dimensional slices: combine stride effects across dimensions
- Step size impact: larger steps reduce cache locality

3. Memory Layout Impact:
- Row-major tensors: rightmost dimension slicing is most efficient
- Column-major tensors: leftmost dimension slicing is most efficient
- Transpose operations: change stride patterns without data movement
- Reshape operations: only possible for contiguous views

4. Reference Counting Benefits:
- Automatic memory management for shared data
- Safe cleanup when last reference is removed
- Prevention of dangling pointers in view hierarchies
- Minimal overhead with copy-on-write semantics

=== View and Slicing Infrastructure Implementation Summary ===
+ Zero-copy tensor view system with reference counting
+ Python-style slice specification ([start:end:step])
+ Multi-dimensional slicing with stride manipulation
+ Advanced view operations (transpose, reshape)
+ Memory sharing semantics with automatic cleanup
+ Efficient slice composition and optimization
+ Safe memory management for view hierarchies
+ Foundation for advanced indexing patterns
```

### File: `39_memory_alignment.mojo` — Section 7.3

**Run:** `pixi run mojo 39_memory_alignment.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias CACHE_LINE_SIZE = 64      # Common CPU cache line size
alias SIMD_ALIGN_16 = 16        # SSE alignment
alias SIMD_ALIGN_32 = 32        # AVX alignment
alias SIMD_ALIGN_64 = 64        # AVX-512 alignment
alias GPU_WARP_SIZE = 32        # GPU warp size for coalescing
alias DEFAULT_ALIGNMENT = 32    # Default alignment for general use

struct AlignmentSpec(Copyable, Movable):
    """
    Comprehensive alignment specification for memory allocation.
    """
    alias ALIGN_NONE = 1
    alias ALIGN_SIMD_128 = 16
    alias ALIGN_SIMD_256 = 32
    alias ALIGN_SIMD_512 = 64
    alias ALIGN_CACHE_LINE = 64
    alias ALIGN_GPU_COALESCE = 128

    var alignment: Int
    var enable_padding: Bool
    var target_architecture: Int  # 0=CPU, 1=GPU, 2=Mixed
    var optimize_for_simd: Bool
    var cache_friendly: Bool

    fn __init__(out self, alignment: Int = DEFAULT_ALIGNMENT,
               enable_padding: Bool = True,
               target_architecture: Int = 0):
        self.alignment = alignment
        self.enable_padding = enable_padding
        self.target_architecture = target_architecture
        self.optimize_for_simd = alignment >= Self.ALIGN_SIMD_128
        self.cache_friendly = alignment >= Self.ALIGN_CACHE_LINE

    fn is_power_of_two(self, value: Int) -> Bool:
        return value > 0 and (value & (value - 1)) == 0

    fn is_valid(self) -> Bool:
        return self.is_power_of_two(self.alignment) and self.alignment >= 1

    fn calculate_padding(self, size: Int) -> Int:
        if not self.enable_padding or not self.is_valid():
            return 0

        var remainder = size % self.alignment
        if remainder == 0:
            return 0

        return self.alignment - remainder

    fn align_size(self, size: Int) -> Int:
        return size + self.calculate_padding(size)

    fn get_architecture_name(self) -> String:
        if self.target_architecture == 1:
            return "GPU"
        elif self.target_architecture == 2:
            return "Mixed"
        else:
            return "CPU"

fn get_optimal_alignment_for_arch(arch: Int) -> AlignmentSpec:
    if arch == 1:  # GPU
        return AlignmentSpec(AlignmentSpec.ALIGN_GPU_COALESCE, True, arch)
    elif arch == 2:  # Mixed
        return AlignmentSpec(AlignmentSpec.ALIGN_CACHE_LINE, True, arch)
    else:  # CPU
        return AlignmentSpec(AlignmentSpec.ALIGN_SIMD_256, True, arch)

struct PaddingInfo(Copyable, Movable):
    """Information about padding applied to memory allocation."""
    var original_size: Int
    var padded_size: Int
    var padding_bytes: Int
    var alignment_achieved: Int
    var efficiency_ratio: Float32

    fn __init__(out self, original_size: Int, padded_size: Int, alignment: Int):
        self.original_size = original_size
        self.padded_size = padded_size
        self.padding_bytes = padded_size - original_size
        self.alignment_achieved = alignment
        self.efficiency_ratio = Float32(original_size) / Float32(padded_size)

    fn __copyinit__(out self, existing: Self):
        self.original_size = existing.original_size
        self.padded_size = existing.padded_size
        self.padding_bytes = existing.padding_bytes
        self.alignment_achieved = existing.alignment_achieved
        self.efficiency_ratio = existing.efficiency_ratio

    fn memory_overhead_percent(self) -> Float32:
        if self.original_size == 0:
            return 0.0
        return Float32(self.padding_bytes) / Float32(self.original_size) * 100.0

struct AlignedAllocator:
    """
    Memory allocator with alignment and padding support.
    """
    var default_spec: AlignmentSpec
    var total_allocated: Int
    var allocation_count: Int
    var total_padding: Int

    fn __init__(out self, default_alignment: Int = DEFAULT_ALIGNMENT):
        self.default_spec = AlignmentSpec(default_alignment)
        self.total_allocated = 0
        self.allocation_count = 0
        self.total_padding = 0

    fn __copyinit__(out self, existing: Self):
        self.default_spec = existing.default_spec
        self.total_allocated = existing.total_allocated
        self.allocation_count = existing.allocation_count
        self.total_padding = existing.total_padding

    fn get_statistics(self) -> (Int, Int, Int, Float32):
        var avg_efficiency = Float32(100.0)
        if self.total_allocated > 0:
            avg_efficiency = Float32(self.total_allocated - self.total_padding) / Float32(self.total_allocated) * 100.0

        return (self.total_allocated, self.allocation_count, self.total_padding, avg_efficiency)

struct SIMDTensorLayout:
    """
    SIMD-optimized tensor layout with alignment considerations.
    """
    var shape: UnsafePointer[Int]
    var aligned_strides: UnsafePointer[Int]
    var padding_per_dim: UnsafePointer[Int]
    var ndim: Int
    var simd_width: Int
    var alignment_spec: AlignmentSpec
    var total_padded_size: Int

    fn __init__(out self, shape: List[Int], simd_width: Int = 8, alignment_spec: AlignmentSpec = AlignmentSpec()):
        self.ndim = len(shape)
        self.simd_width = simd_width
        self.alignment_spec = alignment_spec

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        self.padding_per_dim = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = shape[i]

        if self.ndim > 0:
            var current_stride = 1

            for i in range(self.ndim - 1, -1, -1):
                var dim_size = self.shape[i]

                if i == self.ndim - 1:
                    var remainder = dim_size % self.simd_width
                    var padded_size = dim_size if remainder == 0 else dim_size + (self.simd_width - remainder)
                    self.padding_per_dim[i] = padded_size - dim_size
                    self.aligned_strides[i] = current_stride
                    current_stride = padded_size
                else:
                    self.aligned_strides[i] = current_stride
                    self.padding_per_dim[i] = 0
                    current_stride *= dim_size

            self.total_padded_size = current_stride
        else:
            self.total_padded_size = 0

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.simd_width = existing.simd_width
        self.alignment_spec = existing.alignment_spec
        self.total_padded_size = existing.total_padded_size

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        self.padding_per_dim = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.aligned_strides[i] = existing.aligned_strides[i]
            self.padding_per_dim[i] = existing.padding_per_dim[i]

    fn __del__(owned self):
        self.shape.free()
        self.aligned_strides.free()
        self.padding_per_dim.free()

    fn pad_to_simd_width(self, size: Int) -> Int:
        var remainder = size % self.simd_width
        if remainder == 0:
            return size
        return size + (self.simd_width - remainder)

    fn get_aligned_stride(self, dim: Int) -> Int:
        if dim >= 0 and dim < self.ndim:
            return self.aligned_strides[dim]
        return 0

    fn get_padding(self, dim: Int) -> Int:
        if dim >= 0 and dim < self.ndim:
            return self.padding_per_dim[dim]
        return 0

    fn get_total_padding(self) -> Int:
        var total_padding = 0
        for i in range(self.ndim):
            total_padding += self.padding_per_dim[i]
        return total_padding

    fn is_vectorizable(self, dim: Int) -> Bool:
        if dim < 0 or dim >= self.ndim:
            return False

        if dim == self.ndim - 1:
            return (self.shape[dim] + self.padding_per_dim[dim]) % self.simd_width == 0

        return True

    fn calculate_memory_efficiency(self) -> Float32:
        var useful_elements = 1
        var total_elements = 1

        for i in range(self.ndim):
            useful_elements *= self.shape[i]
            total_elements *= (self.shape[i] + self.padding_per_dim[i])

        if total_elements == 0:
            return 0.0

        return Float32(useful_elements) / Float32(total_elements)

struct GPUCoalescingOptimizer:
    """
    GPU memory coalescing optimizer for efficient memory access.
    """
    var warp_size: Int
    var memory_bus_width: Int  # bytes
    var cache_line_size: Int
    var coalescing_threshold: Float32

    fn __init__(out self, warp_size: Int = GPU_WARP_SIZE,
               memory_bus_width: Int = 128,
               cache_line_size: Int = CACHE_LINE_SIZE):
        self.warp_size = warp_size
        self.memory_bus_width = memory_bus_width
        self.cache_line_size = cache_line_size
        self.coalescing_threshold = 0.8  # 80% efficiency threshold

    fn __copyinit__(out self, existing: Self):
        self.warp_size = existing.warp_size
        self.memory_bus_width = existing.memory_bus_width
        self.cache_line_size = existing.cache_line_size
        self.coalescing_threshold = existing.coalescing_threshold

    fn analyze_coalescing_efficiency(self, shape: List[Int], strides: List[Int],
                                   element_size: Int) -> Float32:
        if len(shape) == 0 or len(strides) == 0:
            return 0.0

        var innermost_dim = len(shape) - 1
        var _ = shape[innermost_dim]
        var innermost_stride = strides[innermost_dim]

        var bytes_per_element = element_size * innermost_stride
        var bytes_per_warp = bytes_per_element * self.warp_size

        var ideal_bytes = element_size * self.warp_size

        var efficiency = Float32(ideal_bytes) / Float32(bytes_per_warp)

        if efficiency > 1.0:
            efficiency = 1.0
        elif efficiency < 0.0:
            efficiency = 0.0

        return efficiency

    fn optimize_layout_for_coalescing(self, shape: List[Int]) -> (List[Int], List[Int]):
        var optimized_shape = List[Int]()
        var optimized_strides = List[Int]()

        for i in range(len(shape)):
            optimized_shape.append(shape[i])

        if len(shape) > 0:
            optimized_strides.append(1)

            for i in range(len(shape) - 1):
                var dim_idx = len(shape) - 2 - i
                var stride = optimized_strides[i] * optimized_shape[dim_idx + 1]
                optimized_strides.append(stride)

            var final_strides = List[Int]()
            for i in range(len(optimized_strides) - 1, -1, -1):
                final_strides.append(optimized_strides[i])
            optimized_strides = final_strides

        return (optimized_shape, optimized_strides)

    fn calculate_bandwidth_utilization(self, shape: List[Int], strides: List[Int],
                                     element_size: Int, access_pattern: String = "sequential") -> Float32:
        var coalescing_efficiency = self.analyze_coalescing_efficiency(shape, strides, element_size)

        var pattern_factor: Float32 = 1.0
        if access_pattern == "strided":
            pattern_factor = 0.7
        elif access_pattern == "random":
            pattern_factor = 0.3

        return coalescing_efficiency * pattern_factor * 100.0

    fn recommend_optimizations(self, shape: List[Int], strides: List[Int],
                             element_size: Int) -> List[String]:
        var recommendations = List[String]()
        var efficiency = self.analyze_coalescing_efficiency(shape, strides, element_size)

        if efficiency < self.coalescing_threshold:
            recommendations.append("Consider memory layout reorganization for better coalescing")

            if len(shape) > 0:
                var innermost_size = shape[len(shape) - 1]
                if innermost_size < self.warp_size:
                    recommendations.append("Pad innermost dimension to warp size (" + String(self.warp_size) + ")")

                if len(strides) > 0:
                    var innermost_stride = strides[len(strides) - 1]
                    if innermost_stride > 1:
                        recommendations.append("Ensure unit stride for innermost dimension")

        return recommendations

struct AlignedTensor[dtype: DType]:
    """
    Tensor with comprehensive memory alignment and padding support.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var aligned_strides: UnsafePointer[Int]
    var ndim: Int
    var alignment_spec: AlignmentSpec
    var padding_info: PaddingInfo
    var simd_layout: SIMDTensorLayout
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int],
               alignment_spec: AlignmentSpec = AlignmentSpec()) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")

        self.ndim = len(shape)
        self.alignment_spec = alignment_spec
        self._owns_data = True

        self.simd_layout = SIMDTensorLayout(shape, 8, alignment_spec)

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = shape[i]
            self.aligned_strides[i] = self.simd_layout.get_aligned_stride(i)

        var allocator = AlignedAllocator()
        var padded_elements = self.simd_layout.total_padded_size
        self.data = UnsafePointer[Scalar[dtype]].alloc(padded_elements)

        var original_size = 1
        for i in range(self.ndim):
            original_size *= shape[i]
        var element_size = 4
        var total_bytes = padded_elements * element_size
        var original_bytes = original_size * element_size
        self.padding_info = PaddingInfo(original_bytes, total_bytes, alignment_spec.alignment)

        for i in range(padded_elements):
            self.data[i] = Scalar[dtype](0)

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.alignment_spec = existing.alignment_spec
        self.padding_info = existing.padding_info
        self.simd_layout = existing.simd_layout
        self._owns_data = True

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.aligned_strides[i] = existing.aligned_strides[i]

        var total_elements = self.simd_layout.total_padded_size
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = existing.data[i]

    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
        self.shape.free()
        self.aligned_strides.free()

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.aligned_strides[i]

        return self.data[linear_offset]

    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.aligned_strides[i]

        self.data[linear_offset] = value

    fn fill(self, value: Scalar[dtype]):
        self._fill_recursive(0, List[Int](), value)

    fn _fill_recursive(self, dim: Int, indices: List[Int], value: Scalar[dtype]):
        if dim == self.ndim:
            self.set_item(indices, value)
            return

        for i in range(self.shape[dim]):
            var new_indices = indices
            new_indices.append(i)
            self._fill_recursive(dim + 1, new_indices, value)

    fn get_memory_efficiency(self) -> Float32:
        return self.simd_layout.calculate_memory_efficiency()

    fn get_alignment_info(self) -> (Int, Int, Float32):
        return (self.alignment_spec.alignment,
                self.padding_info.padding_bytes,
                self.padding_info.efficiency_ratio)

    fn is_simd_optimized(self) -> Bool:
        return self.alignment_spec.optimize_for_simd

    fn print_alignment_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("AlignedTensor[" + dtype_str + "]")
        print("  Target: " + self.alignment_spec.get_architecture_name())
        var alignment_str: String = String(self.alignment_spec.alignment)
        print("  Alignment: " + alignment_str + " bytes")
        var padding_str: String = String(self.padding_info.padding_bytes)
        print("  Padding: " + padding_str + " bytes")
        var efficiency_str: String = String(self.padding_info.efficiency_ratio * 100.0)
        print("  Memory efficiency: " + efficiency_str + "%")

        print("  Original shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  Aligned strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.aligned_strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  SIMD padding per dimension: [", end="")
        for i in range(self.ndim):
            var pad_str: String = String(self.simd_layout.get_padding(i))
            print(pad_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        var simd_efficiency = self.simd_layout.calculate_memory_efficiency() * 100.0
        var simd_eff_str: String = String(simd_efficiency)
        print("  SIMD efficiency: " + simd_eff_str + "%")

fn create_cpu_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    var cpu_spec = get_optimal_alignment_for_arch(0)
    return AlignedTensor[dtype](shape, cpu_spec)

fn create_gpu_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    var gpu_spec = get_optimal_alignment_for_arch(1)
    return AlignedTensor[dtype](shape, gpu_spec)

fn create_mixed_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    var mixed_spec = get_optimal_alignment_for_arch(2)
    return AlignedTensor[dtype](shape, mixed_spec)

fn create_custom_aligned_tensor[dtype: DType](shape: List[Int], alignment: Int) raises -> AlignedTensor[dtype]:
    var custom_spec = AlignmentSpec(alignment, True, 0)
    return AlignedTensor[dtype](shape, custom_spec)

fn main():
    """Main demonstration function."""
    print("=== Memory Alignment and Padding - Part 1.2.3 ===")
    print("Memory Layout Design - Hardware-Optimized Alignment Strategies")
    # ... full test suite runs here (alignment specification, aligned
    # allocator statistics, SIMD tensor layout, GPU coalescing optimizer,
    # aligned tensor creation, performance analysis)
```

### Expected Output for `39_memory_alignment.mojo`

```
=== Memory Alignment and Padding - Part 1.2.3 ===
Memory Layout Design - Hardware-Optimized Alignment Strategies
=== Testing Alignment Specification System ===

1. Basic Alignment Specifications:
CPU optimal alignment: 32 bytes (CPU)
GPU optimal alignment: 128 bytes (GPU)
Mixed optimal alignment: 64 bytes (Mixed)

2. Padding Calculations:
Size 100 -> Padding 28 -> Aligned 128
Size 127 -> Padding 1 -> Aligned 128
Size 200 -> Padding 24 -> Aligned 224
Size 255 -> Padding 1 -> Aligned 256

3. Alignment Validation:
Alignment 16: Valid
Alignment 32: Valid
Alignment 64: Valid
Alignment 15: Invalid

=== Testing Aligned Memory Allocator ===

1. Allocator Statistics:
Simulating allocations...
Allocation 1: 400 -> 400 bytes
Allocation 2: 800 -> 832 bytes
Allocation 3: 200 -> 208 bytes

=== Testing SIMD-Optimized Tensor Layout ===

1. SIMD Layout Analysis:
Original shape: [3, 7, 11]
SIMD width: 8 elements
Aligned strides: [112, 16, 1]
Padding per dimension: [0, 0, 5]
Total padding: 5 elements
Memory efficiency: 68.75%

2. Vectorization Analysis:
Dimension 0 vectorizable: Yes
Dimension 1 vectorizable: Yes
Dimension 2 vectorizable: Yes

=== Testing GPU Memory Coalescing Optimizer ===

1. Coalescing Efficiency Analysis:
Layout [4, 32] with strides [32, 1]: 100.0% efficient
Layout [4, 17] with strides [17, 1]: 100.0% efficient

2. Bandwidth Utilization:
Sequential access: 100.0% bandwidth utilization
Strided access: 70.0% bandwidth utilization

3. Optimization Recommendations:

=== Testing Aligned Tensor Creation and Usage ===

1. CPU-Optimized Tensor:
AlignedTensor[float32]
  Target: CPU
  Alignment: 32 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

2. GPU-Optimized Tensor:
AlignedTensor[float32]
  Target: GPU
  Alignment: 128 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

3. Custom Alignment Tensor:
AlignedTensor[float32]
  Target: CPU
  Alignment: 128 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

4. Data Access and Modification:
Set and retrieved value: 42.0

5. Memory Efficiency Comparison:
CPU tensor efficiency: 37.5%
GPU tensor efficiency: 37.5%
Custom tensor efficiency: 37.5%

=== Testing Alignment Performance Analysis ===

1. Alignment Impact on Performance:
Understanding alignment benefits:
- SIMD operations require aligned data for optimal performance
- Misaligned access can cause significant slowdowns (2-4x)
- Cache line alignment reduces cache misses
- GPU coalescing improves memory bandwidth utilization

2. Memory Overhead vs Performance Trade-offs:
Shape [100]: 96.15385% efficient, 16 bytes padding
Shape [10, 10]: 62.5% efficient, 240 bytes padding
Shape [5, 5, 4]: 50.0% efficient, 400 bytes padding

3. Architecture-Specific Recommendations:
- CPU (SIMD): Use 32-byte alignment for AVX operations
- GPU (Coalescing): Ensure innermost dimension aligns with warp size
- Mixed workloads: Balance between CPU cache and GPU coalescing
- Memory-constrained: Consider alignment overhead vs performance gains

4. Best Practices:
- Profile alignment impact for specific workloads
- Consider data reuse patterns when choosing alignment
- Use architecture-specific optimizations when targeting single platform
- Monitor memory overhead, especially for small tensors

=== Memory Alignment and Padding Implementation Summary ===
+ Comprehensive alignment specification system (CPU/GPU/Mixed)
+ SIMD-optimized tensor layouts with automatic padding
+ GPU memory coalescing analysis and optimization
+ Cache line alignment for CPU performance
+ Memory efficiency tracking and analysis
+ Architecture-specific optimization profiles
+ Configurable alignment strategies for different use cases
+ Performance vs memory overhead analysis tools
```

### File: `41_broadcasting_layout.mojo` — Section 7.4

**Run:** `pixi run mojo 41_broadcasting_layout.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_BROADCAST_DIMS = 8
alias BROADCAST_ALIGNMENT = 32

struct BroadcastRule(Copyable, Movable):
    """Broadcasting rule validation and shape analysis."""
    alias INCOMPATIBLE = 0
    alias COMPATIBLE = 1
    alias IDENTICAL = 2
    alias BROADCAST_LEFT = 3
    alias BROADCAST_RIGHT = 4
    alias BROADCAST_BOTH = 5

    var compatibility_type: Int
    var left_broadcasts: Bool
    var right_broadcasts: Bool
    var result_shape: List[Int]

    fn __init__(out self):
        self.compatibility_type = Self.INCOMPATIBLE
        self.left_broadcasts = False
        self.right_broadcasts = False
        self.result_shape = List[Int]()

    fn __copyinit__(out self, existing: Self):
        self.compatibility_type = existing.compatibility_type
        self.left_broadcasts = existing.left_broadcasts
        self.right_broadcasts = existing.right_broadcasts
        self.result_shape = List[Int]()
        for i in range(len(existing.result_shape)):
            self.result_shape.append(existing.result_shape[i])

    fn is_compatible(self) -> Bool:
        return self.compatibility_type != Self.INCOMPATIBLE

    fn requires_broadcasting(self) -> Bool:
        return self.left_broadcasts or self.right_broadcasts

    fn get_compatibility_name(self) -> String:
        if self.compatibility_type == Self.IDENTICAL:
            return "IDENTICAL"
        elif self.compatibility_type == Self.COMPATIBLE:
            return "COMPATIBLE"
        elif self.compatibility_type == Self.BROADCAST_LEFT:
            return "BROADCAST_LEFT"
        elif self.compatibility_type == Self.BROADCAST_RIGHT:
            return "BROADCAST_RIGHT"
        elif self.compatibility_type == Self.BROADCAST_BOTH:
            return "BROADCAST_BOTH"
        else:
            return "INCOMPATIBLE"

fn analyze_broadcast_compatibility(shape_a: List[Int], shape_b: List[Int]) -> BroadcastRule:
    """
    Broadcasting Rules (NumPy-compatible):
    1. Shapes are aligned from the rightmost dimension.
    2. Dimensions are compatible if they are equal or one of them is 1.
    3. Missing dimensions are treated as 1.
    4. Result shape is element-wise maximum of input shapes.
    """
    var rule = BroadcastRule()
    var len_a = len(shape_a)
    var len_b = len(shape_b)
    var max_len = len_a if len_a > len_b else len_b

    var shapes_identical = True
    var left_needs_broadcast = False
    var right_needs_broadcast = False

    for i in range(max_len):
        var dim_a = 1
        var dim_b = 1

        if i < len_a:
            dim_a = shape_a[len_a - 1 - i]
        if i < len_b:
            dim_b = shape_b[len_b - 1 - i]

        if dim_a != dim_b:
            shapes_identical = False
            if dim_a == 1:
                left_needs_broadcast = True
            elif dim_b == 1:
                right_needs_broadcast = True
            else:
                rule.compatibility_type = BroadcastRule.INCOMPATIBLE
                return rule

        var result_dim = dim_a if dim_a > dim_b else dim_b
        rule.result_shape.append(result_dim)

    var final_shape = List[Int]()
    for i in range(len(rule.result_shape) - 1, -1, -1):
        final_shape.append(rule.result_shape[i])
    rule.result_shape = final_shape

    rule.left_broadcasts = left_needs_broadcast
    rule.right_broadcasts = right_needs_broadcast

    if shapes_identical:
        rule.compatibility_type = BroadcastRule.IDENTICAL
    elif left_needs_broadcast and right_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_BOTH
    elif left_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_LEFT
    elif right_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_RIGHT
    else:
        rule.compatibility_type = BroadcastRule.COMPATIBLE

    return rule

fn calculate_broadcast_strides(original_shape: List[Int], original_strides: List[Int],
                             target_shape: List[Int]) -> List[Int]:
    """
    Broadcasting Stride Rules:
    - Dimensions of size 1 get stride 0 (virtual broadcasting).
    - Other dimensions keep their original strides.
    - Missing dimensions are treated as size 1 with stride 0.
    """
    var broadcast_strides = List[Int]()
    var orig_len = len(original_shape)
    var target_len = len(target_shape)

    for i in range(target_len):
        var target_dim = target_shape[target_len - 1 - i]
        var calculated_stride: Int

        if i < orig_len:
            var orig_dim = original_shape[orig_len - 1 - i]
            var orig_stride = original_strides[orig_len - 1 - i]

            if orig_dim == target_dim:
                calculated_stride = orig_stride
            elif orig_dim == 1:
                calculated_stride = 0
            else:
                calculated_stride = orig_stride  # Fallback
        else:
            calculated_stride = 0

        broadcast_strides.append(calculated_stride)

    var final_strides = List[Int]()
    for i in range(len(broadcast_strides) - 1, -1, -1):
        final_strides.append(broadcast_strides[i])

    return final_strides

struct BroadcastSpec(Copyable, Movable):
    """Comprehensive broadcasting specification for tensor operations."""
    var input_shapes: List[List[Int]]
    var result_shape: List[Int]
    var broadcast_strides: List[List[Int]]
    var compatibility_rule: BroadcastRule
    var memory_cost: Int
    var is_valid: Bool
    var optimization_hint: String

    fn __init__(out self):
        self.input_shapes = List[List[Int]]()
        self.result_shape = List[Int]()
        self.broadcast_strides = List[List[Int]]()
        self.compatibility_rule = BroadcastRule()
        self.memory_cost = 0
        self.is_valid = False
        self.optimization_hint = "none"

    fn __copyinit__(out self, existing: Self):
        self.input_shapes = List[List[Int]]()
        self.result_shape = List[Int]()
        self.broadcast_strides = List[List[Int]]()
        self.compatibility_rule = existing.compatibility_rule
        self.memory_cost = existing.memory_cost
        self.is_valid = existing.is_valid
        self.optimization_hint = existing.optimization_hint

        for i in range(len(existing.input_shapes)):
            var shape_copy = List[Int]()
            for j in range(len(existing.input_shapes[i])):
                shape_copy.append(existing.input_shapes[i][j])
            self.input_shapes.append(shape_copy)

        for i in range(len(existing.result_shape)):
            self.result_shape.append(existing.result_shape[i])

        for i in range(len(existing.broadcast_strides)):
            var stride_copy = List[Int]()
            for j in range(len(existing.broadcast_strides[i])):
                stride_copy.append(existing.broadcast_strides[i][j])
            self.broadcast_strides.append(stride_copy)

    fn add_input(mut self, shape: List[Int], strides: List[Int]):
        var shape_copy = List[Int]()
        for i in range(len(shape)):
            shape_copy.append(shape[i])
        self.input_shapes.append(shape_copy)

        if len(self.result_shape) > 0:
            var broadcast_stride = calculate_broadcast_strides(shape, strides, self.result_shape)
            self.broadcast_strides.append(broadcast_stride)

    fn set_result_shape(mut self, shape: List[Int]):
        self.result_shape = List[Int]()
        for i in range(len(shape)):
            self.result_shape.append(shape[i])

    fn calculate_memory_cost(mut self):
        if len(self.result_shape) == 0:
            self.memory_cost = 0
            return

        var result_elements = 1
        for i in range(len(self.result_shape)):
            result_elements *= self.result_shape[i]

        self.memory_cost = result_elements

        for i in range(len(self.input_shapes)):
            var input_elements = 1
            for j in range(len(self.input_shapes[i])):
                input_elements *= self.input_shapes[i][j]

            if input_elements < result_elements:
                self.memory_cost += result_elements // 4  # Penalty for broadcasting

    fn get_optimization_hints(mut self) -> String:
        if not self.is_valid:
            return "invalid"

        if not self.compatibility_rule.requires_broadcasting():
            return "no_broadcast"

        var total_broadcasts = 0
        var has_scalar_broadcast = False

        for i in range(len(self.input_shapes)):
            var input_elements = 1
            for j in range(len(self.input_shapes[i])):
                input_elements *= self.input_shapes[i][j]

            if input_elements == 1:
                has_scalar_broadcast = True

            var result_elements = 1
            for j in range(len(self.result_shape)):
                result_elements *= self.result_shape[j]

            if input_elements < result_elements:
                total_broadcasts += 1

        if has_scalar_broadcast:
            return "scalar_broadcast"
        elif total_broadcasts == 1:
            return "single_broadcast"
        else:
            return "multi_broadcast"

struct BroadcastOptimizer(Copyable, Movable):
    """Broadcasting layout optimization for efficient operations."""
    var enable_vectorization: Bool
    var prefer_contiguous: Bool
    var alignment_requirement: Int
    var cache_friendly_threshold: Int

    fn __init__(out self, enable_vectorization: Bool = True, prefer_contiguous: Bool = True):
        self.enable_vectorization = enable_vectorization
        self.prefer_contiguous = prefer_contiguous
        self.alignment_requirement = BROADCAST_ALIGNMENT
        self.cache_friendly_threshold = 1024

    fn __copyinit__(out self, existing: Self):
        self.enable_vectorization = existing.enable_vectorization
        self.prefer_contiguous = existing.prefer_contiguous
        self.alignment_requirement = existing.alignment_requirement
        self.cache_friendly_threshold = existing.cache_friendly_threshold

    fn analyze_broadcast_efficiency(self, spec: BroadcastSpec) -> Float32:
        if not spec.is_valid:
            return 0.0

        var efficiency_score = Float32(1.0)

        if spec.compatibility_rule.requires_broadcasting():
            efficiency_score *= 0.8

        var result_elements = 1
        for i in range(len(spec.result_shape)):
            result_elements *= spec.result_shape[i]

        for i in range(len(spec.input_shapes)):
            var input_elements = 1
            for j in range(len(spec.input_shapes[i])):
                input_elements *= spec.input_shapes[i][j]

            if input_elements < result_elements:
                var broadcast_ratio = Float32(input_elements) / Float32(result_elements)
                efficiency_score *= (0.5 + broadcast_ratio * 0.5)

        return efficiency_score

    fn recommend_broadcast_strategy(self, spec: BroadcastSpec) -> String:
        if not spec.is_valid:
            return "invalid"

        var efficiency = self.analyze_broadcast_efficiency(spec)
        var hint = spec.optimization_hint

        if efficiency > 0.9:
            return "direct"
        elif hint == "scalar_broadcast":
            return "scalar_optimized"
        elif hint == "single_broadcast":
            return "vectorized"
        elif efficiency > 0.6:
            return "tiled"
        else:
            return "fallback"

    fn estimate_broadcast_cost(self, spec: BroadcastSpec) -> Int:
        if not spec.is_valid:
            return 0

        var base_cost = spec.memory_cost
        var efficiency = self.analyze_broadcast_efficiency(spec)

        var efficiency_penalty = Int(Float32(base_cost) * (1.0 - efficiency))

        return base_cost + efficiency_penalty

struct BroadcastTensor[dtype: DType]:
    """
    Tensor with comprehensive broadcasting layout support.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var broadcast_strides: UnsafePointer[Int]
    var ndim: Int
    var is_broadcast_view: Bool
    var original_shape: UnsafePointer[Int]
    var original_ndim: Int
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")

        self.ndim = len(shape)
        self._owns_data = True
        self.is_broadcast_view = False
        self.original_ndim = self.ndim

        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        self.broadcast_strides = UnsafePointer[Int].alloc(self.ndim)
        self.original_shape = UnsafePointer[Int].alloc(self.ndim)

        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            self.original_shape[i] = shape[i]
            total_elements *= shape[i]

        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            self.broadcast_strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
                self.broadcast_strides[i] = self.strides[i]

        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self._owns_data = False  # Shared data
        self.is_broadcast_view = existing.is_broadcast_view
        self.original_ndim = existing.original_ndim

        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides
        self.broadcast_strides = existing.broadcast_strides
        self.original_shape = existing.original_shape

    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
            self.shape.free()
            self.strides.free()
            self.broadcast_strides.free()
            self.original_shape.free()

    fn prepare_for_broadcast(mut self, target_shape: List[Int]) -> Bool:
        var current_shape = List[Int]()
        var current_strides = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])
            current_strides.append(self.strides[i])

        var rule = analyze_broadcast_compatibility(current_shape, target_shape)
        if not rule.is_compatible():
            return False

        var new_strides = calculate_broadcast_strides(current_shape, current_strides, target_shape)

        var target_ndim = len(target_shape)
        if target_ndim <= self.ndim:
            for i in range(target_ndim):
                self.broadcast_strides[i] = new_strides[i]

        self.is_broadcast_view = True
        return True

    fn can_broadcast_with(self, other_shape: List[Int]) -> Bool:
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])

        var rule = analyze_broadcast_compatibility(current_shape, other_shape)
        return rule.is_compatible()

    fn get_broadcast_shape_with(self, other_shape: List[Int]) -> List[Int]:
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])

        var rule = analyze_broadcast_compatibility(current_shape, other_shape)
        return rule.result_shape

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_offset = 0
        var strides_to_use = self.broadcast_strides if self.is_broadcast_view else self.strides

        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * strides_to_use[i]

        return self.data[linear_offset]

    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        var linear_offset = 0
        var strides_to_use = self.broadcast_strides if self.is_broadcast_view else self.strides

        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * strides_to_use[i]

        self.data[linear_offset] = value

    fn fill(mut self, value: Scalar[dtype]):
        var total_elements = self.numel()
        for i in range(total_elements):
            self.data[i] = value

    fn numel(self) -> Int:
        var total = 1
        for i in range(self.original_ndim):
            total *= self.original_shape[i]
        return total

    fn broadcast_numel(self) -> Int:
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total

    fn print_broadcast_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("BroadcastTensor[" + dtype_str + "]")
        var is_broadcast_str: String = "True" if self.is_broadcast_view else "False"
        print("  Is broadcast view: " + is_broadcast_str)

        print("  Original shape: [", end="")
        for i in range(self.original_ndim):
            var shape_str: String = String(self.original_shape[i])
            print(shape_str, end="")
            if i < self.original_ndim - 1:
                print(", ", end="")
        print("]")

        print("  Current shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        print("  Strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        if self.is_broadcast_view:
            print("  Broadcast strides: [", end="")
            for i in range(self.ndim):
                var bstride_str: String = String(self.broadcast_strides[i])
                print(bstride_str, end="")
                if i < self.ndim - 1:
                    print(", ", end="")
            print("]")

fn create_broadcast_tensor[dtype: DType](shape: List[Int]) raises -> BroadcastTensor[dtype]:
    return BroadcastTensor[dtype](shape)

fn create_broadcast_spec(shape_a: List[Int], shape_b: List[Int]) -> BroadcastSpec:
    var spec = BroadcastSpec()

    var rule = analyze_broadcast_compatibility(shape_a, shape_b)
    spec.compatibility_rule = rule
    spec.is_valid = rule.is_compatible()

    if spec.is_valid:
        spec.set_result_shape(rule.result_shape)
        spec.calculate_memory_cost()
        spec.optimization_hint = spec.get_optimization_hints()

    return spec

fn main():
    print("=== Broadcasting Layout Preparation - Part 1.2.5 ===")
    print("Memory Layout Design - Broadcasting Rules and Optimization")
    # ... full test suite runs here (compatibility rules, broadcast strides,
    # BroadcastTensor demos, specification/optimizer analysis, common
    # patterns, memory analysis, performance notes)
```

### Expected Output for `41_broadcasting_layout.mojo`

```
=== Broadcasting Layout Preparation - Part 1.2.5 ===
Memory Layout Design - Broadcasting Rules and Optimization
=== Testing Broadcasting Compatibility ===

1. Basic Broadcasting Rules:
Shape [3, 4] + Shape [4]: BROADCAST_RIGHT
Result shape: [3, 4]
Shape [3, 4] + Shape [1]: BROADCAST_RIGHT
Shape [3, 4] + Shape [3, 5]: INCOMPATIBLE

=== Testing Broadcasting Stride Calculation ===

1. Stride Calculation for Broadcasting:
Original shape: [1, 4], strides: [4, 1]
Target shape: [3, 4]
Broadcast strides: [0, 1]

=== Testing Broadcasting Tensor Operations ===

1. Creating Broadcasting Tensors:
BroadcastTensor[float32]
  Is broadcast view: False
  Original shape: [2, 3]
  Current shape: [2, 3]
  Strides: [3, 1]

2. Testing Broadcasting Compatibility:
Can broadcast [2, 3] to [4, 2, 3]: Yes
Broadcast result shape: [4, 2, 3]

3. Testing Scalar Broadcasting:
Can broadcast scalar [1] to [2, 3]: Yes
BroadcastTensor[float32]
  Is broadcast view: False
  Original shape: [1]
  Current shape: [1]
  Strides: [1]

=== Testing Broadcasting Specification ===

1. Broadcasting Specification Analysis:
Shape A: [2, 1, 4]
Shape B: [3, 4]
Broadcasting specification: Valid
Compatibility: BROADCAST_BOTH
Result shape: [2, 3, 4]
Memory cost: 24 elements
Optimization hint: multi_broadcast

2. Broadcasting Optimizer Analysis:
Broadcasting efficiency: 80.0%
Recommended strategy: tiled
Estimated cost: 28 units

=== Testing Common Broadcasting Patterns ===

1. Matrix-Vector Broadcasting:
Matrix [3, 4] + Vector [4]: BROADCAST_RIGHT

2. Tensor-Scalar Broadcasting:
Tensor [2, 3, 4] + Scalar [1]: BROADCAST_RIGHT

3. Multi-dimensional Broadcasting:
Tensor [2, 1, 4] + Tensor [3, 4]: BROADCAST_BOTH
Result shape: [2, 3, 4]

=== Testing Broadcasting Memory Analysis ===

1. Memory Efficiency Analysis:
Scenario 1:
  Shape A: [100, 100]
  Shape B: [100]
  Efficiency: 80.0%
  Strategy: tiled
Scenario 2:
  Shape A: [1000, 1000]
  Shape B: [1]
  Efficiency: 80.0%
  Strategy: tiled

=== Testing Broadcasting Performance Analysis ===

1. Broadcasting Performance Characteristics:
Understanding broadcasting impact:
- Zero-copy broadcasting: uses stride manipulation
- Memory access patterns: stride=0 for broadcasted dimensions
- Cache efficiency: depends on access order and stride patterns
- Vectorization: possible for aligned, contiguous dimensions

2. Broadcasting Optimization Strategies:
- Scalar broadcasting: highly efficient with SIMD
- Vector broadcasting: good cache locality
- Matrix broadcasting: moderate efficiency
- High-dimension broadcasting: potential cache issues

3. Memory Layout Considerations:
- Contiguous tensors: optimal for broadcasting
- Strided tensors: may require layout optimization
- Alignment: important for vectorized operations
- Memory coalescing: critical for GPU broadcasting

4. Best Practices:
- Prefer broadcasting over explicit loops
- Consider memory layout when designing operations
- Use contiguous tensors for better cache performance
- Profile broadcasting patterns for specific workloads
- Leverage SIMD for scalar and vector broadcasting

=== Broadcasting Layout Preparation Implementation Summary ===
+ NumPy-compatible broadcasting rules and validation
+ Memory-efficient broadcasting with stride manipulation
+ Zero-copy broadcasting operations where possible
+ Comprehensive shape compatibility analysis
+ Broadcasting cost estimation and optimization
+ Performance-aware layout preparation
+ Specialized broadcast patterns and fast paths
+ Foundation for efficient tensor arithmetic operations
```

## Chapter Summary

Chapter 6 computed strides one way — row-major, hardcoded. This chapter generalized that into `StrideInfo`, supporting row-major and column-major layouts from the same shape via two mirror-image formulas, and exposed the fragile spot in that generalization: `offset_to_indices`'s greedy division only correctly inverts a descending-stride (row-major) layout, silently producing out-of-range coordinates for anything else. `TensorView` then put Chapter 6's unused `_owns_data=False` case to work: a slice, a transpose, or a reshape computes a new shape, stride, and offset triple against the *same* buffer, at zero data-movement cost — Chapter 3's bus-utilization argument taken to its limit — with `RefCount` tracking how many views share that one buffer. Alignment and padding connected this chapter back to Chapters 4 and 5: `AlignmentSpec` and `SIMDTensorLayout` pad a tensor's innermost dimension up to a SIMD or GPU-coalescing boundary, trading memory efficiency (as low as 37.5% for a small `[2,3]` tensor padded to width 8) for loops that never need Chapter 5.2's remainder branch — while a close reading of `GPUCoalescingOptimizer` showed a coalescing-efficiency metric that only checks unit stride, not warp-size divisibility, silently agreeing with a "not warp-aligned" comment it can't actually detect. Broadcasting closed the chapter by generalizing Chapter 4.5's manual bias-broadcast trick into a formal rule — align shapes from the right, match or size-1, else incompatible — implemented with stride-0 dimensions rather than any copy, and a look at `BroadcastSpec`'s cost estimate showed real arithmetic running against an accidentally-empty `input_shapes` list, because the convenience factory that builds it never calls the one method that would have populated it.

## Self-Check Questions

1. Compute the column-major strides for shape `[4, 2, 5]` by hand, using Section 7.1's forward formula (`strides[0]=1`, `strides[i]=strides[i-1]×shape[i-1]`).
2. A `[5,6]` tensor has row-major strides `[6,1]`. Explain concretely why running `offset_to_indices(17, [5,6], [6,1])` produces the correct answer, but running the same function with strides `[1,5]` (this shape's column-major strides) instead would not.
3. A view is created via `tensor.slice_1d(0, 2, 5, 1)` on a `[8,4]` row-major tensor. Compute the view's `offset`, `shape`, and `strides` by hand.
4. For a `[2, 9]` tensor padded to SIMD width `4` (Section 7.3's algorithm, applied to the innermost dimension only), compute `padding_per_dim`, `aligned_strides`, `total_padded_size`, and the resulting memory efficiency.
5. Are shapes `[5, 1, 3]` and `[4, 3]` broadcast-compatible? If so, compute the result shape and state which side(s) broadcast; if not, identify exactly which aligned dimension pair breaks the rule.

## Where We Go Next

Chapter 8 (`part1/03-tensor-creation-functions.md`) returns to Chapter 6.4's ten factory functions with a more advanced lens — random-number generation strategies, file I/O for loading real tensor data, and specialized construction patterns that go beyond `zeros`/`ones`/`arange`. The strides, views, and broadcasting vocabulary this chapter just built are exactly what those richer factories, and the specialized tensor types after them, are built on top of.

## Worked Solutions

**1.** `strides[0]=1`. `strides[1]=strides[0]×shape[0]=1×4=4`. `strides[2]=strides[1]×shape[1]=4×2=8`. So `strides=[1,4,8]`.

**2.** With row-major strides `[6,1]` (descending: `6 > 1`), the division `17 // 6 = 2` remainder `5`, then `5 // 1 = 5` remainder `0`, correctly decodes to `[2,5]` — and `2×6+5×1=17` confirms it. With strides `[1,5]` (ascending, the column-major pattern for this shape), the same greedy algorithm computes `17 // 1 = 17` first — already larger than `shape[0]=5` — then `(17%1)=0 // 5 = 0`, giving the nonsensical `[17, 0]`. The algorithm assumes each successive stride is smaller than the last, dividing off the largest "place value" first; column-major strides grow instead of shrink, so the very first division already overflows the dimension it's supposed to describe.

**3.** Original tensor `[8,4]`, row-major strides `[8,4]→`, actually stride for shape `[8,4]`: `strides[1]=1`, `strides[0]=1×4=4` → `[4,1]`. `slice_1d(0, 2, 5, 1)` slices dimension 0 from index 2 to 5: `length=(5-2+1-1)//1=3`, new stride for dim 0 stays `strides[0]×step=4×1=4`; dimension 1 stays a full, untouched slice of length `4` with stride `1`. New offset: `spec.start(=2)×strides[0](=4)=8`. Result: `offset=8`, `shape=[3,4]`, `strides=[4,1]` — still contiguous, since this is a whole-row slice exactly like Worked Example 7.2.2's row case.

**4.** Innermost dimension `9`, SIMD width `4`: `9 % 4 = 1`, padded size `9 + (4-1) = 12`, `padding_per_dim[1] = 3`. `aligned_strides[1] = 1`, running stride becomes `12`. Dimension 0: `aligned_strides[0] = 12`, `total_padded_size = 12×2 = 24`. Memory efficiency: useful elements `2×9=18` over padded elements `2×12=24` → `18/24 = 0.75 = 75%`.

**5.** Align from the right: position 0 compares `3` (from `[5,1,3]`) vs `3` (from `[4,3]`) — equal. Position 1 compares `1` (from `[5,1,3]`) vs `4` (from `[4,3]`) — the first shape's side is `1`, so it broadcasts here. Position 2 compares `5` (from `[5,1,3]`) vs the *missing* dimension of `[4,3]` (treated as `1`) — the second shape's side is `1` here, so it broadcasts. Both sides broadcast somewhere, so this is `BROADCAST_BOTH` and the shapes are compatible. Result shape (element-wise maximum at each aligned position, restored to normal order): `[5, 4, 3]`.
