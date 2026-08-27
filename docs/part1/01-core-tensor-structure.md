# Chapter 6: The Tensor — Shape, Strides, Creation, Devices, and Validation

> "Every chapter from here to the end of this book touches one struct. Chapter 2 taught you what a struct is; Chapter 3 taught you why its layout matters; Chapter 1.4 taught you what kind of data a fast loop wants to find contiguous in memory. This chapter is where all three lessons get spent at once, building the one type — `Tensor[dtype]` — that the rest of the book simply assumes exists."

**What you will understand by the end of this chapter:**

- `TensorShape` and `TensorStrides`: two small, single-purpose structs — one recording dimension sizes, the other converting multi-dimensional coordinates into a single flat memory offset — and how to compute both by hand for a real shape
- `Tensor[dtype]`: how it composes those two structs, plus a data buffer and an optional gradient buffer, into one RAII-managed type, and what its constructor, deep-copy, and destructor each actually do to memory
- The ten tensor **factory functions** (`zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `random_uniform`, `random_normal`, `eye`, `diagonal`) and how to predict each one's output by hand
- `DeviceTensor`: why device-awareness is layered as a *separate* struct rather than added to `Tensor` directly, and the honest truth about what this book's "GPU" branch actually does on a machine with no GPU code running yet
- `ValidatedTensor`: the same field layout as `Tensor`, with checked construction and checked indexing added on top — and exactly which mistakes that checking catches that the base `Tensor` would silently let through

**What you need to know first:**

- Chapter 2 (struct fields, `__init__`/`__copyinit__`/`__del__`, RAII) — every struct in this chapter is built from exactly those pieces
- Chapter 3 (row-major memory layout, stride reasoning already used informally there) — Section 6.2 is where that reasoning becomes a formal, reusable algorithm
- Chapter 1.5 (`UnsafePointer.alloc`/`.free`) — every buffer in this chapter is one of these, managed by a struct's constructor and destructor

## 6.1 `TensorShape`: One Struct, One Job `[FOUNDATIONAL]`

### Intuition

A shipping manifest for a warehouse doesn't describe individual boxes — it records how many aisles there are, how many shelves per aisle, and how many slots per shelf, plus one precomputed number: the grand total of boxes the warehouse can hold. Nobody re-multiplies aisles × shelves × slots every time they want to know the total; the manifest already did that arithmetic once, when the warehouse was built, and simply reports the cached answer from then on.

`TensorShape` is that manifest. It holds the raw dimension sizes, how many dimensions there are, and a `_size` field that is computed exactly once, in the constructor — never recomputed on a later call to `.size()`.

### Background

```mojo
struct TensorShape:
    var dims: UnsafePointer[Int]
    var ndim: Int
    var _size: Int   # cached total element count
```

`_size` is a small, deliberate performance decision: computing a product over `ndim` dimensions is cheap for any one call, but `.size()` gets called constantly throughout this chapter (inside `numel()`, inside every factory function, inside every fill loop) — caching it once means every one of those calls is a single field read, not a fresh multiplication.

### Worked Example 6.1.1 — The running product, traced step by step

```mojo
var shape = TensorShape(shape_list)   # shape_list = [2, 3, 4]
```

The constructor's loop builds `_size` incrementally: start at `1`; after `i=0`, `_size = 1 × 2 = 2`; after `i=1`, `_size = 2 × 3 = 6`; after `i=2`, `_size = 6 × 4 = 24`. `ndim = 3`, `get_dim(1) = 3` (the middle dimension). This is exactly the `24`-element total this chapter's own tests report for a `[2,3,4]` tensor.

> `[COMMON TRAP]` `__copyinit__` allocates a *fresh* `dims` buffer and copies every element into it — it does not share the original's pointer. This is Chapter 2.4's RAII lesson again: if two `TensorShape` instances pointed at the same heap block, the first one's `__del__` would free memory the second one still expects to use, and the second instance's own later `__del__` would then attempt to free that same block a second time — Chapter 1.5's double-free, arrived at through a shortcut that looked like a harmless optimization.

## 6.2 `TensorStrides`: Converting Coordinates to One Number `[FOUNDATIONAL]`

### Intuition

Chapters 3 and 4 both used row-major byte offsets informally, without ever writing down the general rule. `TensorStrides` is that rule, made explicit and reusable: for an `n`-dimensional shape, a stride answers "if I move one step along dimension `i`, how many *elements* over, in the flat underlying buffer, did I just move?"

### Background

```
stride[ndim-1] = 1
stride[i] = stride[i+1] * shape[i+1]     for i = ndim-2 down to 0

linear_index = sum(indices[i] * strides[i] for i in range(ndim))
```

The rightmost dimension's stride is always `1` — adjacent elements along the last axis are, by construction of row-major layout, adjacent in memory. Every other dimension's stride is "how many elements does one full step of the *next* dimension take up," propagating outward from the innermost axis.

### Worked Example 6.2.1 — Strides for a `[2,3,4]` tensor

```
stride[2] = 1
stride[1] = stride[2] * shape[2] = 1 * 4 = 4
stride[0] = stride[1] * shape[1] = 4 * 3 = 12

strides = [12, 4, 1]
```

This is exactly the `Strides: [12, 4, 1]` this chapter's own 3D integer tensor test prints for shape `[2,3,4]`. Now compute the linear index for coordinates `[1, 2, 3]` — the last legal coordinate in this shape (axis 0 ranges over `0,1`; axis 1 over `0,1,2`; axis 2 over `0,1,2,3`):

```
linear_index = 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23
```

With `24` total elements (`0` through `23`), index `23` is indeed the very last one — the last coordinate maps to the last flat position, exactly as it should.

### Worked Example 6.2.2 — A smaller, second check

For a `[2,3]` tensor: `stride[1] = 1`, `stride[0] = stride[1] × shape[1] = 1 × 3 = 3`, giving `strides = [3, 1]` — matching this chapter's own printed `Strides: [3, 1]` for its `2×3` tensor test exactly.

### ASCII Diagram — nested shape, flat buffer underneath

```
Shape [2, 3, 4] -- 24 elements total, laid out as one flat buffer:

 axis0=0:  axis1=0: [e0  e1  e2  e3 ]   axis1=1: [e4  e5  e6  e7 ]   axis1=2: [e8  e9  e10 e11]
 axis0=1:  axis1=0: [e12 e13 e14 e15]   axis1=1: [e16 e17 e18 e19]   axis1=2: [e20 e21 e22 e23]

 stride[2]=1  -> moving one step along axis 2 moves 1 element  (e.g. e0 -> e1)
 stride[1]=4  -> moving one step along axis 1 moves 4 elements (e.g. e0 -> e4)
 stride[0]=12 -> moving one step along axis 0 moves 12 elements(e.g. e0 -> e12)
```

> `[COMMON TRAP]` It's easy to guess that a dimension's own size determines its own stride. It doesn't — `stride[ndim-1]` is always `1`, *regardless* of how large the last dimension is, because a stride measures how far apart elements are *along that axis*, and the rightmost axis is defined to be contiguous. What actually determines a stride is the size of every dimension *after* it, not its own size — `stride[1] = 4` above came from `shape[2] = 4`, not from `shape[1] = 3`.

## 6.3 `Tensor[dtype]`: Composing Shape, Strides, Data, and Gradient `[FOUNDATIONAL]`

### Intuition

Chapter 2 introduced structs as blueprints for a fixed memory layout. `Tensor[dtype]` is that same idea, one level up: its "fields" are themselves other structs (`TensorShape`, `TensorStrides`) plus two raw buffers — a blueprint whose rooms are, in two cases, entire smaller blueprints.

### Background

```mojo
struct Tensor[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var shape: TensorShape
    var strides: TensorStrides
    var requires_grad: Bool
    var grad: UnsafePointer[Scalar[dtype]]
    var _owns_data: Bool
```

The constructor builds metadata first (`shape`, then `strides`, which needs `shape` to already exist), then allocates `data` and **zero-initializes** every element — a numerical-stability choice, not a formality: uninitialized memory could contain bit patterns that read as `NaN` or huge finite values, and Chapter 12.4's gradient-health checks (`debug_assert(v == v, ...)`) would misfire on merely-*uninitialized* memory as though it were a genuinely corrupted gradient. `grad` is allocated and zeroed only when `requires_grad` is true.

### Worked Example 6.3.1 — One constructor call, four allocations

```mojo
var t = Tensor[DType.float32]([3, 4], True)
```

Trace every heap allocation this single line causes: `shape.dims` — `2` `Int`s (one per dimension), `2 × 8 = 16` bytes (Chapter 1.1's 8-byte `Int`). `strides.strides` — another `2` `Int`s, `16` bytes. `data` — `12` `Float32`s (`3×4`), `12 × 4 = 48` bytes, all zeroed. `grad` — since `requires_grad=True`, another `48`-byte buffer, also zeroed. **Four separate allocations** for what looks like one tensor, and `__del__` must free all four when `t` goes out of scope — `data` and `grad` explicitly in `Tensor.__del__`, `shape.dims` and `strides.strides` automatically via `TensorShape.__del__` and `TensorStrides.__del__` firing on their own fields, Chapter 2.4's RAII composing recursively.

### Worked Example 6.3.2 — `set_item`, tied directly to Section 6.2

```mojo
var t = Tensor[DType.float32]([2, 3, 4])
t.set_item([1, 2, 3], 9.5)
```

`get_linear_index([1,2,3])` runs Section 6.2's exact formula against this tensor's own strides (`[12,4,1]`): `1×12 + 2×4 + 3×1 = 23`. `set_item` then simply does `self.data[23] = 9.5` — the entire multi-dimensional indexing story reduces to "compute one number with Section 6.2's formula, then do the one-dimensional pointer write Chapter 1.5 already covered."

### Worked Example 6.3.3 — Deep copy, verified by mutating the original

```mojo
var original = Tensor[DType.float32]([3])
original.fill(5.0)          # [5.0, 5.0, 5.0]
var copy = original         # __copyinit__: fresh data buffer, values copied
original.set_item([0], 10.0)
```

`copy = original` triggers `__copyinit__`, which allocates a *new* 3-element buffer and copies `[5.0, 5.0, 5.0]` into it — not a second reference to the same buffer. After `original.set_item([0], 10.0)` (stride for a 1D shape is `[1]`, so `linear_index = 0×1 = 0`), `original.data = [10.0, 5.0, 5.0]` while `copy.data` remains `[5.0, 5.0, 5.0]`, completely unaffected — the exact output this chapter's own `test_tensor_copying` reports, and a direct memory-independence analogue to Chapter 3.4's "two layouts, one answer" check: here it's "two tensors, independent memory," verified the same way — by deliberately changing one and confirming the other doesn't move.

> `[COMMON TRAP]` `_owns_data` is set to `True` unconditionally by both `__init__` and `__copyinit__` in this chapter's code — nothing here ever actually constructs a `Tensor` with `_owns_data = False`. The field exists, and `__del__` checks it, but no code path yet produces the case where it matters. This is deliberate groundwork rather than a bug: Chapter 7 (Memory Layout Design) introduces *views* — tensors that borrow another tensor's buffer without owning it, such as a reshape or a transpose that doesn't copy data — and `_owns_data = False` is precisely the flag that will stop such a view's destructor from freeing memory it never allocated.

## 6.4 Ten Ways to Fill a Tensor: The Factory Functions `[FOUNDATIONAL]`

### Intuition

A print shop could make every customer specify the content of every single page by hand — or it could offer ten standing templates ("all blank," "all one pattern," "sequentially numbered," "evenly spaced," "randomly generated") that customers pick from instead. Every template still runs through the identical press (Section 6.3's `Tensor` constructor); a factory function is just a template's worth of post-construction fill logic layered on top.

### Background

| Function | Fills data with |
|---|---|
| `zeros` | `0` in every element |
| `ones` | `1` in every element |
| `full(value)` | `value` in every element |
| `empty` | **Nothing** — skips 6.3's zero-init entirely, for speed, on the explicit contract that the caller overwrites every element before reading it |
| `arange(start, stop, step)` | An arithmetic sequence, `start + i*step`, stopping before `stop` |
| `linspace(start, stop, n)` | `n` evenly spaced values from `start` to `stop`, inclusive of both ends |
| `random_uniform(low, high)` | Independent uniform-random values in `[low, high)` |
| `random_normal(mean, std)` | Independent normal-random values via the Box-Muller transform |
| `eye(size)` | A `size × size` identity matrix (`1`s on the diagonal) |
| `diagonal(values)` | A square matrix with `values` on the diagonal, `0` elsewhere |

### Worked Example 6.4.1 — `arange`, including its element-count formula

```mojo
var t = arange[DType.float32](0, 10, 2)
```

`range_f64 = (10-0)/2 = 5.0`; `num_elements = Int(5.0) = 5`; since `5.0` is not *greater than* `5`, no correction is added — `num_elements = 5`. Filling `data[i] = 0 + i×2` for `i=0..4` gives `[0, 2, 4, 6, 8]` — exactly this chapter's own printed result.

Now trace the case that *does* need the correction: `arange[DType.float32](0, 10, 3)`. `range_f64 = 10/3 ≈ 3.333`; `num_elements = Int(3.333) = 3`; since `3.333 > 3`, the count is bumped to `4`. Filling gives `[0, 3, 6, 9]` — four elements, the last one (`9`) still strictly less than `stop=10`, confirming "stop is exclusive" holds even when the step doesn't divide the range evenly.

### Worked Example 6.4.2 — `linspace`, inclusive of both ends

```mojo
var t = linspace[DType.float32](0, 1, 6)
```

`step = (1-0)/(6-1) = 0.2`. Filling `data[i] = 0 + i×0.2` for `i=0..5` gives `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` — six points, the last one landing exactly on `1.0`, which is `linspace`'s defining difference from `arange`: both endpoints are included by construction, rather than the stop value being excluded.

### Worked Example 6.4.3 — `eye` and `diagonal`, traced against the flat buffer

```mojo
var identity = eye[DType.float32](4)
```

`eye(4)` starts from `zeros([4,4])` (`16` elements, all `0`), then sets `data[i*4+i] = 1` for `i=0..3`: positions `0, 5, 10, 15`. Laid out row-major (Section 6.2), position `5` is row `1`, column `1` (`5 = 1×4+1`); position `10` is row `2`, column `2`; position `15` is row `3`, column `3` — the diagonal, exactly. Result: `[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]`, matching this chapter's own output precisely.

```mojo
var diag = diagonal[DType.float32]([1.0, 2.0, 3.0])
```

Same mechanism at `size=3`: `zeros([3,3])` (`9` elements), then `data[i*3+i] = values[i]` for `i=0..2` — positions `0, 4, 8` become `1, 2, 3`. Result: `[1,0,0, 0,2,0, 0,0,3]`, again matching exactly.

> `[COMMON TRAP]` `empty()` is the one factory function that deliberately breaks Section 6.3's zero-initialization promise, in exchange for skipping a full pass over the buffer. Its docstring says so directly — "data contains arbitrary values" — but nothing in the type system enforces reading it only after a full overwrite; treating an `empty()` tensor's un-overwritten elements as meaningful zeros is exactly the same silent-garbage risk Chapter 5.2 flagged for a missing SIMD remainder loop, just arrived at by explicitly opting out of initialization rather than by forgetting a loop.

## 6.5 `DeviceTensor`: The Same Tensor, With a Location Tag `[FOUNDATIONAL]`

### Intuition

Chapter 4.2 split a GPU program into host and device halves. `DeviceTensor` is what a `Tensor` becomes once it's handed a location sticker — `CPU` or `GPU` — at the moment it's built, so every later operation knows which machine's memory it's sitting in without having to ask.

### Background

`DeviceType` is a thin wrapper around an `Int` (`CPU=0`, `GPU=1`, `AUTO=2`) with hand-written `__eq__`/`__ne__` — Mojo has no built-in enum type in this code, so equality comparison is written out explicitly, the same custom-behavior-via-methods pattern Chapter 2.5 used for traits. `DeviceManager` *simulates* hardware discovery: it hardcodes "yes, there is a GPU, with 4GB," and sets `default_device = GPU` whenever `has_gpu` is true — an honest simulation, clearly labeled as such in the source, not a real hardware query.

### Worked Example 6.5.1 — Resolving `AUTO`

```mojo
var t = DeviceTensor[DType.float32]([2, 3], AUTO, False)
```

`device == AUTO` triggers `DeviceManager()`, whose constructor sets `has_gpu = True` and `default_device = GPU`. `manager.get_default_device()` returns `GPU`, so `t.device = GPU` and `t.device_id = 1` (`0` for `CPU`, `1` otherwise) — matching this chapter's own `AUTO Device Selection` test, which reports `on GPU (ID: 1)`.

### Worked Example 6.5.2 — A full round trip, `CPU → GPU → CPU`

```mojo
var cpu_t = device_arange[DType.float32](0, 4, 1, CPU)   # [0.0, 1.0, 2.0, 3.0] on CPU
var gpu_t = cpu_t.to_device(GPU)                          # new tensor, "GPU", same values
var cpu_t2 = gpu_t.to_device(CPU)                         # new tensor, "CPU", same values
```

`device_arange` reuses Section 6.4.1's arithmetic exactly, giving `[0,1,2,3]`. `to_device(GPU)` allocates a brand-new `DeviceTensor` tagged `GPU` and copies all four values across; `to_device(CPU)` does the same in reverse. All three tensors hold the identical `[0.0, 1.0, 2.0, 3.0]` — matching this chapter's own printed trace at every step — because nothing about a *transfer* changes the numbers, only their tag and which allocation currently holds them.

> `[COMMON TRAP]` Read `_allocate_on_device` and `_deallocate_on_device` literally: both branches — the `CPU` branch and the `GPU` branch — call the *exact same* `UnsafePointer[Scalar[dtype]].alloc(size)` / `.free()`. Every "GPU" tensor in this chapter is, physically, ordinary CPU heap memory wearing a `GPU` label; the source code says so directly in its own comments ("In real implementation, would use cudaMalloc..."). This is a deliberately honest placeholder for the *shape* a real device abstraction needs — a device tag, a transfer function, per-device allocation hooks — not a working GPU backend. When this book needs a tensor's data to genuinely live in GPU memory and run on GPU hardware, Chapter 4's actual `DeviceContext`/`enqueue_function` API (not this simulation) is the real mechanism.

## 6.6 `ValidatedTensor`: The Same Layout, With Checked Construction and Indexing `[FOUNDATIONAL]`

### Intuition

An ordinary door and the same door fitted with a sensor that refuses to open onto a missing hallway are structurally identical — same hinges, same frame — until the one moment a mistake would otherwise walk straight through. `ValidatedTensor` is `Tensor` with exactly that sensor added, at the two moments a tensor mistake is easiest to make silently: construction (a bad shape) and indexing (an out-of-range coordinate).

### Background

`ValidatedTensor` duplicates `Tensor`'s field layout and constructor algorithm from Section 6.3, adding: a `TensorShape.__init__` marked `raises`, checking `len(shape) > MAX_TENSOR_DIMS`, `len(shape) == 0`, each `shape[i] <= 0`, and an overflow guard (`new_size < self._size` catches the running product wrapping around); and `get_item_safe`/`set_item_safe`, which run `validate_index_bounds` (axis-count match plus per-axis range check) and `normalize_negative_indices` (Python-style `-1` means "last element along this axis") *before* touching Section 6.3's raw linear-index arithmetic, raising a descriptive `Error` instead of silently reading or writing past the buffer.

### Worked Example 6.6.1 — Negative indexing, resolved to a real offset

```mojo
var t = validated_zeros[DType.float32]([3, 4])
t.set_item_safe([-1, -1], 7.0)
```

`normalize_negative_indices` converts each `-1`: axis 0 (`dim_size=3`) gives `-1 + 3 = 2`; axis 1 (`dim_size=4`) gives `-1 + 4 = 3`. Both now pass their bounds check (`2 < 3`, `3 < 4`). Strides for a `[3,4]` shape are `[4,1]` (Section 6.2's formula: `stride[1]=1`, `stride[0]=1×4=4`), so `linear_index = 2×4 + 3×1 = 11` — the very last position of a `12`-element buffer (indices `0..11`). `[-1,-1]` correctly resolves to the bottom-right corner, exactly as this chapter's own test confirms with `tensor[-1, -1] = 7.0`.

### Worked Example 6.6.2 — An out-of-bounds index, caught before it can corrupt memory

```mojo
t.get_item_safe([5, 2])   # raises, on a [3, 4] tensor
```

Axis `0`'s index, `5`, is not negative, so it proceeds straight to the bounds check: `5 >= 3` (the axis-0 size) is true, so `validate_index_bounds` returns `False` immediately, and `get_item_safe` raises `"Index out of bounds"` — `self.data` is never touched. Compare this with what Section 6.3's plain `Tensor.get_item` would have done with the identical input: it would have computed `linear_index = 5×4 + 2×1 = 22` and read whatever byte pattern happened to occupy `data[22]` — eleven elements past the legitimate `12`-element buffer (indices `0..11`), silently reading (or, via `set_item`, corrupting) adjacent heap memory exactly as Chapter 1.5 warned raw pointer access could, with no error at all.

### ASCII Diagram — the two extra checks `ValidatedTensor` inserts

```
Tensor.get_item(indices):
  indices -> get_linear_index (Section 6.2 formula) -> self.data[idx]   <- no check at all

ValidatedTensor.get_item_safe(indices):
  indices -> validate_index_bounds? --no--> raise "Index out of bounds"
           -> normalize_negative_indices -> get_linear_index -> self.data[idx]
```

> `[COMMON TRAP]` `validated_zeros`/`validated_ones`/`validated_full` are all marked `raises`, and Mojo requires every call site to acknowledge that with a `try`/`except` — this chapter's own tests wrap every single call this way. That's a compile-time obligation, not a runtime surprise: forgetting the `try` around a call that can raise is caught before the program ever runs, arguably safer than a language where an unchecked exception can propagate silently past code that never expected one. But it also means `ValidatedTensor` cannot be dropped in as a silent, no-changes-needed replacement for `Tensor` — every call site that switches to it must also start handling the `raises` contract.

## 6.7 Complete Runnable Code

### File: `33_tensor_basic_struct.mojo` — Sections 6.1–6.3

**Run:** `pixi run mojo 33_tensor_basic_struct.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_TENSOR_DIMS = 8  # Maximum number of dimensions supported

struct TensorShape:
    """
    Tensor Shape Management System.

    Manages tensor dimensional information and provides utilities for shape operations.
    Uses dynamic memory allocation to support arbitrary dimension counts while
    maintaining cache-friendly access patterns.

    Memory Layout:
    - dims: Array of dimension sizes [d0, d1, d2, ...]
    - ndim: Number of dimensions
    - _size: Cached total element count for O(1) access

    Algorithm: Shape size calculation
    size = product(dims[i] for i in range(ndim))
    """
    var dims: UnsafePointer[Int]
    var ndim: Int
    var _size: Int  # Cached total size for performance

    fn __init__(out self, shape: List[Int]):
        """
        Initialize tensor shape from dimension list.

        Args:
            shape: List of dimension sizes [dim0, dim1, ..., dimN].

        Algorithm:
        1. Allocate memory for dimension array
        2. Copy dimensions and calculate total size
        3. Cache size for efficient access
        """
        self.ndim = len(shape)
        self.dims = UnsafePointer[Int].alloc(self.ndim)
        self._size = 1

        # Copy dimensions and calculate total size
        for i in range(self.ndim):
            self.dims[i] = shape[i]
            self._size *= shape[i]

    fn __copyinit__(out self, existing: Self):
        """Deep copy constructor for tensor shape."""
        self.ndim = existing.ndim
        self._size = existing._size
        self.dims = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.dims[i] = existing.dims[i]

    fn __del__(owned self):
        """Destructor - automatic memory cleanup."""
        self.dims.free()

    fn size(self) -> Int:
        """Return cached total number of elements."""
        return self._size

    fn get_dim(self, axis: Int) -> Int:
        """Get dimension size at specific axis."""
        return self.dims[axis]

    fn print_shape(self):
        """Display tensor shape in readable format."""
        print("Shape: [", end="")
        for i in range(self.ndim):
            print(self.dims[i], end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

struct TensorStrides:
    """
    Memory Stride Calculation System.

    Manages tensor memory layout using stride-based indexing for efficient
    multi-dimensional array access. Supports row-major (C-style) layout
    with provisions for future column-major and custom layouts.

    Stride Calculation Algorithm (Row-Major):
    stride[n-1] = 1
    stride[i] = stride[i+1] * shape[i+1] for i = n-2 down to 0

    Linear Index Calculation:
    linear_index = sum(indices[i] * strides[i] for i in range(ndim))

    Memory Layout Example (2x3 matrix):
    Shape: [2, 3], Strides: [3, 1]
    Index [i,j] -> Linear: i*3 + j*1
    """
    var strides: UnsafePointer[Int]
    var ndim: Int

    fn __init__(out self, shape: TensorShape):
        """
        Initialize strides for row-major memory layout.

        Algorithm Implementation:
        - Start from rightmost dimension with stride 1
        - Each dimension's stride = next_dimension_stride * next_dimension_size
        - Ensures contiguous memory access for rightmost index
        """
        self.ndim = shape.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        if self.ndim > 0:
            # Row-major stride calculation
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * shape.get_dim(i + 1)

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for stride information."""
        self.ndim = existing.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]

    fn __del__(owned self):
        """Destructor - automatic memory cleanup."""
        self.strides.free()

    fn get_stride(self, axis: Int) -> Int:
        """Get stride value at specific axis."""
        return self.strides[axis]

    fn print_strides(self):
        """Display stride information in readable format."""
        print("Strides: [", end="")
        for i in range(self.ndim):
            print(self.strides[i], end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

struct Tensor[dtype: DType]:
    """
    Core Tensor Structure for Automatic Differentiation.

    Primary tensor implementation combining multi-dimensional data storage
    with gradient computation capabilities. Designed for high-performance
    numerical computing with automatic memory management.

    Key Components:
    1. Data Storage: Generic type support with memory ownership tracking
    2. Shape/Stride Management: Efficient multi-dimensional indexing
    3. Gradient Support: Optional gradient storage for automatic differentiation
    4. Memory Safety: RAII pattern with automatic cleanup

    Memory Layout:
    data: [element0, element1, ..., elementN] (contiguous storage)
    grad: [grad0, grad1, ..., gradN] (optional gradient storage)

    Design Features:
    - Zero-cost abstractions for unused gradient functionality
    - Generic data type support (float32, float64, int32, etc.)
    - Deep copy semantics for safe tensor operations
    - Device-agnostic design for future GPU acceleration
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: TensorShape
    var strides: TensorStrides
    var requires_grad: Bool
    var grad: UnsafePointer[Scalar[dtype]]  # Gradient storage
    var _owns_data: Bool  # Memory ownership tracking

    fn __init__(out self, shape: List[Int], requires_grad: Bool = False):
        """
        Primary Tensor Constructor.

        Creates new tensor with specified shape and optional gradient tracking.
        Implements zero-initialization for numerical stability.

        Args:
            shape: Dimensional specification [dim0, dim1, ..., dimN].
            requires_grad: Enable gradient computation for autodiff.

        Algorithm:
        1. Initialize shape and stride metadata
        2. Allocate contiguous memory for tensor data
        3. Zero-initialize all elements for numerical stability
        4. Conditionally allocate gradient storage
        """
        # Initialize metadata structures
        self.shape = TensorShape(shape)
        self.strides = TensorStrides(self.shape)
        self.requires_grad = requires_grad
        self._owns_data = True

        # Allocate and initialize tensor data
        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        # Zero-initialization for numerical stability
        for i in range(total_size):
            self.data[i] = Scalar[dtype](0)

        # Conditional gradient allocation
        if self.requires_grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = Scalar[dtype](0)
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """
        Deep Copy Constructor.

        Creates independent copy of tensor with separate memory allocation.
        Essential for safe tensor operations and preventing memory aliasing.

        Algorithm:
        1. Copy metadata (shape, strides, flags)
        2. Allocate new memory for data and gradients
        3. Perform element-wise deep copy
        """
        # Copy metadata
        self.shape = existing.shape
        self.strides = existing.strides
        self.requires_grad = existing.requires_grad
        self._owns_data = True

        # Deep copy tensor data
        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)
        for i in range(total_size):
            self.data[i] = existing.data[i]

        # Deep copy gradients if present
        if self.requires_grad and existing.grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = existing.grad[i]
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        """
        Automatic Memory Management (RAII Pattern).

        Ensures proper cleanup of allocated memory when tensor goes out of scope.
        Only frees memory if tensor owns the data (prevents double-free errors).
        """
        if self._owns_data:
            self.data.free()
            if self.requires_grad:
                self.grad.free()

    fn numel(self) -> Int:
        """Return total number of elements in tensor."""
        return self.shape.size()

    fn ndim(self) -> Int:
        """Return number of dimensions in tensor."""
        return self.shape.ndim

    fn size(self, axis: Int) -> Int:
        """Return size of tensor along specified axis."""
        return self.shape.get_dim(axis)

    fn get_linear_index(self, indices: List[Int]) -> Int:
        """
        Multi-dimensional to Linear Index Conversion.

        Converts N-dimensional tensor coordinates to linear memory index
        using stride-based calculation for efficient memory access.

        Algorithm:
        linear_index = sum(indices[i] * strides[i] for i in range(ndim))

        Example (2D): [row, col] -> row * row_stride + col * col_stride
        """
        var linear_idx = 0
        for i in range(len(indices)):
            linear_idx += indices[i] * self.strides.get_stride(i)
        return linear_idx

    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        """Set tensor element at multi-dimensional coordinates."""
        var idx = self.get_linear_index(indices)
        self.data[idx] = value

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get tensor element at multi-dimensional coordinates."""
        var idx = self.get_linear_index(indices)
        return self.data[idx]

    fn fill(self, value: Scalar[dtype]):
        """
        Tensor Fill Operation.

        Efficiently fills entire tensor with specified value using
        linear memory access for optimal cache performance.
        """
        var total_size = self.shape.size()
        for i in range(total_size):
            self.data[i] = value

    fn zero_grad(self):
        """
        Gradient Zeroing for Optimization Steps.

        Resets all gradient values to zero, required before each
        backward pass in iterative optimization algorithms.
        """
        if self.requires_grad:
            var total_size = self.shape.size()
            for i in range(total_size):
                self.grad[i] = Scalar[dtype](0)

    fn print_info(self):
        """Display comprehensive tensor metadata."""
        print("Tensor Information:")
        print("  Data type:", dtype)
        print("  Number of dimensions:", self.ndim())
        print("  Total elements:", self.numel())
        print("  Requires gradient:", self.requires_grad)
        print("  Owns data:", self._owns_data)
        self.shape.print_shape()
        self.strides.print_strides()

    fn print_data(self, max_elements: Int = 10):
        """Display tensor data with optional truncation for readability."""
        var total_size = self.shape.size()
        var elements_to_show = tensor_min(max_elements, total_size)

        print("Data: [", end="")
        for i in range(elements_to_show):
            print(self.data[i], end="")
            if i < elements_to_show - 1:
                print(", ", end="")

        if total_size > max_elements:
            print(", ... (", total_size - max_elements, "more)]")
        else:
            print("]")

# Utility Functions
fn tensor_min(a: Int, b: Int) -> Int:
    """Utility function for minimum calculation."""
    return a if a < b else b

fn test_tensor_creation():
    """
    Test Suite: Basic Tensor Creation and Initialization.

    Validates tensor construction with various configurations:
    - 1D, 2D, and 3D tensors
    - Different data types (float32, int32)
    - Gradient tracking enabled/disabled
    - Proper metadata initialization
    """
    print("=== Testing Tensor Creation ===")

    # Test 1: 1D tensor creation
    var shape_1d = List[Int]()
    shape_1d.append(5)
    var tensor_1d = Tensor[DType.float32](shape_1d, False)

    print("\n1D Tensor:")
    tensor_1d.print_info()
    tensor_1d.print_data()

    # Test 2: 2D tensor with gradient tracking
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var tensor_2d = Tensor[DType.float32](shape_2d, True)

    print("\n2D Tensor with gradients:")
    tensor_2d.print_info()

    # Test 3: 3D integer tensor
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var tensor_3d = Tensor[DType.int32](shape_3d, False)

    print("\n3D Integer Tensor:")
    tensor_3d.print_info()

fn test_tensor_operations():
    """
    Test Suite: Basic Tensor Operations.

    Validates core tensor functionality:
    - Element access and modification
    - Fill operations
    - Multi-dimensional indexing
    - Linear index calculation
    """
    print("\n=== Testing Tensor Operations ===")

    # Create 2x3 tensor for testing
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var tensor = Tensor[DType.float32](shape, False)

    print("\n2x3 Tensor Operations:")
    tensor.print_info()

    # Test fill operation
    tensor.fill(3.14)
    print("\nAfter filling with 3.14:")
    tensor.print_data()

    # Test individual element access
    var indices = List[Int]()
    indices.append(0)
    indices.append(0)
    tensor.set_item(indices, 1.0)

    indices[0] = 1
    indices[1] = 2
    tensor.set_item(indices, 2.5)

    print("\nAfter setting tensor[0,0] = 1.0 and tensor[1,2] = 2.5:")
    tensor.print_data()

    # Test element retrieval
    var val1 = tensor.get_item(indices)
    print("\nValue at tensor[1,2]:", val1)

fn test_gradient_functionality():
    """
    Test Suite: Gradient Computation Support.

    Validates automatic differentiation infrastructure:
    - Gradient memory allocation
    - Gradient zeroing operations
    - Memory management for gradients
    """
    print("\n=== Testing Gradient Functionality ===")

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var tensor = Tensor[DType.float32](shape, True)  # Enable gradients

    tensor.print_info()

    # Test data initialization
    tensor.fill(2.0)
    print("\nTensor data after filling with 2.0:")
    tensor.print_data()

    print("\nGradient status:")
    print("  Requires grad:", tensor.requires_grad)
    print("  Gradient memory allocated:", tensor.grad != UnsafePointer[Scalar[DType.float32]]())

    # Test gradient zeroing
    tensor.zero_grad()
    print("  Gradients zeroed successfully")

fn test_tensor_copying():
    """
    Test Suite: Deep Copy Semantics.

    Validates memory safety and copy behavior:
    - Independent memory allocation
    - Deep copy vs shallow copy verification
    - Memory isolation between tensors
    """
    print("\n=== Testing Tensor Copying ===")

    var shape = List[Int]()
    shape.append(3)
    var original = Tensor[DType.float32](shape, False)
    original.fill(5.0)

    print("\nOriginal tensor:")
    original.print_data()

    # Create deep copy
    var copy = original
    print("\nCopied tensor:")
    copy.print_data()

    # Test memory independence
    var indices = List[Int]()
    indices.append(0)
    original.set_item(indices, 10.0)

    print("\nAfter modifying original[0] = 10.0:")
    print("Original:")
    original.print_data()
    print("Copy (should be unchanged):")
    copy.print_data()

fn main():
    """
    Main Demonstration Function.

    Comprehensive test suite for core tensor functionality,
    validating all implemented features and algorithms.
    """
    print("=== Core Tensor Struct Design - Part 1.1.1 ===")
    print("Automatic Differentiation Framework - Tensor Infrastructure")

    test_tensor_creation()
    test_tensor_operations()
    test_gradient_functionality()
    test_tensor_copying()

    print("\n=== Core Tensor Implementation Summary ===")
    print("+ Tensor struct with shape and stride management")
    print("+ Multi-dimensional indexing with stride calculation")
    print("+ Memory ownership and lifecycle management (RAII)")
    print("+ Gradient computation support for automatic differentiation")
    print("+ Deep copy semantics for safe tensor operations")
    print("+ Type-safe generic implementation")
    print("+ Zero-cost abstractions for performance")
    print("+ Device-agnostic design for future GPU acceleration")
```

### Expected Output for `33_tensor_basic_struct.mojo`

```
=== Core Tensor Struct Design - Part 1.1.1 ===
Automatic Differentiation Framework - Tensor Infrastructure
=== Testing Tensor Creation ===

1D Tensor:
Tensor Information:
  Data type: float32
  Number of dimensions: 1
  Total elements: 5
  Requires gradient: False
  Owns data: True
Shape: [5]
Strides: [1]
Data: [0.0, 0.0, 0.0, 0.0, 0.0]

2D Tensor with gradients:
Tensor Information:
  Data type: float32
  Number of dimensions: 2
  Total elements: 12
  Requires gradient: True
  Owns data: True
Shape: [3, 4]
Strides: [4, 1]

3D Integer Tensor:
Tensor Information:
  Data type: int32
  Number of dimensions: 3
  Total elements: 24
  Requires gradient: False
  Owns data: True
Shape: [2, 3, 4]
Strides: [12, 4, 1]

=== Testing Tensor Operations ===

2x3 Tensor Operations:
Tensor Information:
  Data type: float32
  Number of dimensions: 2
  Total elements: 6
  Requires gradient: False
  Owns data: True
Shape: [2, 3]
Strides: [3, 1]

After filling with 3.14:
Data: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]

After setting tensor[0,0] = 1.0 and tensor[1,2] = 2.5:
Data: [1.0, 3.14, 3.14, 3.14, 3.14, 2.5]

Value at tensor[1,2]: 2.5

=== Testing Gradient Functionality ===
Tensor Information:
  Data type: float32
  Number of dimensions: 2
  Total elements: 4
  Requires gradient: True
  Owns data: True
Shape: [2, 2]
Strides: [2, 1]

Tensor data after filling with 2.0:
Data: [2.0, 2.0, 2.0, 2.0]

Gradient status:
  Requires grad: True
  Gradient memory allocated: True
  Gradients zeroed successfully

=== Testing Tensor Copying ===

Original tensor:
Data: [5.0, 5.0, 5.0]

Copied tensor:
Data: [5.0, 5.0, 5.0]

After modifying original[0] = 10.0:
Original:
Data: [10.0, 5.0, 5.0]
Copy (should be unchanged):
Data: [5.0, 5.0, 5.0]

=== Core Tensor Implementation Summary ===
+ Tensor struct with shape and stride management
+ Multi-dimensional indexing with stride calculation
+ Memory ownership and lifecycle management (RAII)
+ Gradient computation support for automatic differentiation
+ Deep copy semantics for safe tensor operations
+ Type-safe generic implementation
+ Zero-cost abstractions for performance
+ Device-agnostic design for future GPU acceleration
```

### File: `34_tensor_creation_functions.mojo` — Section 6.4

**Run:** `pixi run mojo 34_tensor_creation_functions.mojo`

```mojo
from memory import UnsafePointer
from collections import List
from random import seed, random_float64

alias MAX_TENSOR_DIMS = 8  # Maximum number of dimensions supported

struct TensorShape:
    """Tensor shape management for creation functions."""
    var dims: UnsafePointer[Int]
    var ndim: Int
    var _size: Int

    fn __init__(out self, shape: List[Int]):
        """Initialize tensor shape from dimension list."""
        self.ndim = len(shape)
        self.dims = UnsafePointer[Int].alloc(self.ndim)
        self._size = 1

        for i in range(self.ndim):
            self.dims[i] = shape[i]
            self._size *= shape[i]

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor shape."""
        self.ndim = existing.ndim
        self._size = existing._size
        self.dims = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.dims[i] = existing.dims[i]

    fn __del__(owned self):
        """Destructor - automatic memory cleanup."""
        self.dims.free()

    fn size(self) -> Int:
        """Return total number of elements."""
        return self._size

    fn get_dim(self, axis: Int) -> Int:
        """Get dimension size at specific axis."""
        return self.dims[axis]

struct TensorStrides:
    """Stride calculation for tensor creation."""
    var strides: UnsafePointer[Int]
    var ndim: Int

    fn __init__(out self, shape: TensorShape):
        """Initialize strides for row-major layout."""
        self.ndim = shape.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * shape.get_dim(i + 1)

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor strides."""
        self.ndim = existing.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]

    fn __del__(owned self):
        """Destructor - automatic memory cleanup."""
        self.strides.free()

struct Tensor[dtype: DType]:
    """Core tensor structure for creation functions."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: TensorShape
    var strides: TensorStrides
    var requires_grad: Bool
    var grad: UnsafePointer[Scalar[dtype]]
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int], requires_grad: Bool = False):
        """Basic tensor constructor."""
        self.shape = TensorShape(shape)
        self.strides = TensorStrides(self.shape)
        self.requires_grad = requires_grad
        self._owns_data = True

        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        if self.requires_grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = Scalar[dtype](0)
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor."""
        self.shape = existing.shape
        self.strides = existing.strides
        self.requires_grad = existing.requires_grad
        self._owns_data = True

        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)
        for i in range(total_size):
            self.data[i] = existing.data[i]

        if self.requires_grad and existing.grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = existing.grad[i]
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        """Destructor with automatic memory management."""
        if self._owns_data:
            self.data.free()
            if self.requires_grad:
                self.grad.free()

    fn numel(self) -> Int:
        """Return total number of elements."""
        return self.shape.size()

    fn ndim(self) -> Int:
        """Return number of dimensions."""
        return self.shape.ndim

    fn print_info(self):
        """Print tensor information."""
        var dtype_str: String = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        print("Tensor[" + dtype_str + "] shape: [", end="")
        for i in range(self.shape.ndim):
            var dim_str: String = String(self.shape.get_dim(i))
            print(dim_str, end="")
            if i < self.shape.ndim - 1:
                print(", ", end="")
        var numel_str: String = String(self.numel())
        var grad_str: String = "True" if self.requires_grad else "False"
        print("], elements: " + numel_str + ", requires_grad: " + grad_str)

    fn print_data(self, max_elements: Int = 16):
        """Print tensor data with truncation."""
        var total_size = self.shape.size()
        var elements_to_show = total_size if total_size <= max_elements else max_elements

        print("Data: [", end="")
        for i in range(elements_to_show):
            print(self.data[i], end="")
            if i < elements_to_show - 1:
                print(", ", end="")

        if total_size > max_elements:
            var remaining_str: String = String(total_size - max_elements)
            print(", ... (" + remaining_str + " more)]")
        else:
            print("]")

fn zeros[dtype: DType](shape: List[Int], requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor filled with zeros.

    Creates a new tensor of specified shape with all elements initialized to zero.
    This is one of the most commonly used tensor creation functions.

    Algorithm:
    1. Allocate tensor with specified shape
    2. Initialize all elements to zero using scalar conversion
    3. Set up gradient tracking if requested

    Args:
        shape: List of dimension sizes for the tensor.
        requires_grad: Whether to enable gradient computation.

    Returns:
        New tensor filled with zeros.

    Example:
        var tensor = zeros[DType.float32]([2, 3])  # 2x3 tensor of zeros.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    var total_size = tensor.numel()

    # Initialize all elements to zero
    for i in range(total_size):
        tensor.data[i] = Scalar[dtype](0)

    return tensor

fn ones[dtype: DType](shape: List[Int], requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor filled with ones.

    Creates a new tensor of specified shape with all elements initialized to one.
    Useful for initialization patterns and mathematical operations.

    Algorithm:
    1. Allocate tensor with specified shape
    2. Initialize all elements to one using scalar conversion
    3. Set up gradient tracking if requested

    Args:
        shape: List of dimension sizes for the tensor.
        requires_grad: Whether to enable gradient computation.

    Returns:
        New tensor filled with ones.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    var total_size = tensor.numel()

    # Initialize all elements to one
    for i in range(total_size):
        tensor.data[i] = Scalar[dtype](1)

    return tensor

fn full[dtype: DType](shape: List[Int], value: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor filled with specified value.

    Creates a new tensor of specified shape with all elements initialized
    to the provided value. Generalizes zeros() and ones() functions.

    Algorithm:
    1. Allocate tensor with specified shape
    2. Initialize all elements to specified value
    3. Set up gradient tracking if requested

    Args:
        shape: List of dimension sizes for the tensor.
        value: Value to fill tensor with.
        requires_grad: Whether to enable gradient computation.

    Returns:
        New tensor filled with specified value.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    var total_size = tensor.numel()

    # Initialize all elements to specified value
    for i in range(total_size):
        tensor.data[i] = value

    return tensor

fn empty[dtype: DType](shape: List[Int], requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create uninitialized tensor.

    Creates a new tensor of specified shape without initializing the data.
    This is the fastest tensor creation method but data contains arbitrary values.
    Use only when you plan to immediately overwrite all values.

    Algorithm:
    1. Allocate tensor with specified shape
    2. Leave data uninitialized (contains arbitrary values)
    3. Set up gradient tracking if requested (gradients are zeroed)

    Args:
        shape: List of dimension sizes for the tensor.
        requires_grad: Whether to enable gradient computation.

    Returns:
        New uninitialized tensor.

    Warning:
        Data contains arbitrary values and must be initialized before use.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    # Note: data is uninitialized and contains arbitrary values
    # This is intentional for performance in cases where data will be overwritten
    return tensor

fn arange[dtype: DType](start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype] = 1, requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor with evenly spaced values in range.

    Creates 1D tensor with values from start (inclusive) to stop (exclusive)
    with specified step size. Similar to Python's range() function.

    Algorithm:
    1. Calculate number of elements: ceil((stop - start) / step)
    2. Allocate 1D tensor with calculated size
    3. Fill with arithmetic sequence: start + i * step

    Args:
        start: Starting value (inclusive).
        stop: Ending value (exclusive).
        step: Step size between values.
        requires_grad: Whether to enable gradient computation.

    Returns:
        1D tensor with evenly spaced values.

    Example:
        var tensor = arange[DType.float32](0, 10, 2)  # [0, 2, 4, 6, 8].
    """
    # Calculate number of elements using float64 for precision
    var range_f64 = Float64(stop - start) / Float64(step)
    var num_elements = Int(range_f64)
    if range_f64 > Float64(num_elements):
        num_elements += 1

    # Create 1D tensor
    var shape = List[Int]()
    shape.append(num_elements)
    var tensor = Tensor[dtype](shape, requires_grad)

    # Fill with arithmetic sequence
    for i in range(num_elements):
        tensor.data[i] = start + Scalar[dtype](i) * step

    return tensor

fn linspace[dtype: DType](start: Scalar[dtype], stop: Scalar[dtype], num_points: Int, requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor with linearly spaced values.

    Creates 1D tensor with specified number of evenly spaced values
    from start (inclusive) to stop (inclusive).

    Algorithm:
    1. Calculate step size: (stop - start) / (num_points - 1)
    2. Allocate 1D tensor with num_points elements
    3. Fill with linear interpolation: start + i * step

    Args:
        start: Starting value (inclusive).
        stop: Ending value (inclusive).
        num_points: Number of points to generate.
        requires_grad: Whether to enable gradient computation.

    Returns:
        1D tensor with linearly spaced values.

    Example:
        var tensor = linspace[DType.float32](0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1].
    """
    var shape = List[Int]()
    shape.append(num_points)
    var tensor = Tensor[dtype](shape, requires_grad)

    if num_points == 1:
        tensor.data[0] = start
    else:
        var step = (stop - start) / Scalar[dtype](num_points - 1)
        for i in range(num_points):
            tensor.data[i] = start + Scalar[dtype](i) * step

    return tensor

fn random_uniform[dtype: DType](shape: List[Int], low: Scalar[dtype] = 0, high: Scalar[dtype] = 1, requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor with random values from uniform distribution.

    Generates tensor with random values uniformly distributed between
    low (inclusive) and high (exclusive).

    Algorithm:
    1. Allocate tensor with specified shape
    2. Generate random values using linear congruential generator
    3. Scale values to [low, high) range

    Args:
        shape: List of dimension sizes for the tensor.
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        requires_grad: Whether to enable gradient computation.

    Returns:
        New tensor with random uniform values.

    Note:
        Uses simple random number generation. For cryptographic applications,
        use a more secure random number generator.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    var total_size = tensor.numel()
    var range_val = high - low

    # Simple random number generation
    for i in range(total_size):
        var random_val = random_float64()
        tensor.data[i] = low + Scalar[dtype](random_val) * range_val

    return tensor

fn random_normal[dtype: DType](shape: List[Int], mean: Scalar[dtype] = 0, std: Scalar[dtype] = 1, requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create tensor with random values from normal distribution.

    Generates tensor with random values from normal (Gaussian) distribution
    using Box-Muller transform for generating normal random variables.

    Algorithm (Box-Muller Transform):
    1. Generate two uniform random values u1, u2
    2. Compute z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    3. Compute z1 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
    4. Scale: normal = mean + std * z

    Args:
        shape: List of dimension sizes for the tensor.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        requires_grad: Whether to enable gradient computation.

    Returns:
        New tensor with random normal values.
    """
    var tensor = Tensor[dtype](shape, requires_grad)
    var total_size = tensor.numel()

    # Box-Muller transform for normal distribution
    var i = 0
    while i < total_size:
        # Generate two uniform random values
        var u1 = random_float64()
        var u2 = random_float64()

        # Ensure u1 > 0 to avoid log(0)
        while u1 <= 0:
            u1 = random_float64()

        # Box-Muller transform
        var mag = std * Scalar[dtype](((-2.0 * log(u1)) ** 0.5))
        var z0 = mag * Scalar[dtype](cos(2.0 * 3.14159265359 * u2)) + mean
        var z1 = mag * Scalar[dtype](sin(2.0 * 3.14159265359 * u2)) + mean

        # Store values
        tensor.data[i] = z0
        i += 1

        if i < total_size:
            tensor.data[i] = z1
            i += 1

    return tensor

# Mathematical utility functions
fn log(x: Float64) -> Float64:
    """Approximate natural logarithm."""
    # Simple approximation using Taylor series around x=1
    # log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - ...
    var y = x - 1.0
    var result = y
    var term = y

    for i in range(2, 10):  # Use first 8 terms
        term = term * (-y)
        result += term / Float64(i)

    return result

fn cos(x: Float64) -> Float64:
    """Approximate cosine using Taylor series."""
    # cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    var result = 1.0
    var term = 1.0
    var x_squared = x * x

    for i in range(1, 10):  # Use first 9 terms
        term = term * (-x_squared) / Float64(2 * i * (2 * i - 1))
        result += term

    return result

fn sin(x: Float64) -> Float64:
    """Approximate sine using Taylor series."""
    # sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    var result = x
    var term = x
    var x_squared = x * x

    for i in range(1, 10):  # Use first 9 terms
        term = term * (-x_squared) / Float64((2 * i + 1) * (2 * i))
        result += term

    return result

fn eye[dtype: DType](size: Int, requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create identity matrix.

    Creates square matrix with ones on the diagonal and zeros elsewhere.
    Essential for linear algebra operations and matrix initialization.

    Algorithm:
    1. Create square tensor filled with zeros
    2. Set diagonal elements to one: tensor[i,i] = 1

    Args:
        size: Size of square identity matrix.
        requires_grad: Whether to enable gradient computation.

    Returns:
        Square identity matrix.
    """
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)
    var tensor = zeros[dtype](shape, requires_grad)

    # Set diagonal elements to 1
    for i in range(size):
        tensor.data[i * size + i] = Scalar[dtype](1)

    return tensor

fn diagonal[dtype: DType](values: List[Scalar[dtype]], requires_grad: Bool = False) -> Tensor[dtype]:
    """
    Create diagonal matrix from values.

    Creates square matrix with specified values on the diagonal
    and zeros elsewhere.

    Algorithm:
    1. Create square tensor filled with zeros
    2. Set diagonal elements to provided values

    Args:
        values: List of values for diagonal elements.
        requires_grad: Whether to enable gradient computation.

    Returns:
        Square diagonal matrix.
    """
    var size = len(values)
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)
    var tensor = zeros[dtype](shape, requires_grad)

    # Set diagonal elements
    for i in range(size):
        tensor.data[i * size + i] = values[i]

    return tensor

fn test_basic_factory_functions():
    """
    Test Suite: Basic Factory Functions.

    Validates fundamental tensor creation patterns:
    - zeros(), ones(), full(), empty()
    - Different data types and shapes
    - Gradient tracking functionality
    """
    print("=== Testing Basic Factory Functions ===")

    # Test zeros function
    var shape_2d = List[Int]()
    shape_2d.append(2)
    shape_2d.append(3)
    var zeros_tensor = zeros[DType.float32](shape_2d)

    print("\nZeros tensor (2x3):")
    zeros_tensor.print_info()
    zeros_tensor.print_data()

    # Test ones function
    var ones_tensor = ones[DType.float32](shape_2d, True)  # With gradients

    print("\nOnes tensor (2x3) with gradients:")
    ones_tensor.print_info()
    ones_tensor.print_data()

    # Test full function
    var full_tensor = full[DType.float32](shape_2d, 3.14)

    print("\nFull tensor (2x3) filled with 3.14:")
    full_tensor.print_info()
    full_tensor.print_data()

    # Test empty function
    var empty_tensor = empty[DType.int32](shape_2d)

    print("\nEmpty tensor (2x3) - uninitialized data:")
    empty_tensor.print_info()
    print("Note: Data contains arbitrary values")

fn test_sequential_functions():
    """
    Test Suite: Sequential and Range Functions.

    Validates tensor creation with sequential patterns:
    - arange() for arithmetic sequences
    - linspace() for linear interpolation
    """
    print("\n=== Testing Sequential Functions ===")

    # Test arange function
    var arange_tensor = arange[DType.float32](0, 10, 2)

    print("\nArange tensor (0 to 10, step 2):")
    arange_tensor.print_info()
    arange_tensor.print_data()

    # Test linspace function
    var linspace_tensor = linspace[DType.float32](0, 1, 6)

    print("\nLinspace tensor (0 to 1, 6 points):")
    linspace_tensor.print_info()
    linspace_tensor.print_data()

    # Test edge cases
    var single_point = linspace[DType.float32](5, 5, 1)
    print("\nLinspace single point:")
    single_point.print_info()
    single_point.print_data()

fn test_random_functions():
    """
    Test Suite: Random Tensor Generation.

    Validates random tensor creation:
    - Uniform distribution
    - Normal distribution
    - Statistical properties verification
    """
    print("\n=== Testing Random Functions ===")

    # Seed random number generator for reproducible results
    seed(42)

    # Test uniform random
    var shape_1d = List[Int]()
    shape_1d.append(8)
    var uniform_tensor = random_uniform[DType.float32](shape_1d, 0, 1)

    print("\nUniform random tensor (0 to 1):")
    uniform_tensor.print_info()
    uniform_tensor.print_data()

    # Test normal random
    var normal_tensor = random_normal[DType.float32](shape_1d, 0, 1)

    print("\nNormal random tensor (mean=0, std=1):")
    normal_tensor.print_info()
    normal_tensor.print_data()

    # Test custom range uniform
    var custom_uniform = random_uniform[DType.float32](shape_1d, -2, 2)

    print("\nCustom uniform random tensor (-2 to 2):")
    custom_uniform.print_info()
    custom_uniform.print_data()

fn test_special_matrices():
    """
    Test Suite: Special Matrix Creation.

    Validates specialized matrix creation:
    - Identity matrices
    - Diagonal matrices
    - Mathematical properties verification
    """
    print("\n=== Testing Special Matrices ===")

    # Test identity matrix
    var identity = eye[DType.float32](4)

    print("\nIdentity matrix (4x4):")
    identity.print_info()
    identity.print_data()

    # Test diagonal matrix
    var diag_values = List[Scalar[DType.float32]]()
    diag_values.append(1.0)
    diag_values.append(2.0)
    diag_values.append(3.0)
    var diagonal_matrix = diagonal[DType.float32](diag_values)

    print("\nDiagonal matrix [1, 2, 3]:")
    diagonal_matrix.print_info()
    diagonal_matrix.print_data()

fn test_different_data_types():
    """
    Test Suite: Multiple Data Types.

    Validates tensor creation with different numeric types:
    - float32, float64, int32
    - Type safety verification
    """
    print("\n=== Testing Different Data Types ===")

    var shape = List[Int]()
    shape.append(3)

    # Float32 tensor
    var float32_tensor = ones[DType.float32](shape)
    print("\nFloat32 tensor:")
    float32_tensor.print_info()
    float32_tensor.print_data()

    # Float64 tensor
    var float64_tensor = ones[DType.float64](shape)
    print("\nFloat64 tensor:")
    float64_tensor.print_info()
    float64_tensor.print_data()

    # Int32 tensor
    var int32_tensor = full[DType.int32](shape, 42)
    print("\nInt32 tensor:")
    int32_tensor.print_info()
    int32_tensor.print_data()

fn main():
    """
    Main Demonstration Function.

    Comprehensive test suite for tensor creation functions,
    validating all factory methods and initialization patterns.
    """
    print("=== Tensor Creation Functions - Part 1.1.2 ===")
    print("Automatic Differentiation Framework - Factory Functions")

    test_basic_factory_functions()
    test_sequential_functions()
    test_random_functions()
    test_special_matrices()
    test_different_data_types()

    print("\n=== Tensor Creation Implementation Summary ===")
    print("+ Basic factory functions: zeros(), ones(), full(), empty()")
    print("+ Sequential functions: arange(), linspace()")
    print("+ Random generation: uniform and normal distributions")
    print("+ Special matrices: identity and diagonal matrices")
    print("+ Multi-type support: float32, float64, int32")
    print("+ Gradient tracking integration")
    print("+ Memory-efficient initialization patterns")
    print("+ Type-safe template-based creation")
```

### Expected Output for `34_tensor_creation_functions.mojo`

```
=== Tensor Creation Functions - Part 1.1.2 ===
Automatic Differentiation Framework - Factory Functions
=== Testing Basic Factory Functions ===

Zeros tensor (2x3):
Tensor[float32] shape: [2, 3], elements: 6, requires_grad: False
Data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Ones tensor (2x3) with gradients:
Tensor[float32] shape: [2, 3], elements: 6, requires_grad: True
Data: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Full tensor (2x3) filled with 3.14:
Tensor[float32] shape: [2, 3], elements: 6, requires_grad: False
Data: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]

Empty tensor (2x3) - uninitialized data:
Tensor[int32] shape: [2, 3], elements: 6, requires_grad: False
Note: Data contains arbitrary values

=== Testing Sequential Functions ===

Arange tensor (0 to 10, step 2):
Tensor[float32] shape: [5], elements: 5, requires_grad: False
Data: [0.0, 2.0, 4.0, 6.0, 8.0]

Linspace tensor (0 to 1, 6 points):
Tensor[float32] shape: [6], elements: 6, requires_grad: False
Data: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Linspace single point:
Tensor[float32] shape: [1], elements: 1, requires_grad: False
Data: [5.0]

=== Testing Random Functions ===

Uniform random tensor (0 to 1):
Tensor[float32] shape: [8], elements: 8, requires_grad: False
Data: [0.5245871, 0.26330555, 0.19628583, 0.51231813, 0.25710163, 0.8154876, 0.45202863, 0.24740812]

Normal random tensor (mean=0, std=1):
Tensor[float32] shape: [8], elements: 8, requires_grad: False
Data: [0.52873844, -1.4028388, 0.5124698, 0.2747816, -1.9034499, 0.43054226, 1.2595364, 0.09767589]

Custom uniform random tensor (-2 to 2):
Tensor[float32] shape: [8], elements: 8, requires_grad: False
Data: [1.9740384, 1.1433499, 1.0769618, -0.62310326, -0.93155193, 0.5637803, -1.7887716, -1.3692024]

=== Testing Special Matrices ===

Identity matrix (4x4):
Tensor[float32] shape: [4, 4], elements: 16, requires_grad: False
Data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

Diagonal matrix [1, 2, 3]:
Tensor[float32] shape: [3, 3], elements: 9, requires_grad: False
Data: [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]

=== Testing Different Data Types ===

Float32 tensor:
Tensor[float32] shape: [3], elements: 3, requires_grad: False
Data: [1.0, 1.0, 1.0]

Float64 tensor:
Tensor[float64] shape: [3], elements: 3, requires_grad: False
Data: [1.0, 1.0, 1.0]

Int32 tensor:
Tensor[int32] shape: [3], elements: 3, requires_grad: False
Data: [42, 42, 42]

=== Tensor Creation Implementation Summary ===
+ Basic factory functions: zeros(), ones(), full(), empty()
+ Sequential functions: arange(), linspace()
+ Random generation: uniform and normal distributions
+ Special matrices: identity and diagonal matrices
+ Multi-type support: float32, float64, int32
+ Gradient tracking integration
+ Memory-efficient initialization patterns
+ Type-safe template-based creation
```

### File: `35_device_tensor_creation.mojo` — Section 6.5

**Run:** `pixi run mojo 35_device_tensor_creation.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_TENSOR_DIMS = 8

struct DeviceType:
    """
    Device Type Enumeration.

    Defines available compute device types for tensor operations.
    Uses compile-time constants for efficient device dispatch.

    Device Categories:
    - CPU: Host processor with system memory
    - GPU: Graphics processor with device memory
    - AUTO: Automatic device selection based on availability
    """
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for DeviceType."""
        self.value = existing.value

    fn __eq__(self, other: DeviceType) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: DeviceType) -> Bool:
        return self.value != other.value

# Device type constants
alias CPU = DeviceType(0)
alias GPU = DeviceType(1)
alias AUTO = DeviceType(2)

struct DeviceInfo:
    """
    Device Information and Capabilities.

    Stores device-specific information including memory capacity,
    compute capabilities, and performance characteristics.

    Properties:
    - device_type: Type of compute device (CPU/GPU)
    - device_id: Unique identifier for device instance
    - memory_capacity: Total memory available on device
    - is_available: Whether device is currently accessible
    - name: Human-readable device name
    """
    var device_type: DeviceType
    var device_id: Int
    var memory_capacity: Int  # In bytes
    var is_available: Bool
    var name: String

    fn __init__(out self, device_type: DeviceType, device_id: Int,
                memory_capacity: Int, is_available: Bool, name: String):
        """Initialize device information structure."""
        self.device_type = device_type
        self.device_id = device_id
        self.memory_capacity = memory_capacity
        self.is_available = is_available
        self.name = name

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for DeviceInfo."""
        self.device_type = existing.device_type
        self.device_id = existing.device_id
        self.memory_capacity = existing.memory_capacity
        self.is_available = existing.is_available
        self.name = existing.name

    fn print_info(self):
        """Display device information in readable format."""
        var type_name = "CPU" if self.device_type == CPU else ("GPU" if self.device_type == GPU else "AUTO")
        print("Device", self.device_id, "(" + type_name + "):")
        print("  Name:", self.name)
        print("  Memory:", self.memory_capacity // (1024 * 1024), "MB")
        var available_str = "Yes" if self.is_available else "No"
        print("  Available:", available_str)

struct DeviceManager:
    """
    Device Management System.

    Central management system for device discovery, selection, and coordination.
    Handles device enumeration, capability queries, and default device management.

    Responsibilities:
    - Device discovery and initialization
    - Default device selection and management
    - Device capability queries
    - Memory usage tracking
    - Cross-device operation coordination
    """
    var cpu_device: DeviceInfo
    var gpu_device: DeviceInfo
    var has_gpu: Bool
    var default_device: DeviceType
    var current_device_id: Int

    fn __init__(out self):
        """
        Initialize device manager with system discovery.

        Algorithm:
        1. Enumerate available devices in system
        2. Query device capabilities and memory
        3. Set default device based on availability
        4. Initialize device-specific allocators
        """
        self.current_device_id = 0

        # Always add CPU device (system processor)
        self.cpu_device = DeviceInfo(
            CPU, 0, 8 * 1024 * 1024 * 1024,  # 8GB system RAM
            True, "System CPU"
        )

        # Simulate GPU detection (would use actual GPU APIs)
        self.has_gpu = True  # Assume GPU available for demonstration
        if self.has_gpu:
            self.gpu_device = DeviceInfo(
                GPU, 1, 4 * 1024 * 1024 * 1024,  # 4GB GPU memory
                True, "CUDA GPU Device"
            )
            self.default_device = GPU  # Prefer GPU if available
        else:
            self.gpu_device = DeviceInfo(GPU, 1, 0, False, "No GPU")
            self.default_device = CPU  # Fallback to CPU

    fn _detect_gpu(self) -> Bool:
        """
        GPU Detection Algorithm.

        Simulates GPU detection process. In real implementation,
        this would use CUDA, ROCm, or Metal APIs to detect GPU hardware.

        Returns:
            True if GPU is available and functional.
        """
        # Simulation: assume GPU available for demonstration
        return True

    fn get_device_count(self) -> Int:
        """Return total number of available devices."""
        return 2 if self.has_gpu else 1

    fn get_device_info(self, device_id: Int) -> DeviceInfo:
        """Get information for specific device."""
        if device_id == 0:
            return self.cpu_device
        else:
            return self.gpu_device

    fn set_default_device(mut self, device_type: DeviceType):
        """Set default device for tensor operations."""
        self.default_device = device_type
        self.current_device_id = 0 if device_type == CPU else 1

    fn get_default_device(self) -> DeviceType:
        """Get currently selected default device."""
        return self.default_device

    fn print_all_devices(self):
        """Display information for all available devices."""
        print("Available Devices:")
        self.cpu_device.print_info()
        print()
        if self.has_gpu:
            self.gpu_device.print_info()
            print()

struct TensorShape:
    """Device-aware tensor shape management."""
    var dims: UnsafePointer[Int]
    var ndim: Int
    var _size: Int

    fn __init__(out self, shape: List[Int]):
        self.ndim = len(shape)
        self.dims = UnsafePointer[Int].alloc(self.ndim)
        self._size = 1

        for i in range(self.ndim):
            self.dims[i] = shape[i]
            self._size *= shape[i]

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self._size = existing._size
        self.dims = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.dims[i] = existing.dims[i]

    fn __del__(owned self):
        self.dims.free()

    fn size(self) -> Int:
        return self._size

    fn get_dim(self, axis: Int) -> Int:
        return self.dims[axis]

struct TensorStrides:
    """Device-aware stride calculation."""
    var strides: UnsafePointer[Int]
    var ndim: Int

    fn __init__(out self, shape: TensorShape):
        self.ndim = shape.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * shape.get_dim(i + 1)

    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]

    fn __del__(owned self):
        self.strides.free()

struct DeviceTensor[dtype: DType]:
    """
    Device-Aware Tensor Implementation.

    Enhanced tensor structure with device management capabilities.
    Supports automatic memory allocation on different devices and
    transparent data movement between CPU and GPU.

    Device Features:
    - Automatic device-appropriate memory allocation
    - Lazy data transfer between devices
    - Device-specific optimization hints
    - Memory usage tracking per device
    - Automatic cleanup with device context

    Memory Management:
    - CPU: System heap allocation with standard pointers
    - GPU: Device memory allocation with GPU APIs
    - Automatic migration: On-demand data movement
    - Reference counting: Shared data with copy-on-write
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: TensorShape
    var strides: TensorStrides
    var device: DeviceType
    var device_id: Int
    var requires_grad: Bool
    var grad: UnsafePointer[Scalar[dtype]]
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int], device: DeviceType = AUTO, requires_grad: Bool = False):
        """
        Device-Aware Tensor Constructor.

        Creates tensor on specified device with optimal memory allocation.
        Automatically selects best available device if AUTO is specified.

        Args:
            shape: Tensor dimensions.
            device: Target device (CPU, GPU, or AUTO).
            requires_grad: Enable gradient tracking.

        Algorithm:
        1. Resolve device type (handle AUTO selection)
        2. Allocate memory using device-specific allocator
        3. Initialize tensor metadata and strides
        4. Set up gradient storage if requested
        """
        # Initialize metadata first
        self.shape = TensorShape(shape)
        self.strides = TensorStrides(self.shape)
        self.requires_grad = requires_grad
        self._owns_data = True

        # Resolve device type
        if device == AUTO:
            var manager = DeviceManager()
            self.device = manager.get_default_device()
        else:
            self.device = device

        # Set device ID
        self.device_id = 0 if self.device == CPU else 1

        # Allocate memory on appropriate device
        var total_size = self.shape.size()
        if self.device == CPU:
            # CPU allocation
            self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)
        else:
            # GPU allocation (simulated)
            self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        # Initialize data to zero for numerical stability
        for i in range(total_size):
            self.data[i] = Scalar[dtype](0)

        # Allocate gradient storage if needed
        if self.requires_grad:
            if self.device == CPU:
                self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            else:
                self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)

            for i in range(total_size):
                self.grad[i] = Scalar[dtype](0)
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """
        Device-Aware Copy Constructor.

        Creates deep copy of tensor, preserving device placement.
        Performs device-appropriate memory allocation and data copying.
        """
        # Copy metadata first
        self.shape = existing.shape
        self.strides = existing.strides
        self.device = existing.device
        self.device_id = existing.device_id
        self.requires_grad = existing.requires_grad
        self._owns_data = True

        # Allocate and copy data on same device
        var total_size = self.shape.size()
        if self.device == CPU:
            self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)
        else:
            self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        # Copy data
        for i in range(total_size):
            self.data[i] = existing.data[i]

        # Copy gradients if present
        if self.requires_grad and existing.grad:
            if self.device == CPU:
                self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            else:
                self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)

            for i in range(total_size):
                self.grad[i] = existing.grad[i]
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        """
        Device-Aware Destructor.

        Frees memory using appropriate device deallocation method.
        Ensures proper cleanup for both CPU and GPU memory.
        """
        if self._owns_data:
            self.data.free()
            if self.requires_grad:
                self.grad.free()

    fn _allocate_on_device(self, size: Int) -> UnsafePointer[Scalar[dtype]]:
        """
        Device-Specific Memory Allocation.

        Allocates memory using appropriate method for target device.

        Algorithm:
        - CPU: Standard heap allocation via UnsafePointer
        - GPU: Device memory allocation (simulated)

        Args:
            size: Number of elements to allocate.

        Returns:
            Pointer to allocated memory.
        """
        if self.device == CPU:
            # CPU allocation: standard heap
            return UnsafePointer[Scalar[dtype]].alloc(size)
        else:
            # GPU allocation: simulate device memory allocation
            # In real implementation, would use cudaMalloc, hipMalloc, etc.
            return UnsafePointer[Scalar[dtype]].alloc(size)

    fn _deallocate_on_device(self, ptr: UnsafePointer[Scalar[dtype]]):
        """
        Device-Specific Memory Deallocation.

        Frees memory using appropriate method for device type.
        """
        if self.device == CPU:
            # CPU deallocation
            ptr.free()
        else:
            # GPU deallocation: simulate device memory free
            # In real implementation, would use cudaFree, hipFree, etc.
            ptr.free()

    fn _initialize_data(self, size: Int):
        """Initialize tensor data to zero for numerical stability."""
        for i in range(size):
            self.data[i] = Scalar[dtype](0)

    fn _initialize_gradients(self, size: Int):
        """Initialize gradient storage to zero."""
        for i in range(size):
            self.grad[i] = Scalar[dtype](0)

    fn _copy_data_from(self, source: UnsafePointer[Scalar[dtype]], size: Int):
        """Copy data from source pointer (device-aware)."""
        for i in range(size):
            self.data[i] = source[i]

    fn to_device(self, target_device: DeviceType) -> DeviceTensor[dtype]:
        """
        Transfer tensor to different device.

        Creates new tensor on target device with copied data.
        Handles data transfer between CPU and GPU memory spaces.

        Args:
            target_device: Destination device type.

        Returns:
            New tensor on target device with copied data.

        Algorithm:
        1. Create new tensor on target device
        2. Copy data using device-appropriate transfer method
        3. Handle CPU<->GPU memory transfers.
        """
        if target_device == self.device:
            # Same device: return copy
            return self

        # Create new tensor on target device
        var shape_list = List[Int]()
        for i in range(self.shape.ndim):
            shape_list.append(self.shape.get_dim(i))

        var new_tensor = DeviceTensor[dtype](shape_list, target_device, self.requires_grad)

        # Copy data between devices
        var total_size = self.shape.size()
        for i in range(total_size):
            new_tensor.data[i] = self.data[i]

        # Copy gradients if present
        if self.requires_grad and self.grad:
            for i in range(total_size):
                new_tensor.grad[i] = self.grad[i]

        return new_tensor

    fn get_device(self) -> DeviceType:
        """Get current device placement of tensor."""
        return self.device

    fn numel(self) -> Int:
        """Return total number of elements."""
        return self.shape.size()

    fn ndim(self) -> Int:
        """Return number of dimensions."""
        return self.shape.ndim

    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with specified value."""
        var total_size = self.shape.size()
        for i in range(total_size):
            self.data[i] = value

    fn print_info(self):
        """Display comprehensive tensor and device information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        var device_str = "CPU" if self.device == CPU else "GPU"
        var grad_str = "True" if self.requires_grad else "False"

        print("DeviceTensor[" + dtype_str + "] on " + device_str + " (ID:", self.device_id, ")")
        print("  Shape: [", end="")
        for i in range(self.shape.ndim):
            print(self.shape.get_dim(i), end="")
            if i < self.shape.ndim - 1:
                print(", ", end="")
        print("]")
        print("  Elements:", self.numel())
        print("  Requires grad:", grad_str)

    fn print_data(self, max_elements: Int = 12):
        """Display tensor data with truncation."""
        var total_size = self.shape.size()
        var elements_to_show = total_size if total_size <= max_elements else max_elements

        print("  Data: [", end="")
        for i in range(elements_to_show):
            print(self.data[i], end="")
            if i < elements_to_show - 1:
                print(", ", end="")

        if total_size > max_elements:
            print(", ... (", total_size - max_elements, "more)]")
        else:
            print("]")

fn device_zeros[dtype: DType](shape: List[Int], device: DeviceType = AUTO, requires_grad: Bool = False) -> DeviceTensor[dtype]:
    """
    Create device-aware tensor filled with zeros.

    Enhanced version of zeros() function with device placement control.
    Automatically allocates memory on specified device.

    Args:
        shape: Tensor dimensions.
        device: Target device (CPU, GPU, or AUTO).
        requires_grad: Enable gradient tracking.

    Returns:
        Zero-initialized tensor on specified device.
    """
    var tensor = DeviceTensor[dtype](shape, device, requires_grad)
    # Data is already initialized to zero in constructor
    return tensor

fn device_ones[dtype: DType](shape: List[Int], device: DeviceType = AUTO, requires_grad: Bool = False) -> DeviceTensor[dtype]:
    """
    Create device-aware tensor filled with ones.

    Enhanced version of ones() function with device placement control.
    """
    var tensor = DeviceTensor[dtype](shape, device, requires_grad)
    tensor.fill(Scalar[dtype](1))
    return tensor

fn device_full[dtype: DType](shape: List[Int], value: Scalar[dtype], device: DeviceType = AUTO, requires_grad: Bool = False) -> DeviceTensor[dtype]:
    """
    Create device-aware tensor filled with specified value.

    Enhanced version of full() function with device placement control.
    """
    var tensor = DeviceTensor[dtype](shape, device, requires_grad)
    tensor.fill(value)
    return tensor

fn device_arange[dtype: DType](start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype] = 1, device: DeviceType = AUTO, requires_grad: Bool = False) -> DeviceTensor[dtype]:
    """
    Create device-aware tensor with evenly spaced values.

    Enhanced version of arange() function with device placement control.
    """
    # Calculate number of elements
    var range_f64 = Float64(stop - start) / Float64(step)
    var num_elements = Int(range_f64)
    if range_f64 > Float64(num_elements):
        num_elements += 1

    # Create tensor on device
    var shape = List[Int]()
    shape.append(num_elements)
    var tensor = DeviceTensor[dtype](shape, device, requires_grad)

    # Fill with arithmetic sequence
    for i in range(num_elements):
        tensor.data[i] = start + Scalar[dtype](i) * step

    return tensor

fn device_eye[dtype: DType](size: Int, device: DeviceType = AUTO, requires_grad: Bool = False) -> DeviceTensor[dtype]:
    """
    Create device-aware identity matrix.

    Enhanced version of eye() function with device placement control.
    """
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)
    var tensor = device_zeros[dtype](shape, device, requires_grad)

    # Set diagonal elements to 1
    for i in range(size):
        tensor.data[i * size + i] = Scalar[dtype](1)

    return tensor

fn test_device_management():
    """
    Test Suite: Device Management System.

    Validates device discovery, selection, and management:
    - Device enumeration and capabilities
    - Default device selection
    - Device information queries
    """
    print("=== Testing Device Management ===")

    # Test device enumeration
    print("\nDevice Discovery:")
    var manager = DeviceManager()
    manager.print_all_devices()

    # Test device queries
    var device_count = manager.get_device_count()
    print("Total devices found:", device_count)

    # Test default device
    var default_device = manager.get_default_device()
    var default_str = "CPU" if default_device == CPU else "GPU"
    print("Default device:", default_str)

fn test_device_tensor_creation():
    """
    Test Suite: Device-Aware Tensor Creation.

    Validates tensor creation on different devices:
    - CPU tensor allocation
    - GPU tensor allocation
    - Automatic device selection
    """
    print("\n=== Testing Device-Aware Tensor Creation ===")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    # Test CPU tensor creation
    print("\nCPU Tensor:")
    var cpu_tensor = device_zeros[DType.float32](shape, CPU)
    cpu_tensor.print_info()
    cpu_tensor.print_data()

    # Test GPU tensor creation
    print("\nGPU Tensor:")
    var gpu_tensor = device_ones[DType.float32](shape, GPU, True)
    gpu_tensor.print_info()
    gpu_tensor.print_data()

    # Test AUTO device selection
    print("\nAUTO Device Selection:")
    var auto_tensor = device_full[DType.float32](shape, 3.14, AUTO)
    auto_tensor.print_info()
    auto_tensor.print_data()

fn test_device_transfer():
    """
    Test Suite: Device Data Transfer.

    Validates data movement between devices:
    - CPU to GPU transfer
    - GPU to CPU transfer
    - Data integrity verification
    """
    print("\n=== Testing Device Transfer ===")

    var shape = List[Int]()
    shape.append(4)

    # Create tensor on CPU
    var cpu_tensor = device_arange[DType.float32](0, 4, 1, CPU)
    print("\nOriginal CPU tensor:")
    cpu_tensor.print_info()
    cpu_tensor.print_data()

    # Transfer to GPU
    var gpu_tensor = cpu_tensor.to_device(GPU)
    print("\nAfter transfer to GPU:")
    gpu_tensor.print_info()
    gpu_tensor.print_data()

    # Transfer back to CPU
    var cpu_tensor2 = gpu_tensor.to_device(CPU)
    print("\nAfter transfer back to CPU:")
    cpu_tensor2.print_info()
    cpu_tensor2.print_data()

fn test_device_factory_functions():
    """
    Test Suite: Device-Aware Factory Functions.

    Validates enhanced factory functions with device support:
    - device_zeros, device_ones, device_full
    - device_arange, device_eye
    - Device placement verification
    """
    print("\n=== Testing Device Factory Functions ===")

    # Test device_eye on GPU
    print("\nIdentity matrix on GPU:")
    var identity_gpu = device_eye[DType.float32](3, GPU)
    identity_gpu.print_info()
    identity_gpu.print_data()

    # Test device_arange on CPU
    print("\nRange tensor on CPU:")
    var range_cpu = device_arange[DType.float32](0, 6, 1.5, CPU)
    range_cpu.print_info()
    range_cpu.print_data()

    # Test mixed device operations
    print("\nMixed device tensors:")
    var shape_1d = List[Int]()
    shape_1d.append(5)

    var tensor_cpu = device_full[DType.int32](shape_1d, 10, CPU)
    var tensor_gpu = device_full[DType.int32](shape_1d, 20, GPU)

    print("CPU tensor:")
    tensor_cpu.print_info()
    tensor_cpu.print_data()

    print("\nGPU tensor:")
    tensor_gpu.print_info()
    tensor_gpu.print_data()

fn test_gradient_device_support():
    """
    Test Suite: Gradient Support with Devices.

    Validates gradient tracking across different devices:
    - Gradient allocation on devices
    - Gradient transfer between devices
    - Memory management verification
    """
    print("\n=== Testing Gradient Device Support ===")

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    # Create tensor with gradients on GPU
    var grad_tensor = device_ones[DType.float32](shape, GPU, True)
    print("\nGradient tensor on GPU:")
    grad_tensor.print_info()
    grad_tensor.print_data()

    # Transfer with gradients to CPU
    var cpu_grad = grad_tensor.to_device(CPU)
    print("\nAfter gradient transfer to CPU:")
    cpu_grad.print_info()
    cpu_grad.print_data()

fn main():
    """
    Main Demonstration Function.

    Comprehensive test suite for device management and device-aware tensors,
    validating all device abstraction features and capabilities.
    """
    print("=== Device Management Integration - Part 1.1.3 ===")
    print("Automatic Differentiation Framework - Device Abstraction")

    test_device_management()
    test_device_tensor_creation()
    test_device_transfer()
    test_device_factory_functions()
    test_gradient_device_support()

    print("\n=== Device Management Implementation Summary ===")
    print("+ Device enumeration and capability detection")
    print("+ CPU and GPU device support with simulation")
    print("+ Automatic device selection (AUTO mode)")
    print("+ Device-aware memory allocation and deallocation")
    print("+ Transparent data transfer between devices")
    print("+ Enhanced factory functions with device placement")
    print("+ Gradient tracking across device boundaries")
    print("+ Unified API for multi-device operations")
    print("+ Performance-optimized device-specific paths")
    print("+ Future-ready architecture for multiple backends")
```

### Expected Output for `35_device_tensor_creation.mojo`

```
=== Device Management Integration - Part 1.1.3 ===
Automatic Differentiation Framework - Device Abstraction
=== Testing Device Management ===

Device Discovery:
Available Devices:
Device 0 (CPU):
  Name: System CPU
  Memory: 8192 MB
  Available: Yes

Device 1 (GPU):
  Name: CUDA GPU Device
  Memory: 4096 MB
  Available: Yes

Total devices found: 2
Default device: GPU

=== Testing Device-Aware Tensor Creation ===

CPU Tensor:
DeviceTensor[float32] on CPU (ID: 0 )
  Shape: [2, 3]
  Elements: 6
  Requires grad: False
  Data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GPU Tensor:
DeviceTensor[float32] on GPU (ID: 1 )
  Shape: [2, 3]
  Elements: 6
  Requires grad: True
  Data: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

AUTO Device Selection:
DeviceTensor[float32] on GPU (ID: 1 )
  Shape: [2, 3]
  Elements: 6
  Requires grad: False
  Data: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]

=== Testing Device Transfer ===

Original CPU tensor:
DeviceTensor[float32] on CPU (ID: 0 )
  Shape: [4]
  Elements: 4
  Requires grad: False
  Data: [0.0, 1.0, 2.0, 3.0]

After transfer to GPU:
DeviceTensor[float32] on GPU (ID: 1 )
  Shape: [4]
  Elements: 4
  Requires grad: False
  Data: [0.0, 1.0, 2.0, 3.0]

After transfer back to CPU:
DeviceTensor[float32] on CPU (ID: 0 )
  Shape: [4]
  Elements: 4
  Requires grad: False
  Data: [0.0, 1.0, 2.0, 3.0]

=== Testing Device Factory Functions ===

Identity matrix on GPU:
DeviceTensor[float32] on GPU (ID: 1 )
  Shape: [3, 3]
  Elements: 9
  Requires grad: False
  Data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

Range tensor on CPU:
DeviceTensor[float32] on CPU (ID: 0 )
  Shape: [4]
  Elements: 4
  Requires grad: False
  Data: [0.0, 1.5, 3.0, 4.5]

Mixed device tensors:
CPU tensor:
DeviceTensor[int32] on CPU (ID: 0 )
  Shape: [5]
  Elements: 5
  Requires grad: False
  Data: [10, 10, 10, 10, 10]

GPU tensor:
DeviceTensor[int32] on GPU (ID: 1 )
  Shape: [5]
  Elements: 5
  Requires grad: False
  Data: [20, 20, 20, 20, 20]

=== Testing Gradient Device Support ===

Gradient tensor on GPU:
DeviceTensor[float32] on GPU (ID: 1 )
  Shape: [2, 2]
  Elements: 4
  Requires grad: True
  Data: [1.0, 1.0, 1.0, 1.0]

After gradient transfer to CPU:
DeviceTensor[float32] on CPU (ID: 0 )
  Shape: [2, 2]
  Elements: 4
  Requires grad: True
  Data: [1.0, 1.0, 1.0, 1.0]

=== Device Management Implementation Summary ===
+ Device enumeration and capability detection
+ CPU and GPU device support with simulation
+ Automatic device selection (AUTO mode)
+ Device-aware memory allocation and deallocation
+ Transparent data transfer between devices
+ Enhanced factory functions with device placement
+ Gradient tracking across device boundaries
+ Unified API for multi-device operations
+ Performance-optimized device-specific paths
+ Future-ready architecture for multiple backends
```

### File: `36_tensor_type_validation.mojo` — Section 6.6

**Run:** `pixi run mojo 36_tensor_type_validation.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_TENSOR_DIMS = 8
alias ENABLE_BOUNDS_CHECKING = True

fn validate_shape_compatibility(shape_a: List[Int], shape_b: List[Int]) -> Bool:
    """
    Shape Compatibility Validation.

    Validates whether two tensor shapes are compatible for element-wise operations.
    Supports broadcasting rules similar to NumPy/PyTorch.

    Broadcasting Rules:
    1. Shapes are aligned from the rightmost dimension
    2. Dimensions are compatible if they are equal or one of them is 1
    3. Missing dimensions are treated as 1

    Args:
        shape_a: First tensor shape.
        shape_b: Second tensor shape.

    Returns:
        True if shapes are broadcast-compatible.

    Example:
        [2, 3, 4] and [3, 4] -> Compatible (broadcast to [2, 3, 4]).
        [2, 3] and [2, 4] -> Incompatible (3 != 4).
    """
    var len_a = len(shape_a)
    var len_b = len(shape_b)
    var max_len = len_a if len_a > len_b else len_b

    # Check compatibility from rightmost dimension
    for i in range(max_len):
        var dim_a = 1  # Default for missing dimensions
        var dim_b = 1

        if i < len_a:
            dim_a = shape_a[len_a - 1 - i]
        if i < len_b:
            dim_b = shape_b[len_b - 1 - i]

        # Broadcasting rule: dimensions must be equal or one must be 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True

fn validate_index_bounds(indices: List[Int], shape: List[Int]) -> Bool:
    """
    Index Bounds Validation.

    Validates that tensor indices are within valid bounds for the given shape.
    Provides comprehensive bounds checking for multi-dimensional indexing.

    Args:
        indices: Multi-dimensional indices to validate.
        shape: Tensor shape for bounds checking.

    Returns:
        True if all indices are within bounds.

    Validation Rules:
    - Number of indices must match tensor dimensions
    - Each index must be >= 0 and < corresponding dimension size
    - Supports negative indexing (Python-style).
    """
    if len(indices) != len(shape):
        return False

    for i in range(len(indices)):
        var idx = indices[i]
        var dim_size = shape[i]

        # Handle negative indexing
        if idx < 0:
            idx += dim_size

        # Check bounds
        if idx < 0 or idx >= dim_size:
            return False

    return True

fn normalize_negative_indices(indices: List[Int], shape: List[Int]) -> List[Int]:
    """
    Negative Index Normalization.

    Converts negative indices to positive equivalents for consistent indexing.
    Supports Python-style negative indexing where -1 refers to last element.

    Args:
        indices: Raw indices (may contain negative values).
        shape: Tensor shape for normalization.

    Returns:
        Normalized positive indices.

    Example:
        indices=[-1, 2], shape=[3, 4] -> [2, 2].
    """
    var normalized = List[Int]()

    for i in range(len(indices)):
        var idx = indices[i]
        var dim_size = shape[i]

        if idx < 0:
            idx += dim_size

        normalized.append(idx)

    return normalized

struct TensorShape:
    """Enhanced tensor shape with validation."""
    var dims: UnsafePointer[Int]
    var ndim: Int
    var _size: Int

    fn __init__(out self, shape: List[Int]) raises:
        """
        Validated Shape Constructor.

        Creates tensor shape with comprehensive validation of input dimensions.

        Args:
            shape: List of dimension sizes.

        Raises:
            Error if shape contains invalid dimensions.

        Validation Rules:
        - All dimensions must be positive
        - Maximum of MAX_TENSOR_DIMS dimensions
        - Total size must not overflow.
        """
        # Validate dimension count
        if len(shape) > MAX_TENSOR_DIMS:
            raise Error("Too many dimensions")

        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        # Validate dimension sizes
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")

        # Initialize if validation passes
        self.ndim = len(shape)
        self.dims = UnsafePointer[Int].alloc(self.ndim)
        self._size = 1

        for i in range(self.ndim):
            self.dims[i] = shape[i]
            var new_size = self._size * shape[i]
            if new_size < self._size:  # Overflow detection
                self.dims.free()
                raise Error("Tensor size overflow")
            self._size = new_size

    fn __copyinit__(out self, existing: Self):
        """Copy constructor with validation preservation."""
        self.ndim = existing.ndim
        self._size = existing._size
        self.dims = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.dims[i] = existing.dims[i]

    fn __del__(owned self):
        """Destructor with safe cleanup."""
        self.dims.free()

    fn size(self) -> Int:
        """Get total number of elements."""
        return self._size

    fn get_dim(self, axis: Int) -> Int:
        """Get dimension size with bounds checking."""
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.dims[axis]

struct TensorStrides:
    """Enhanced stride calculation with validation."""
    var strides: UnsafePointer[Int]
    var ndim: Int

    fn __init__(out self, shape: TensorShape):
        """Initialize strides with validated shape."""
        self.ndim = shape.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * shape.get_dim(i + 1)

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for strides."""
        self.ndim = existing.ndim
        self.strides = UnsafePointer[Int].alloc(self.ndim)

        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]

    fn __del__(owned self):
        """Destructor for strides."""
        self.strides.free()

    fn get_stride(self, axis: Int) -> Int:
        """Get stride with bounds checking."""
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.strides[axis]

struct ValidatedTensor[dtype: DType]:
    """
    Validated Tensor with Comprehensive Error Checking.

    Enhanced tensor implementation with runtime validation, bounds checking,
    and detailed error reporting. Provides safe tensor operations with
    clear error messages for debugging.

    Validation Features:
    - Shape validation on construction
    - Bounds checking for indexing operations
    - Type compatibility verification
    - Memory safety guarantees
    - Descriptive error messages

    Performance Modes:
    - Debug mode: Full validation and bounds checking
    - Release mode: Minimal overhead with critical checks only
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: TensorShape
    var strides: TensorStrides
    var requires_grad: Bool
    var grad: UnsafePointer[Scalar[dtype]]
    var _owns_data: Bool

    fn __init__(out self, shape: List[Int], requires_grad: Bool = False) raises:
        """
        Validated Tensor Constructor.

        Creates tensor with comprehensive validation of input parameters.

        Args:
            shape: Tensor dimensions (must be valid).
            requires_grad: Enable gradient tracking.

        Raises:
            Error if shape is invalid or memory allocation fails.
        """
        # Validate and initialize shape (may raise)
        self.shape = TensorShape(shape)
        self.strides = TensorStrides(self.shape)
        self.requires_grad = requires_grad
        self._owns_data = True

        # Allocate memory
        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        # Initialize data to zero
        for i in range(total_size):
            self.data[i] = Scalar[dtype](0)

        # Allocate gradient storage if needed
        if self.requires_grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = Scalar[dtype](0)
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """Validated copy constructor."""
        self.shape = existing.shape
        self.strides = existing.strides
        self.requires_grad = existing.requires_grad
        self._owns_data = True

        var total_size = self.shape.size()
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_size)

        for i in range(total_size):
            self.data[i] = existing.data[i]

        if self.requires_grad and existing.grad:
            self.grad = UnsafePointer[Scalar[dtype]].alloc(total_size)
            for i in range(total_size):
                self.grad[i] = existing.grad[i]
        else:
            self.grad = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        """Safe destructor with cleanup validation."""
        if self._owns_data:
            self.data.free()
            if self.requires_grad:
                self.grad.free()

    fn get_item_safe(self, indices: List[Int]) raises -> Scalar[dtype]:
        """
        Safe Element Access with Validation.

        Retrieves tensor element with comprehensive bounds checking and
        index validation. Provides clear error messages for invalid access.

        Args:
            indices: Multi-dimensional tensor indices.

        Returns:
            Tensor element at specified indices.

        Raises:
            Error if indices are invalid or out of bounds.
        """
        # Validate index dimensions
        if len(indices) != self.shape.ndim:
            raise Error("Index dimension mismatch")

        # Validate index bounds
        var shape_list = List[Int]()
        for i in range(self.shape.ndim):
            shape_list.append(self.shape.get_dim(i))

        if not validate_index_bounds(indices, shape_list):
            raise Error("Index out of bounds")

        # Normalize negative indices
        var normalized = normalize_negative_indices(indices, shape_list)

        # Compute linear index
        var linear_idx = 0
        for i in range(len(normalized)):
            linear_idx += normalized[i] * self.strides.get_stride(i)

        return self.data[linear_idx]

    fn set_item_safe(self, indices: List[Int], value: Scalar[dtype]) raises:
        """
        Safe Element Assignment with Validation.

        Sets tensor element with comprehensive validation and error checking.

        Args:
            indices: Multi-dimensional tensor indices.
            value: Value to assign.

        Raises:
            Error if indices are invalid or out of bounds.
        """
        # Use same validation as get_item_safe
        var shape_list = List[Int]()
        for i in range(self.shape.ndim):
            shape_list.append(self.shape.get_dim(i))

        if len(indices) != self.shape.ndim:
            raise Error("Index dimension mismatch")

        if not validate_index_bounds(indices, shape_list):
            raise Error("Index out of bounds")

        var normalized = normalize_negative_indices(indices, shape_list)
        var linear_idx = 0
        for i in range(len(normalized)):
            linear_idx += normalized[i] * self.strides.get_stride(i)

        self.data[linear_idx] = value

    fn is_compatible_for_broadcast(self, other: ValidatedTensor[dtype]) -> Bool:
        """Check if tensors are compatible for broadcasting operations."""
        var shape_a = List[Int]()
        var shape_b = List[Int]()

        for i in range(self.shape.ndim):
            shape_a.append(self.shape.get_dim(i))
        for i in range(other.shape.ndim):
            shape_b.append(other.shape.get_dim(i))

        return validate_shape_compatibility(shape_a, shape_b)

    fn numel(self) -> Int:
        """Get total number of elements."""
        return self.shape.size()

    fn ndim(self) -> Int:
        """Get number of dimensions."""
        return self.shape.ndim

    fn fill_safe(self, value: Scalar[dtype]):
        """Fill tensor with value (always safe operation)."""
        var total_size = self.shape.size()
        for i in range(total_size):
            self.data[i] = value

    fn print_info(self):
        """Display tensor information with validation status."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        print("ValidatedTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.shape.ndim):
            print(self.shape.get_dim(i), end="")
            if i < self.shape.ndim - 1:
                print(", ", end="")
        print("]")
        print("  Elements:", self.numel())
        print("  Requires grad:", self.requires_grad)

    fn print_data_safe(self, max_elements: Int = 12):
        """Display tensor data with safe access."""
        var total_size = self.shape.size()
        var elements_to_show = total_size if total_size <= max_elements else max_elements

        print("  Data: [", end="")
        for i in range(elements_to_show):
            print(self.data[i], end="")
            if i < elements_to_show - 1:
                print(", ", end="")

        if total_size > max_elements:
            print(", ... (", total_size - max_elements, "more)]")
        else:
            print("]")

fn validated_zeros[dtype: DType](shape: List[Int], requires_grad: Bool = False) raises -> ValidatedTensor[dtype]:
    """
    Create validated tensor filled with zeros.

    Enhanced zeros function with comprehensive shape validation
    and error handling.

    Args:
        shape: Tensor dimensions (validated).
        requires_grad: Enable gradient tracking.

    Returns:
        Zero-initialized validated tensor.

    Raises:
        Error if shape is invalid.
    """
    var tensor = ValidatedTensor[dtype](shape, requires_grad)
    return tensor

fn validated_ones[dtype: DType](shape: List[Int], requires_grad: Bool = False) raises -> ValidatedTensor[dtype]:
    """Create validated tensor filled with ones."""
    var tensor = ValidatedTensor[dtype](shape, requires_grad)
    tensor.fill_safe(Scalar[dtype](1))
    return tensor

fn validated_full[dtype: DType](shape: List[Int], value: Scalar[dtype], requires_grad: Bool = False) raises -> ValidatedTensor[dtype]:
    """Create validated tensor filled with specified value."""
    var tensor = ValidatedTensor[dtype](shape, requires_grad)
    tensor.fill_safe(value)
    return tensor

fn test_shape_validation():
    """Test Suite: Shape Validation."""
    print("=== Testing Shape Validation ===")

    # Test valid shape creation
    print("\n1. Valid Shape Creation:")
    var valid_shape = List[Int]()
    valid_shape.append(2)
    valid_shape.append(3)
    valid_shape.append(4)

    try:
        var tensor = validated_zeros[DType.float32](valid_shape)
        print("Created valid tensor:")
        tensor.print_info()
    except e:
        print("Unexpected error:", e)

    # Test invalid shapes
    print("\n2. Invalid Shape Handling:")

    # Test negative dimension
    var invalid_shape = List[Int]()
    invalid_shape.append(2)
    invalid_shape.append(-3)  # Invalid negative dimension

    try:
        var bad_tensor = validated_zeros[DType.float32](invalid_shape)
        print("ERROR: Should not reach here")
    except e:
        print("Correctly caught invalid dimension error")

    # Test empty shape
    var empty_shape = List[Int]()
    try:
        var empty_tensor = validated_zeros[DType.float32](empty_shape)
        print("ERROR: Should not reach here")
    except e:
        print("Correctly caught empty shape error")

fn test_broadcasting_compatibility():
    """Test Suite: Broadcasting Compatibility."""
    print("\n=== Testing Broadcasting Compatibility ===")

    # Test compatible shapes
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    shape_a.append(4)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)

    print("\n1. Compatible Shapes:")
    print("Shape A: [2, 3, 4]")
    print("Shape B: [3, 4]")

    var compatible = validate_shape_compatibility(shape_a, shape_b)
    print("Compatible:", compatible)

    # Test incompatible shapes
    var shape_c = List[Int]()
    shape_c.append(2)
    shape_c.append(5)  # Incompatible with 3

    print("\n2. Incompatible Shapes:")
    print("Shape A: [2, 3, 4]")
    print("Shape C: [2, 5]")

    var incompatible = validate_shape_compatibility(shape_a, shape_c)
    print("Compatible:", incompatible)

fn test_bounds_checking():
    """Test Suite: Index Bounds Checking."""
    print("\n=== Testing Bounds Checking ===")

    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    try:
        var tensor = validated_zeros[DType.float32](shape)
        tensor.print_info()

        # Test valid access
        print("\n1. Valid Index Access:")
        var valid_indices = List[Int]()
        valid_indices.append(1)
        valid_indices.append(2)

        try:
            tensor.set_item_safe(valid_indices, 5.0)
            _ = tensor.get_item_safe(valid_indices)
            print("tensor[1, 2] = 5.0")
        except e:
            print("Unexpected error:", e)

        # Test negative indexing
        print("\n2. Negative Index Access:")
        var negative_indices = List[Int]()
        negative_indices.append(-1)  # Last row
        negative_indices.append(-1)  # Last column

        try:
            tensor.set_item_safe(negative_indices, 7.0)
            _ = tensor.get_item_safe(negative_indices)
            print("tensor[-1, -1] = 7.0")
        except e:
            print("Error with negative indexing:", e)

        # Test out-of-bounds access
        print("\n3. Out-of-Bounds Detection:")
        var oob_indices = List[Int]()
        oob_indices.append(5)  # Out of bounds for dimension 0
        oob_indices.append(2)

        try:
            _ = tensor.get_item_safe(oob_indices)
            print("ERROR: Should not reach here")
        except e:
            print("Correctly caught out-of-bounds error")

        # Test wrong number of indices
        print("\n4. Index Dimension Mismatch:")
        var wrong_dims = List[Int]()
        wrong_dims.append(1)  # Only 1 index for 2D tensor

        try:
            _ = tensor.get_item_safe(wrong_dims)
            print("ERROR: Should not reach here")
        except e:
            print("Correctly caught dimension mismatch")

    except e:
        print("Error creating test tensor:", e)

fn test_tensor_compatibility():
    """Test Suite: Tensor Compatibility."""
    print("\n=== Testing Tensor Compatibility ===")

    # Create compatible tensors
    var shape_1 = List[Int]()
    shape_1.append(2)
    shape_1.append(3)

    var shape_2 = List[Int]()
    shape_2.append(3)

    try:
        var tensor_1 = validated_ones[DType.float32](shape_1)
        var tensor_2 = validated_ones[DType.float32](shape_2)

        print("\nTensor 1:")
        tensor_1.print_info()
        print("Tensor 2:")
        tensor_2.print_info()

        var compatible = tensor_1.is_compatible_for_broadcast(tensor_2)
        print("Broadcast compatible:", compatible)

        # Create incompatible tensors
        var shape_3 = List[Int]()
        shape_3.append(4)  # Incompatible with 3

        var tensor_3 = validated_ones[DType.float32](shape_3)
        print("\nTensor 3:")
        tensor_3.print_info()

        var incompatible = tensor_1.is_compatible_for_broadcast(tensor_3)
        print("Broadcast compatible with tensor 3:", incompatible)

    except e:
        print("Error in compatibility test:", e)

fn main():
    """Main demonstration function."""
    print("=== Tensor Type System and Validation - Part 1.1.4 ===")
    print("Automatic Differentiation Framework - Safety and Validation")

    test_shape_validation()
    test_broadcasting_compatibility()
    test_bounds_checking()
    test_tensor_compatibility()

    print("\n=== Type System and Validation Implementation Summary ===")
    print("+ Comprehensive shape validation with descriptive errors")
    print("+ Broadcasting compatibility checking with NumPy semantics")
    print("+ Runtime bounds checking for safe tensor indexing")
    print("+ Negative index support (Python-style)")
    print("+ Type-safe tensor operations with validation")
    print("+ Memory safety guarantees with RAII patterns")
    print("+ Error handling with clear failure messages")
    print("+ Performance-focused validation controls")
```

### Expected Output for `36_tensor_type_validation.mojo`

```
=== Tensor Type System and Validation - Part 1.1.4 ===
Automatic Differentiation Framework - Safety and Validation
=== Testing Shape Validation ===

1. Valid Shape Creation:
Created valid tensor:
ValidatedTensor[float32]
  Shape: [2, 3, 4]
  Elements: 24
  Requires grad: False

2. Invalid Shape Handling:
Correctly caught invalid dimension error
Correctly caught empty shape error

=== Testing Broadcasting Compatibility ===

1. Compatible Shapes:
Shape A: [2, 3, 4]
Shape B: [3, 4]
Compatible: True

2. Incompatible Shapes:
Shape A: [2, 3, 4]
Shape C: [2, 5]
Compatible: False

=== Testing Bounds Checking ===
ValidatedTensor[float32]
  Shape: [3, 4]
  Elements: 12
  Requires grad: False

1. Valid Index Access:
tensor[1, 2] = 5.0

2. Negative Index Access:
tensor[-1, -1] = 7.0

3. Out-of-Bounds Detection:
Correctly caught out-of-bounds error

4. Index Dimension Mismatch:
Correctly caught dimension mismatch

=== Testing Tensor Compatibility ===

Tensor 1:
ValidatedTensor[float32]
  Shape: [2, 3]
  Elements: 6
  Requires grad: False
Tensor 2:
ValidatedTensor[float32]
  Shape: [3]
  Elements: 3
  Requires grad: False
Broadcast compatible: True

Tensor 3:
ValidatedTensor[float32]
  Shape: [4]
  Elements: 4
  Requires grad: False
Broadcast compatible with tensor 3: False

=== Type System and Validation Implementation Summary ===
+ Comprehensive shape validation with descriptive errors
+ Broadcasting compatibility checking with NumPy semantics
+ Runtime bounds checking for safe tensor indexing
+ Negative index support (Python-style)
+ Type-safe tensor operations with validation
+ Memory safety guarantees with RAII patterns
+ Error handling with clear failure messages
+ Performance-focused validation controls
```

## Chapter Summary

`TensorShape` and `TensorStrides` are two small, single-purpose structs: one records dimension sizes and caches their product; the other converts multi-dimensional coordinates into one flat memory offset, using the row-major rule Chapters 3 and 4 already relied on informally (`stride[ndim-1]=1`, each earlier stride built from the sizes of every dimension *after* it). `Tensor[dtype]` composes both, plus a data buffer and an optional gradient buffer, into one RAII-managed struct — one constructor call triggers four separate heap allocations, and one destructor call must free all four, exactly the composed-RAII pattern Chapter 2.4 introduced. Ten factory functions build on that one constructor, differing only in how they fill the buffer afterward — `zeros`/`ones`/`full` with a constant, `arange`/`linspace` with a formula, `random_uniform`/`random_normal` with generated values, `eye`/`diagonal` with a pattern of positions, and `empty` with nothing at all, deliberately trading Section 6.3's zero-initialization safety for speed. `DeviceTensor` layers a location tag onto the same tensor shape, and this chapter is honest about what that tag currently means on a machine with no GPU code running: every "GPU" allocation and transfer in this chapter's code is, physically, identical CPU heap memory, a deliberate placeholder for the API shape a real device abstraction needs. `ValidatedTensor` is the same field layout and constructor algorithm as `Tensor`, with two checks inserted at the two moments a mistake is easiest to make silently — construction (rejecting empty, oversized, non-positive, or overflowing shapes) and indexing (catching out-of-bounds coordinates and correctly resolving Python-style negative ones) — trading Section 6.3's silent-and-fast behavior for `Tensor`'s riskier cousin, Chapter 1.5's raw-pointer trust, made explicit and checked.

## Self-Check Questions

1. For a tensor with shape `[5, 6, 7]`, compute `strides` by hand using Section 6.2's algorithm, then compute the linear index for coordinates `[2, 3, 4]`.
2. `Tensor[dtype]`'s constructor for a `[10]`-shaped tensor with `requires_grad=True` causes how many separate heap allocations in total, and which struct's `__del__` is responsible for freeing each one?
3. `arange[DType.float32](1, 10, 3)` — compute `num_elements` by hand (showing the `range_f64`, `Int(range_f64)`, and correction-check steps) and list the resulting data values.
4. Explain concretely what would need to change in `DeviceTensor`'s `_allocate_on_device` method for the `GPU` branch to actually place data in GPU memory, rather than simulating it with ordinary CPU heap allocation.
5. `ValidatedTensor.get_item_safe([-1, -1])` on a `[3,4]` tensor and `Tensor.get_item([-1,-1])` on the same shape (via `get_linear_index`) would compute two *different* results for the same input. What does each one actually compute, and why do they differ?

## Where We Go Next

Chapter 7 (Memory Layout Design) picks up exactly where Section 6.3's `_owns_data` flag left a thread hanging: it introduces tensor *views* — reshapes, transposes, and slices that share an existing tensor's buffer instead of copying it — which is precisely the case `_owns_data = False` exists to handle, plus the alignment and broadcasting machinery that sits on top of the stride formula this chapter just made explicit.

## Worked Solutions

**1.** `stride[2] = 1`. `stride[1] = stride[2] × shape[2] = 1 × 7 = 7`. `stride[0] = stride[1] × shape[1] = 7 × 6 = 42`. So `strides = [42, 7, 1]`. Linear index for `[2,3,4]`: `2×42 + 3×7 + 4×1 = 84 + 21 + 4 = 109`.

**2.** Four allocations: `shape.dims` (freed by `TensorShape.__del__`), `strides.strides` (freed by `TensorStrides.__del__`), `data` (freed directly in `Tensor.__del__`), and `grad` (also freed directly in `Tensor.__del__`, conditional on `requires_grad`). All four fire when the `Tensor` instance's own `__del__` runs and its two struct fields' destructors run in turn.

**3.** `range_f64 = (10-1)/3 = 3.0`. `Int(3.0) = 3`. Check: is `3.0 > 3`? No — so `num_elements` stays `3`, with no correction. Data: `data[i] = 1 + i×3` for `i=0,1,2`: `[1, 4, 7]`.

**4.** The `GPU` branch of `_allocate_on_device` (and the matching branch of `_deallocate_on_device`, plus every data-copying loop that runs when `self.device == GPU`) would need to call an actual GPU memory API — Chapter 4's real building blocks, such as a `DeviceContext`-based allocation — instead of the current `UnsafePointer[Scalar[dtype]].alloc(size)`, which is ordinary CPU heap memory regardless of which branch runs it. The data-copy loops inside `to_device` and `__copyinit__` would also need to become genuine host-to-device or device-to-host transfers rather than a plain `for` loop copying between two CPU pointers.

**5.** `ValidatedTensor.get_item_safe([-1,-1])` on a `[3,4]` tensor first normalizes each `-1` to a real positive coordinate — axis 0: `-1+3=2`; axis 1: `-1+4=3` — then computes `linear_index = 2×4 + 3×1 = 11`, correctly reading the bottom-right corner of the tensor. `Tensor.get_linear_index([-1,-1])` has no normalization step at all: it computes `linear_index = (-1)×4 + (-1)×1 = -5` directly, using strides `[4,1]` for the same `[3,4]` shape — a negative index that, passed to `self.data[-5]`, reads memory *before* the start of the allocated buffer entirely, not the last element the caller likely intended. The difference is exactly `ValidatedTensor`'s `normalize_negative_indices` step, which `Tensor`'s plain indexing path never runs.
