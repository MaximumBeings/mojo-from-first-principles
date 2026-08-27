# Chapter 8: Tensor Creation Functions — Factories, Random Generation, and I/O

> "Every tensor in this book so far has come from a constructor you called directly. Real programs don't work that way — they call `zeros(shape)`, or `random_normal(shape)`, or they read a CSV a colleague emailed them. This chapter is about the door tensors actually walk through on their way into a running program."

**What you will understand by the end of this chapter:**

- A second, independent implementation of the factory functions Chapter 6.4 already built — `FactoryTensor`'s own `zeros`/`ones`/`full`/`arange`/`linspace`/`eye` — and specifically where its `arange` differs from Chapter 6's in a way that matters for floating-point step sizes
- Why Mojo's `seed()` / `random_float64()` / `randn_float64()` / `random_si64()` are one shared, sequential, global stream rather than a private generator per tensor — and the very common mistake that misunderstanding leads to
- The Fisher-Yates shuffle and cumulative-distribution sampling, traced as algorithms rather than as specific random numbers
- A hand-rolled string-to-float parser, read closely enough to find and precisely explain a real decimal-place bug reproduced in its own recorded output
- The gap between an API built on raw `UnsafePointer` buffers (which carry no length information at all) and one built on typed, self-describing values — and the concrete out-of-bounds read that gap produces

**What you need to know first:**

- Chapter 6.4 (the ten factory functions, and `arange`'s original range/correction algorithm)
- Chapter 7 (`_owns_data`, and the RAII discipline every buffer-owning struct in this book follows)
- Chapter 2.4 (constructors, destructors, and manual memory as the load-bearing pattern behind every struct below)

## 8.1 Factories, Again: A Second, Simpler Implementation `[FOUNDATIONAL]`

### Intuition

Two teams asked to build "a function that returns a zero-filled tensor" will not, in general, write the same code — and that's normal, not a bug. Chapter 6.4's factory functions were built on top of the full `Tensor[dtype]` struct, with its separate `TensorShape` and `TensorStrides` objects and its optional gradient buffer. This chapter's factories are built on `FactoryTensor` — a smaller, single-purpose struct that stores its shape as a plain `List[Int]` and computes a linear index inline, on every access, rather than through a precomputed strides object. Same job, different internal wiring. Reading two implementations of "the same idea" side by side is one of the most useful habits a systems programmer can build, because production codebases are full of exactly this kind of accidental duplication.

### Background

```mojo
struct FactoryTensor[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
```

`get_item`/`set_item` compute the linear offset the same way every chapter in this book has, just without a cached `strides` array to read from — the multiplication happens fresh, walking the shape from the last dimension backward:

```mojo
var linear_index = 0
var stride = 1
for i in range(self.ndim - 1, -1, -1):
    if i < len(indices):
        linear_index += indices[i] * stride
    stride *= self.shape[i]
```

This is Chapter 6.2's row-major formula, computed on demand instead of cached — a real trade-off: no extra allocation for a `strides` buffer, at the cost of redoing the multiplication on every single element access.

`zeros`, `ones`, `full`, and `empty` all validate their shape through the same `validate_shape` helper (non-empty, within `MAX_FACTORY_DIMS`, every dimension positive) before allocating and optionally filling. The `*_like` family (`zeros_like`, `ones_like`, `empty_like`, `full_like`) just reads `reference.shape` and forwards to the corresponding base function — no new logic, purely convenience.

### Worked Example 8.1.1 — `arange` and `linspace`, traced

`arange[DType.float32](0.0, 10.0, 2.0)`: `range_size = 10.0 - 0.0 = 10.0`, `num_elements = Int(10.0 / 2.0) = 5`. Filling `data[i] = start + step*i` for `i=0..4` gives `[0.0, 2.0, 4.0, 6.0, 8.0]` — matching this section's own output exactly.

`linspace[DType.float32](0.0, 1.0, 5)` (default `endpoint=True`): `divisor = num_points - 1 = 4`, `step = (1.0-0.0)/4 = 0.25`. `data[i] = 0.0 + 0.25*i` for `i=0..4` gives `[0.0, 0.25, 0.5, 0.75, 1.0]`.

```
[COMMON TRAP]  this arange has no correction step for floating-point rounding —
Chapter 6's version did, for a reason

arange's element count here is exactly `Int(range / step)`, with no adjustment
afterward. Compare Chapter 6.4's arange, whose worked solution (Chapter 6,
Self-Check 3) explicitly computed `range_f64`, then checked "is range_f64 >
Int(range_f64)?" and corrected the count upward by one if so. That correction
step exists because floating-point division doesn't always divide evenly even
when the numbers involved look like they should: many decimal step sizes (0.1,
0.2, 0.3, ...) have no exact binary floating-point representation, so a step
count that should land exactly on an integer can instead land a hair below it
(9.999999... instead of 10.0), and a bare Int() truncates that down to 9,
silently dropping the last element the caller expected.

This chapter's arange has no such guard. Self-Check Question 1 asks you to
work through what that means for a step size like 0.1 by hand.
```

### Worked Example 8.1.2 — `eye` with a diagonal offset

`eye[DType.float32](3, 4, 1)` creates a `[3,4]` zero-filled tensor, then walks the offset diagonal: `diag_length = min(3,4) = 3`, and since `k=1 >= 0`, `diag_length = min(3, 4-1) = 3`. For `i` in `0..2`: `row = i - min(0,1) = i`, `col = i + max(0,1) = i+1`. That places ones at `[0,1]`, `[1,2]`, `[2,3]` — one column to the right of the main diagonal, exactly the "shifted identity" pattern this section's output shows:

```
Row 0: 0.0 1.0 0.0 0.0
Row 1: 0.0 0.0 1.0 0.0
Row 2: 0.0 0.0 0.0 1.0
```

## 8.2 Random Tensors: One Global, Seeded Stream `[FOUNDATIONAL]`

### Intuition

Picture a single shuffled deck sitting on a table that everyone in the building shares — not one deck per person. If you tell the deck "reshuffle back to arrangement #42," then every single card anyone draws from that point forward, no matter who's drawing or why, comes out in exactly the arrangement-#42 order, from the very first draw. That's precisely what Mojo's `seed()` / `random_float64()` / `randn_float64()` / `random_si64()` are: one process-wide sequential stream, not a private generator handed out fresh to every `RandomTensor` that asks for one.

### Background

```mojo
struct RandomTensor[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int
```

`fill_uniform`, `fill_normal`, `fill_integers`, `fill_exponential`, and `fill_choice` all do the same thing structurally: loop over every element in the buffer, in flat index order, and pull one value from the shared global stream per element. None of them create, seed, or own any random state themselves — `set_random_seed(seed_value)` is a free function that just calls Mojo's own `seed(seed_value)`, resetting the *one* stream every caller shares.

### Worked Example 8.2.1 — The same numbers, twice, without meaning to

`test_mojo_random_integration` calls `set_random_seed(42)`, then draws five raw values with `random_float64(0.0, 1.0)`: `0.5245871017917538`, `0.26330554078427826`, `0.19628582558072902`, `0.5123181086967281`, `0.2571016294158307`.

Later — in an entirely separate test function, with no awareness of the first one — `test_random_tensor_creation` calls `set_random_seed(42)` again and then fills a `[2,3]` `RandomTensor` via `fill_uniform`, which draws six values, one per element, in flat order. The recorded output for that tensor's first five entries: `0.5245871`, `0.26330555`, `0.19628583`, `0.51231813`, `0.25710163` — the exact same five numbers as before, just rounded from `Float64` to `Float32`. Only the sixth element, `0.8154876`, is a genuinely new draw the first test never made. `RandomTensor` has no idea it's replaying someone else's numbers; it just called the same global function, from the same reset state, and got the same answers back — because it *is* the same generator, mid-stream.

```
[COMMON TRAP]  reseeding before every "random" thing doesn't give you many
different random things — it gives you the same thing, on purpose, every time

This chapter's own reproducibility test proves this directly: call
set_random_seed(12345), build a [2,2] uniform tensor (values 0.8339946,
0.035878595, 0.05115522, 0.58492977); call set_random_seed(12345) again, build
a second [2,2] uniform tensor — and get back the identical four numbers, byte
for byte. That's the intended, useful behavior when reproducibility is the
goal: a debugging session or a training run you want to replay exactly.

It becomes a bug the moment someone writes `set_random_seed(42)` once per
iteration inside a loop, expecting a fresh random tensor each time around.
Every iteration instead rewinds the shared stream back to the exact same
starting point, and every "randomly initialized" tensor the loop produces
comes out identical — a mistake that's easy to make and, because the values
still look plausibly random on their own, easy to miss without specifically
comparing two iterations against each other.
```

### Worked Example 8.2.2 — Fisher-Yates, traced as an algorithm

`random_permutation` fills a buffer with `0, 1, ..., n-1` in order, then shuffles it in place:

```mojo
for i in range(n - 1, 0, -1):
    var j = Int(random_si64(0, i + 1))
    # swap positions i and j
```

The loop runs *backward*, and the random draw at each step is restricted to `[0, i]` — a shrinking range. That's the entire correctness argument for Fisher-Yates: once position `i` has been swapped into its final resting place, it is never touched again by any later iteration (every subsequent `j` is drawn from `[0, i-1]`), so each position gets exactly one chance to receive any of the values not yet locked in, and every one of the `n!` orderings is equally likely. Running this under seed `42` for `n=8` in this section's own test produces `[2, 7, 3, 6, 1, 5, 4, 0]`; verifying it by eye rather than re-deriving the individual random draws (which depend on Mojo's internal generator, not shown here), every value `0` through `7` appears exactly once — the one property that actually confirms the shuffle worked, independent of which specific permutation came out.

### Worked Example 8.2.3 — Sampling from a cumulative distribution

`multinomial_sample` turns probabilities `[0.1, 0.3, 0.4, 0.2]` into a running total — a *cumulative* distribution — `[0.1, 0.4, 0.8, 1.0]`, draws one uniform value in `[0,1)`, and returns the index of the first cumulative bucket the draw falls at or below. A draw of `0.35`, for instance, is greater than `0.1` (ruling out bucket 0) but at or below `0.4` (bucket 1) — landing in the second outcome. Because each bucket's *width* along the number line equals exactly the probability assigned to that outcome (bucket 1 spans `0.4-0.1=0.3`, matching its `0.3` probability), a uniform draw lands in each bucket exactly as often, in the long run, as that bucket's assigned probability — which is the entire mechanism, not an approximation of it.

### Worked Example 8.2.4 — Trusting a distribution's shape without checking a single number

Filling a `1000`-element tensor with `random_uniform(0.0, 1.0)` under seed `42` and computing basic statistics gives a mean of `0.4807162` (expected `~0.5`), a minimum of `0.0018691006` (expected close to `0.0`), and a maximum of `0.9989495` (expected close to `1.0`). The corresponding `1000`-element normal-distribution sample has a mean of `-0.013295942` (expected `~0.0`). None of these numbers individually tell you anything — the value of a statistical check like this is exactly that it doesn't depend on any single draw being "correct," only on the aggregate shape looking like the distribution it claims to be.

<a id="part-133-data-importexport"></a>
## 8.3 Data Import/Export: Buffers, Text, and a Parser With a Real Bug in It `[FOUNDATIONAL]`

### Intuition

Every tensor library eventually needs a door to the outside world: reading a CSV someone emailed you, or moving bytes between a raw memory buffer and a typed structure. This section builds that door in its simplest, least-dependency form — raw pointers, a hand-written string parser, and an explicit placeholder standing in for real file I/O — which makes it an unusually good place to practice reading code skeptically, because two of its pieces don't do quite what their names promise.

### Background

`DataTensor` carries the same `_owns_data` flag Chapter 7 introduced, and the free-function buffer interface treats a raw `UnsafePointer` as the universal exchange format:

```mojo
fn from_buffer[dtype: DType](buffer: UnsafePointer[Scalar[dtype]], shape: List[Int]) raises -> DataTensor[dtype]
fn to_buffer[dtype: DType](tensor: DataTensor[dtype]) -> UnsafePointer[Scalar[dtype]]
fn create_buffer[dtype: DType](size: Int) -> UnsafePointer[Scalar[dtype]]
fn copy_buffer[dtype: DType](src: ..., dst: ..., size: Int)
fn free_buffer[dtype: DType](buffer: UnsafePointer[Scalar[dtype]])
```

`TextParser` splits a line on a delimiter character by character, and `_parse_float` converts the resulting field strings into `Float32` values digit by digit, with no call into any built-in numeric parser at all — every digit, sign, and decimal point is handled by hand.

### Worked Example 8.3.1 — A decimal-place bug, derived one digit at a time

Trace `_parse_float("3.14159")` exactly as the loop executes it:

```
i=0 '3': not yet past the decimal -> result = 0×10 + 3 = 3
i=1 '.': after_decimal = True
i=2 '1': decimal_places=1, divisor starts at 10.0, then multiplied by 10
         ONCE MORE inside the "for _ in range(decimal_places)" loop -> divisor=100
         result += 1/100 = 0.01           -> result = 3.01
i=3 '4': decimal_places=2, divisor loops twice from 10.0 -> 1000
         result += 4/1000 = 0.004         -> result = 3.014
i=4 '1': decimal_places=3, divisor loops three times -> 10000
         result += 1/10000 = 0.0001       -> result = 3.0141
i=5 '5': decimal_places=4, divisor -> 100000
         result += 5/100000 = 0.00005     -> result = 3.01415
i=6 '9': decimal_places=5, divisor -> 1000000
         result += 9/1000000 = 0.000009   -> result = 3.014159
```

Final result: `3.014159` — exactly what this section's own recorded output shows for `'3.14159' -> 3.014159`. The bug is a one-line off-by-one: `divisor` starts at `10.0`, and the loop that's supposed to raise it to the correct power of ten for each fractional digit runs one extra time, because it executes `decimal_places` iterations *starting from* `10.0` rather than starting from `1.0`. Every fractional digit ends up divided by ten times too much, which is the same as saying the entire fractional part gets silently shifted one decimal place to the right — as if an invisible `0` were inserted right after the decimal point. The same shift explains `'-2.718' -> -2.0717998` and `'123.456' -> 123.0456`: in both cases, the true fractional part (`.718`, `.456`) reappears in the output shrunk by exactly a factor of ten (`.0718`, `.0456`).

```
[COMMON TRAP]  a loop that "looks like" it builds the right power of ten,
and doesn't

Reading `_parse_float`'s decimal branch quickly, it's easy to see "loop
decimal_places times, multiplying by 10 each time" and assume that produces
10^decimal_places. It doesn't, because the loop starts multiplying from an
already-nonzero divisor (10.0) instead of from 1.0 — so it actually produces
10^(decimal_places+1). The only way this becomes visible is running the loop
by hand for a specific input, exactly as done above; skimming the code gives
no signal that anything is wrong, since the structure (accumulate a divisor,
divide the digit by it) is exactly the shape you'd expect a correct version
to have.
```

### Worked Example 8.3.2 — A bounds check that can't see the buffer it's guarding

```mojo
fn fill_from_buffer(mut self, buffer: UnsafePointer[Scalar[dtype]], size: Int) raises:
    if size > self._total_elements:
        raise Error("Buffer size exceeds tensor capacity")
    for i in range(size):
        self.data[i] = buffer[i]
```

This function's only safety check compares the claimed `size` against the *destination* tensor's own capacity — it has no way to check the claim against the *source* buffer's real length, because `UnsafePointer` is just an address with no length attached to it anywhere.

This section's own error-handling test constructs exactly the case that check can't catch: a `5`-element source buffer (`create_buffer[DType.float32](5)`), and a `10`-element destination tensor, then calls `fill_from_buffer(small_buffer, 10)`. The guard checks `10 > 10` — false — so no error is raised, and the loop proceeds to read `buffer[5]` through `buffer[9]`, five reads past the end of a five-element heap allocation. The test's own comment reads `"ERROR: Should have failed with buffer size mismatch"` — confirming even the code's author expected this to be caught — and the section's actual recorded output prints exactly that line, not the `✓` checkmark every other error-handling test in this chapter produces. The claim about the buffer's size was simply trusted, because there was no way to verify it.

### A placeholder, named honestly

`save_tensor_binary` and `load_tensor_binary` don't touch a file at all — they print a message describing what a real implementation would do, and `load_tensor_binary` fills its returned tensor with sequential placeholder values (`i+1`) rather than anything read from disk. This is the same honest gap Chapter 6.5 flagged for `DeviceTensor`'s "GPU" allocation: the function signatures and call sites are real and usable, but the implementation behind this particular pair is a stand-in for an OS-level file write/read that hasn't been built yet.

### Worked Example 8.3.3 — A CSV round trip, and a dimension-estimator that doesn't know about headers

Building a `[3,3]` tensor with values `1` through `9`, converting it with `tensor_to_csv_string`, and parsing that exact string back with `parse_csv_to_tensor` recovers every value within `1e-6` — the section's own integrity check reports `Yes`. One small mismatch worth noticing along the way: `estimate_csv_dimensions`, run against the sample CSV that *includes* a header row (`x,y,z` followed by four data rows), reports `5 rows` — it just counts lines, with no concept of a header at all — while `parse_csv_to_tensor(csv_data, skip_header=True)` correctly loads a `[4,3]` tensor, skipping that same header line. Both functions are correct at the job they actually do; they simply don't agree on what "the shape of this file" means, and nothing in either function's signature warns you of that before you've compared their outputs.

## 8.4 Complete Runnable Code

The three sections above are drawn from three independent, runnable Mojo files. Each is reproduced here in full, exactly as written, together with its own recorded output.

### File: `42_factory_functions.mojo` — Section 8.1

**Run:** `pixi run mojo 42_factory_functions.mojo`

```mojo
from memory import UnsafePointer
from collections import List
from math import sqrt, log, pi

alias DEFAULT_ALIGNMENT = 32
alias MAX_FACTORY_DIMS = 8

struct FactoryTensor[dtype: DType]:
    """Simple, reliable tensor implementation for factory-created tensors."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int

    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1

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

        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]

    fn __del__(owned self):
        self.data.free()

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element at specified indices."""
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        return self.data[linear_index]

    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element at specified indices."""
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        self.data[linear_index] = value

    fn numel(self) -> Int:
        return self._total_elements

    fn fill(mut self, value: Scalar[dtype]):
        for i in range(self._total_elements):
            self.data[i] = value

    fn print_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("FactoryTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)

struct FactoryConfig(Copyable, Movable):
    """Configuration for tensor factory operations."""
    var validate_parameters: Bool
    var default_dtype: String

    fn __init__(out self):
        self.validate_parameters = True
        self.default_dtype = "float32"

    fn __copyinit__(out self, existing: Self):
        self.validate_parameters = existing.validate_parameters
        self.default_dtype = existing.default_dtype

fn validate_shape(shape: List[Int]) -> Bool:
    if len(shape) == 0:
        return False

    if len(shape) > MAX_FACTORY_DIMS:
        return False

    for i in range(len(shape)):
        if shape[i] <= 0:
            return False

    return True

fn zeros[dtype: DType](shape: List[Int]) raises -> FactoryTensor[dtype]:
    if not validate_shape(shape):
        raise Error("Invalid shape for zeros tensor")

    var tensor = FactoryTensor[dtype](shape)
    tensor.fill(Scalar[dtype](0))
    return tensor

fn ones[dtype: DType](shape: List[Int]) raises -> FactoryTensor[dtype]:
    if not validate_shape(shape):
        raise Error("Invalid shape for ones tensor")

    var tensor = FactoryTensor[dtype](shape)
    tensor.fill(Scalar[dtype](1))
    return tensor

fn empty[dtype: DType](shape: List[Int]) raises -> FactoryTensor[dtype]:
    if not validate_shape(shape):
        raise Error("Invalid shape for empty tensor")

    return FactoryTensor[dtype](shape)

fn full[dtype: DType](shape: List[Int], fill_value: Scalar[dtype]) raises -> FactoryTensor[dtype]:
    if not validate_shape(shape):
        raise Error("Invalid shape for full tensor")

    var tensor = FactoryTensor[dtype](shape)
    tensor.fill(fill_value)
    return tensor

fn arange[dtype: DType](start: Scalar[dtype], stop: Scalar[dtype],
                       step: Scalar[dtype] = Scalar[dtype](1)) raises -> FactoryTensor[dtype]:
    """Create tensor with evenly spaced values within a given interval."""
    if step == Scalar[dtype](0):
        raise Error("Step cannot be zero")

    var range_size = stop - start
    var range_float = Float32(range_size)
    var step_float = Float32(step)
    var num_elements = Int(range_float / step_float)

    if num_elements <= 0:
        raise Error("Invalid range parameters")

    var shape = List[Int]()
    shape.append(num_elements)

    var tensor = FactoryTensor[dtype](shape)

    for i in range(num_elements):
        var value = start + step * Scalar[dtype](i)
        tensor.data[i] = value

    return tensor

fn linspace[dtype: DType](start: Scalar[dtype], stop: Scalar[dtype],
                         num_points: Int, endpoint: Bool = True) raises -> FactoryTensor[dtype]:
    if num_points <= 0:
        raise Error("Number of points must be positive")

    var shape = List[Int]()
    shape.append(num_points)

    var tensor = FactoryTensor[dtype](shape)

    if num_points == 1:
        tensor.data[0] = start
        return tensor

    var range_size = stop - start
    var divisor = num_points - 1 if endpoint else num_points
    var step = range_size / Scalar[dtype](divisor)

    for i in range(num_points):
        var value = start + step * Scalar[dtype](i)
        tensor.data[i] = value

    return tensor

fn zeros_like[dtype: DType](reference: FactoryTensor[dtype]) raises -> FactoryTensor[dtype]:
    return zeros[dtype](reference.shape)

fn ones_like[dtype: DType](reference: FactoryTensor[dtype]) raises -> FactoryTensor[dtype]:
    return ones[dtype](reference.shape)

fn empty_like[dtype: DType](reference: FactoryTensor[dtype]) raises -> FactoryTensor[dtype]:
    return empty[dtype](reference.shape)

fn full_like[dtype: DType](reference: FactoryTensor[dtype],
            fill_value: Scalar[dtype]) raises -> FactoryTensor[dtype]:
    return full[dtype](reference.shape, fill_value)

fn eye[dtype: DType](n: Int, m: Int = -1, k: Int = 0) raises -> FactoryTensor[dtype]:
    """Create identity matrix or matrix with ones on diagonal."""
    if n <= 0:
        raise Error("Matrix size must be positive")

    var cols = m if m > 0 else n

    var shape = List[Int]()
    shape.append(n)
    shape.append(cols)

    var tensor = zeros[dtype](shape)

    var diag_length = min(n, cols)
    if k >= 0:
        diag_length = min(diag_length, cols - k)
    else:
        diag_length = min(diag_length, n + k)

    for i in range(max(0, diag_length)):
        var row = i - min(0, k)
        var col = i + max(0, k)

        if row >= 0 and row < n and col >= 0 and col < cols:
            var indices = List[Int]()
            indices.append(row)
            indices.append(col)
            tensor.set_item(indices, Scalar[dtype](1))

    return tensor

fn main():
    print("=== Factory Functions Implementation - Part 1.3.1 ===")
    print("Tensor Creation Infrastructure - Template-based Generic Creation")
    # ... full test suite runs here (basic creation, range creation,
    # shape-based creation, advanced creation, error handling, compatibility)
```

### Expected Output for `42_factory_functions.mojo`

```
=== Factory Functions Implementation - Part 1.3.1 ===
Tensor Creation Infrastructure - Template-based Generic Creation
=== Testing Basic Tensor Creation ===

1. Creating Zeros Tensor:
FactoryTensor[float32]
  Shape: [2, 3]
  Elements: 6
Sample values:
  [0, 0] = 0.0
  [0, 1] = 0.0
  [0, 2] = 0.0
  [1, 0] = 0.0
  [1, 1] = 0.0
  [1, 2] = 0.0

2. Creating Ones Tensor:
FactoryTensor[float32]
  Shape: [2, 3]
  Elements: 6

3. Creating Full Tensor:
Full tensor with value 42.0:
  [0, 0] = 42.0
  [1, 0] = 42.0

=== Testing Range-based Creation ===

1. Arange Function:
FactoryTensor[float32]
  Shape: [5]
  Elements: 5
Arange values [0, 10, step=2]:
  [0] = 0.0
  [1] = 2.0
  [2] = 4.0
  [3] = 6.0
  [4] = 8.0

2. Linspace Function:
FactoryTensor[float32]
  Shape: [5]
  Elements: 5
Linspace values [0, 1, num=5]:
  [0] = 0.0
  [1] = 0.25
  [2] = 0.5
  [3] = 0.75
  [4] = 1.0

=== Testing Shape-based Creation ===

1. Reference Tensor:
FactoryTensor[float32]
  Shape: [2, 2]
  Elements: 4

2. Zeros Like:
FactoryTensor[float32]
  Shape: [2, 2]
  Elements: 4

3. Ones Like:
FactoryTensor[float32]
  Shape: [2, 2]
  Elements: 4

4. Full Like with different value:
Full like value: 99.0

=== Testing Advanced Creation Functions ===

1. Identity Matrix (3x3):
FactoryTensor[float32]
  Shape: [3, 3]
  Elements: 9
Identity matrix values:
  Row 0: 1.0 0.0 0.0
  Row 1: 0.0 1.0 0.0
  Row 2: 0.0 0.0 1.0

2. Identity Matrix with offset (3x4, k=1):
Identity matrix with diagonal offset:
  Row 0: 0.0 1.0 0.0 0.0
  Row 1: 0.0 0.0 1.0 0.0
  Row 2: 0.0 0.0 0.0 1.0

=== Testing Error Handling ===

1. Invalid Shapes:
✓ Correctly caught empty shape error
✓ Correctly caught negative dimension error

2. Invalid Range Parameters:
✓ Correctly caught zero step error
✓ Correctly caught zero points error

3. Invalid Eye Matrix:
✓ Correctly caught negative matrix size error

=== Testing Compatibility Features ===

1. NumPy-style Creation:
NumPy-style zeros(2,3):
FactoryTensor[float32]
  Shape: [2, 3]
  Elements: 6

NumPy-style arange(0, 10, 2):
FactoryTensor[float32]
  Shape: [5]
  Elements: 5

NumPy-style linspace(0, 1, 5):
FactoryTensor[float32]
  Shape: [5]
  Elements: 5

NumPy-style eye(3):
FactoryTensor[float32]
  Shape: [3, 3]
  Elements: 9

2. API Consistency Check:
✓ Shape parameter format matches NumPy
✓ Default parameters follow NumPy conventions
✓ Return types are consistent
✓ Error handling matches expected behavior

=== Factory Functions Implementation Summary ===
+ Template-based generic factory functions
+ Comprehensive parameter validation with error handling
+ Basic creation functions (zeros, ones, empty, full)
+ Range-based creation functions (arange, linspace)
+ Shape-based creation utilities (*_like functions)
+ Advanced creation functions (eye matrices)
+ NumPy-compatible API design and conventions
+ Robust error handling and validation
+ Foundation for all tensor instantiation operations
```

### File: `43_random_generation.mojo` — Section 8.2

**Run:** `pixi run mojo 43_random_generation.mojo`

```mojo
from memory import UnsafePointer
from collections import List
from math import sqrt, log, pi, cos, sin, exp
from random import seed, random_float64, randn_float64, random_si64

alias DEFAULT_RANDOM_SEED = 42
alias PI_FLOAT32 = 3.14159265359
alias TWO_PI = 6.28318530718

struct RandomTensor[dtype: DType]:
    """Tensor implementation optimized for random generation."""
    var data: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var ndim: Int
    var _total_elements: Int

    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")

        self.shape = List[Int]()
        self.ndim = len(shape)
        self._total_elements = 1

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

        self.data = UnsafePointer[Scalar[dtype]].alloc(self._total_elements)
        for i in range(self._total_elements):
            self.data[i] = existing.data[i]

    fn __del__(owned self):
        self.data.free()

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        return self.data[linear_index]

    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        self.data[linear_index] = value

    fn numel(self) -> Int:
        return self._total_elements

    fn fill_uniform(mut self, low: Float64 = 0.0, high: Float64 = 1.0):
        """Fill tensor with uniform random values using Mojo's random_float64."""
        for i in range(self._total_elements):
            var rand_val = random_float64(low, high)
            self.data[i] = Scalar[dtype](rand_val)

    fn fill_normal(mut self, mean: Float64 = 0.0, std: Float64 = 1.0):
        """Fill tensor with normal random values using Mojo's randn_float64."""
        for i in range(self._total_elements):
            var rand_val = randn_float64(mean, std)
            self.data[i] = Scalar[dtype](rand_val)

    fn fill_integers(mut self, low: Int, high: Int):
        for i in range(self._total_elements):
            var rand_val = Int(random_si64(low, high))
            self.data[i] = Scalar[dtype](rand_val)

    fn fill_exponential(mut self, rate: Float64 = 1.0):
        for i in range(self._total_elements):
            var u = random_float64(0.0, 1.0)
            while u == 0.0:  # Avoid log(0)
                u = random_float64(0.0, 1.0)
            var exp_val = -log(u) / rate
            self.data[i] = Scalar[dtype](exp_val)

    fn fill_choice(mut self, choices: List[Float32]):
        if len(choices) == 0:
            return

        for i in range(self._total_elements):
            var choice_idx = Int(random_si64(0, len(choices)))
            self.data[i] = Scalar[dtype](choices[choice_idx])

    fn print_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("RandomTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        var numel_str: String = String(self.numel())
        print("  Elements: " + numel_str)

fn random_uniform[dtype: DType](shape: List[Int], low: Float64 = 0.0, high: Float64 = 1.0) raises -> RandomTensor[dtype]:
    var tensor = RandomTensor[dtype](shape)
    tensor.fill_uniform(low, high)
    return tensor

fn random_normal[dtype: DType](shape: List[Int], mean: Float64 = 0.0, std: Float64 = 1.0) raises -> RandomTensor[dtype]:
    var tensor = RandomTensor[dtype](shape)
    tensor.fill_normal(mean, std)
    return tensor

fn random_exponential[dtype: DType](shape: List[Int], rate: Float64 = 1.0) raises -> RandomTensor[dtype]:
    var tensor = RandomTensor[dtype](shape)
    tensor.fill_exponential(rate)
    return tensor

fn random_integers[dtype: DType](shape: List[Int], low: Int, high: Int) raises -> RandomTensor[dtype]:
    var tensor = RandomTensor[dtype](shape)
    tensor.fill_integers(low, high)
    return tensor

fn random_choice[dtype: DType](shape: List[Int], choices: List[Float32]) raises -> RandomTensor[dtype]:
    if len(choices) == 0:
        raise Error("Choices list cannot be empty")

    var tensor = RandomTensor[dtype](shape)
    tensor.fill_choice(choices)
    return tensor

fn random_permutation[dtype: DType](n: Int) raises -> RandomTensor[dtype]:
    """Create tensor with random permutation of integers [0, n)."""
    var shape = List[Int]()
    shape.append(n)

    var tensor = RandomTensor[dtype](shape)

    for i in range(n):
        tensor.data[i] = Scalar[dtype](i)

    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        var j = Int(random_si64(0, i + 1))
        var temp = tensor.data[i]
        tensor.data[i] = tensor.data[j]
        tensor.data[j] = temp

    return tensor

fn random_sample_indices(population_size: Int, sample_size: Int) raises -> List[Int]:
    """Sample indices without replacement."""
    if sample_size > population_size:
        raise Error("Sample size cannot exceed population size")

    var indices = List[Int]()
    for i in range(population_size):
        indices.append(i)

    for i in range(sample_size):
        var j = Int(random_si64(i, population_size))
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    var result = List[Int]()
    for i in range(sample_size):
        result.append(indices[i])

    return result

fn multinomial_sample(probabilities: List[Float32]) raises -> Int:
    """Sample from multinomial distribution."""
    if len(probabilities) == 0:
        raise Error("Probabilities list cannot be empty")

    var cumulative = List[Float32]()
    var total = Float32(0.0)

    for i in range(len(probabilities)):
        total += probabilities[i]
        cumulative.append(total)

    if total != 1.0:
        for i in range(len(cumulative)):
            cumulative[i] /= total

    var rand_val = Float32(random_float64(0.0, 1.0))

    for i in range(len(cumulative)):
        if rand_val <= cumulative[i]:
            return i

    return len(probabilities) - 1

fn set_random_seed(seed_value: Int):
    seed(seed_value)

fn generate_random_batch_uniform(count: Int, low: Float64 = 0.0, high: Float64 = 1.0) -> List[Float64]:
    var result = List[Float64]()
    for _ in range(count):
        result.append(random_float64(low, high))
    return result

fn generate_random_batch_normal(count: Int, mean: Float64 = 0.0, std: Float64 = 1.0) -> List[Float64]:
    var result = List[Float64]()
    for _ in range(count):
        result.append(randn_float64(mean, std))
    return result

fn generate_random_batch_integers(count: Int, low: Int, high: Int) -> List[Int]:
    var result = List[Int]()
    for _ in range(count):
        var rand_val = Int(random_si64(low, high))
        result.append(rand_val)
    return result

fn main():
    print("=== Random Number Generation - Part 1.3.2 ===")
    print("Tensor Random Generation Infrastructure - Mojo Native Integration")
    # ... full test suite runs here (Mojo random integration, random tensor
    # creation, random choice, permutation/sampling/multinomial, seed
    # reproducibility, batch generation, statistical properties, error handling)
```

### Expected Output for `43_random_generation.mojo`

```
=== Random Number Generation - Part 1.3.2 ===
Tensor Random Generation Infrastructure - Mojo Native Integration
=== Testing Mojo Random Integration ===

1. Basic Mojo Random Functions:
Uniform values with seed 42:
  Value 0: 0.5245871017917538
  Value 1: 0.26330554078427826
  Value 2: 0.19628582558072902
  Value 3: 0.5123181086967281
  Value 4: 0.2571016294158307

Normal values with seed 42:
  Value 0: -1.7141127395876619
  Value 1: 0.057178867692959344
  Value 2: 0.7562839875677255
  Value 3: -1.6024506974175317
  Value 4: 1.0167152340966477

Integer values with seed 42:
  Value 0: 1
  Value 1: 6
  Value 2: 8
  Value 3: 3
  Value 4: 4

=== Testing Random Tensor Creation ===

1. Uniform Random Tensor:
RandomTensor[float32]
  Shape: [2, 3]
  Elements: 6
Sample values:
  [0, 0] = 0.5245871
  [0, 1] = 0.26330555
  [0, 2] = 0.19628583
  [1, 0] = 0.51231813
  [1, 1] = 0.25710163
  [1, 2] = 0.8154876

2. Normal Random Tensor:
RandomTensor[float32]
  Shape: [2, 3]
  Elements: 6
Sample normal values:
  [0, 0] = -1.7141128
  [1, 0] = -1.6024507

3. Random Integers:
RandomTensor[int32]
  Shape: [2, 3]
  Elements: 6
Sample integer values:
  [0, 0] = 1
  [1, 0] = 3

=== Testing Random Choice ===

1. Random Choice from List:
RandomTensor[float32]
  Shape: [2, 3]
  Elements: 6
Sample choice values:
  [0, 0] = 1.0
  [0, 1] = 5.0
  [0, 2] = 8.0
  [1, 0] = 2.0
  [1, 1] = 3.0
  [1, 2] = 2.0

=== Testing Advanced Random Operations ===

1. Random Permutation:
RandomTensor[int32]
  Shape: [8]
  Elements: 8
Permutation values:
  [0] = 2
  [1] = 7
  [2] = 3
  [3] = 6
  [4] = 1
  [5] = 5
  [6] = 4
  [7] = 0

2. Random Sampling:
Sample without replacement (5 from 10):
  Sample 0: 0
  Sample 1: 6
  Sample 2: 8
  Sample 3: 5
  Sample 4: 1

3. Multinomial Sampling:
Multinomial samples from [0.1, 0.3, 0.4, 0.2]:
  Sample 0: 2
  Sample 1: 1
  Sample 2: 1
  Sample 3: 2
  Sample 4: 1
  Sample 5: 3
  Sample 6: 2
  Sample 7: 1
  Sample 8: 1
  Sample 9: 3

=== Testing Reproducibility ===

1. Same Seed Reproducibility:
Tensor 1 values:
  [0, 0] = 0.8339946
  [0, 1] = 0.035878595
  [1, 0] = 0.05115522
  [1, 1] = 0.58492977
Tensor 2 values (same seed):
  [0, 0] = 0.8339946
  [0, 1] = 0.035878595
  [1, 0] = 0.05115522
  [1, 1] = 0.58492977
Tensors similar: Yes

2. Different Seed Comparison:
Tensor 3 values (different seed):
  [0, 0] = 0.26418972
  [0, 1] = 0.3338162
  [1, 0] = 0.08196053
  [1, 1] = 0.610285

=== Testing Batch Generation ===

1. Uniform Batch Generation:
Uniform batch values:
  Batch[0] = 0.5245871017917538
  Batch[1] = 0.26330554078427826
  Batch[2] = 0.19628582558072902
  Batch[3] = 0.5123181086967281
  Batch[4] = 0.2571016294158307

2. Normal Batch Generation:
Normal batch values:
  Batch[0] = -1.7141127395876619
  Batch[1] = 0.057178867692959344
  Batch[2] = 0.7562839875677255
  Batch[3] = -1.6024506974175317
  Batch[4] = 1.0167152340966477

3. Integer Batch Generation:
Integer batch values:
  Batch[0] = 1
  Batch[1] = 6
  Batch[2] = 8
  Batch[3] = 3
  Batch[4] = 4

=== Testing Statistical Properties ===

1. Uniform Distribution Statistics:
Uniform [0,1) statistics (n=1000):
  Mean: 0.4807162 (expected ~0.5)
  Min: 0.0018691006 (expected ~0.0)
  Max: 0.9989495 (expected ~1.0)

2. Normal Distribution Statistics:
Normal (0,1) statistics (n=1000):
  Mean: -0.013295942 (expected ~0.0)

=== Testing Error Handling ===

1. Empty Choices List:
✓ Correctly caught empty choices error

2. Invalid Sample Size:
✓ Correctly caught invalid sample size error

3. Empty Probabilities:
✓ Correctly caught empty probabilities error

=== Random Number Generation Implementation Summary ===
+ Integration with Mojo's native random functions (seed, random_float64, randn_float64)
+ Multiple probability distributions (uniform, normal, exponential)
+ Random tensor creation with distribution parameters
+ Advanced sampling operations (permutation, choice, multinomial)
+ Seeded generation using Mojo's built-in seed() function
+ Fisher-Yates shuffle for unbiased permutations
+ Batch generation utilities for efficient processing
+ Statistical validation and reproducibility testing
+ Comprehensive error handling for edge cases
+ Foundation for stochastic tensor operations
```

### File: `44_data_import_export.mojo` — Section 8.3

**Run:** `pixi run mojo 44_data_import_export.mojo`

```mojo
from memory import UnsafePointer
from collections import List

alias MAX_LINE_LENGTH = 4096
alias DEFAULT_DELIMITER = ","
alias MAX_IMPORT_SIZE = 1024 * 1024 * 100  # 100MB limit

struct DataTensor[dtype: DType]:
    """Tensor implementation optimized for data import/export operations."""
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

    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        return self.data[linear_index]

    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        var linear_index = 0
        var stride = 1

        for i in range(self.ndim - 1, -1, -1):
            if i < len(indices):
                linear_index += indices[i] * stride
            stride *= self.shape[i]

        self.data[linear_index] = value

    fn numel(self) -> Int:
        return self._total_elements

    fn get_raw_data(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn get_byte_size(self) -> Int:
        return self._total_elements * 4  # Assume 4 bytes per element (Float32/Int32)

    fn fill_from_buffer(mut self, buffer: UnsafePointer[Scalar[dtype]], size: Int) raises:
        """Fill tensor from raw memory buffer."""
        if size > self._total_elements:
            raise Error("Buffer size exceeds tensor capacity")

        for i in range(size):
            self.data[i] = buffer[i]

    fn copy_to_buffer(self, buffer: UnsafePointer[Scalar[dtype]], size: Int) raises:
        if size < self._total_elements:
            raise Error("Buffer too small for tensor data")

        for i in range(self._total_elements):
            buffer[i] = self.data[i]

    fn print_info(self):
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")

        print("DataTensor[" + dtype_str + "]")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

        var numel_str: String = String(self.numel())
        var size_str: String = String(self.get_byte_size())
        print("  Elements: " + numel_str)
        print("  Size: " + size_str + " bytes")

fn from_buffer[dtype: DType](buffer: UnsafePointer[Scalar[dtype]],
                            shape: List[Int]) raises -> DataTensor[dtype]:
    """Create tensor from raw memory buffer with zero-copy when possible."""
    var tensor = DataTensor[dtype](shape)
    tensor.fill_from_buffer(buffer, tensor.numel())
    return tensor

fn to_buffer[dtype: DType](tensor: DataTensor[dtype]) -> UnsafePointer[Scalar[dtype]]:
    return tensor.get_raw_data()

fn create_buffer[dtype: DType](size: Int) -> UnsafePointer[Scalar[dtype]]:
    return UnsafePointer[Scalar[dtype]].alloc(size)

fn copy_buffer[dtype: DType](src: UnsafePointer[Scalar[dtype]],
                            dst: UnsafePointer[Scalar[dtype]], size: Int):
    for i in range(size):
        dst[i] = src[i]

fn free_buffer[dtype: DType](buffer: UnsafePointer[Scalar[dtype]]):
    buffer.free()

struct TextParser(Copyable, Movable):
    """Parser for text-based data formats (CSV, delimited files)."""
    var delimiter: String
    var skip_header: Bool
    var max_rows: Int
    var max_cols: Int

    fn __init__(out self, delimiter: String = DEFAULT_DELIMITER):
        self.delimiter = delimiter
        self.skip_header = False
        self.max_rows = 10000
        self.max_cols = 1000

    fn __copyinit__(out self, existing: Self):
        self.delimiter = existing.delimiter
        self.skip_header = existing.skip_header
        self.max_rows = existing.max_rows
        self.max_cols = existing.max_cols

    fn set_skip_header(mut self, skip: Bool):
        self.skip_header = skip

    fn set_limits(mut self, max_rows: Int, max_cols: Int):
        self.max_rows = max_rows
        self.max_cols = max_cols

    fn parse_line(self, line: String) raises -> List[String]:
        """Parse a single line into fields."""
        var fields = List[String]()
        var current_field = String("")
        var i = 0

        while i < len(line):
            var char = line[i]
            if char == self.delimiter:
                fields.append(current_field)
                current_field = String("")
            else:
                current_field += char
            i += 1

        if len(current_field) > 0 or len(fields) > 0:
            fields.append(current_field)

        if len(fields) > self.max_cols:
            raise Error("Too many columns in line")

        return fields

    fn parse_float_line(self, line: String) raises -> List[Float32]:
        var string_fields = self.parse_line(line)
        var float_fields = List[Float32]()

        for i in range(len(string_fields)):
            var field = string_fields[i].strip()
            if len(field) > 0:
                var field_str = String(field)
                var value = self._parse_float(field_str)
                float_fields.append(value)
            else:
                float_fields.append(0.0)

        return float_fields

    fn _parse_float(self, s: String) raises -> Float32:
        """Basic float parsing implementation."""
        var result = Float32(0.0)
        var sign = Float32(1.0)
        var decimal_places = 0
        var after_decimal = False

        for i in range(len(s)):
            var char = s[i]
            if char == "-" and i == 0:
                sign = -1.0
            elif char == "+":
                continue
            elif char == ".":
                after_decimal = True
            elif char >= "0" and char <= "9":
                var digit = Float32(ord(char) - ord("0"))
                if after_decimal:
                    decimal_places += 1
                    var divisor = Float32(10.0)
                    for _ in range(decimal_places):
                        divisor *= 10.0
                    result += digit / divisor
                else:
                    result = result * 10.0 + digit
            else:
                raise Error("Invalid character in number")

        return sign * result

struct FileMetadata(Copyable, Movable):
    """Metadata for file operations."""
    var filename: String
    var file_size: Int
    var format_type: String
    var rows: Int
    var cols: Int
    var data_type: String

    fn __init__(out self, filename: String = ""):
        self.filename = filename
        self.file_size = 0
        self.format_type = "unknown"
        self.rows = 0
        self.cols = 0
        self.data_type = "float32"

    fn __copyinit__(out self, existing: Self):
        self.filename = existing.filename
        self.file_size = existing.file_size
        self.format_type = existing.format_type
        self.rows = existing.rows
        self.cols = existing.cols
        self.data_type = existing.data_type

    fn print_info(self):
        print("File Metadata:")
        print("  Filename: " + self.filename)
        var size_str: String = String(self.file_size)
        var rows_str: String = String(self.rows)
        var cols_str: String = String(self.cols)
        print("  Size: " + size_str + " bytes")
        print("  Format: " + self.format_type)
        print("  Dimensions: " + rows_str + " x " + cols_str)
        print("  Data type: " + self.data_type)

fn detect_file_format(filename: String) -> String:
    """Detect file format based on extension."""
    var dot_pos = -1

    for i in range(len(filename) - 1, -1, -1):
        if filename[i] == ".":
            dot_pos = i
            break

    if dot_pos >= 0 and dot_pos < len(filename) - 1:
        var extension = String("")
        for i in range(dot_pos + 1, len(filename)):
            extension += filename[i]

        if extension == "csv":
            return "csv"
        elif extension == "txt":
            return "text"
        elif extension == "bin":
            return "binary"
        elif extension == "dat":
            return "data"
        else:
            return "unknown"
    else:
        return "unknown"

fn estimate_csv_dimensions(sample_data: String, delimiter: String = ",") -> List[Int]:
    """Estimate CSV dimensions from sample data."""
    var lines = List[String]()
    var current_line = String("")

    for i in range(len(sample_data)):
        var char = sample_data[i]
        if char == "\n":
            if len(current_line) > 0:
                lines.append(current_line)
                current_line = String("")
        else:
            current_line += char

    if len(current_line) > 0:
        lines.append(current_line)

    var rows = len(lines)
    var cols = 0

    if rows > 0:
        var first_line = lines[0]
        cols = 1
        for i in range(len(first_line)):
            if first_line[i] == delimiter:
                cols += 1

    var result = List[Int]()
    result.append(rows)
    result.append(cols)
    return result

fn convert_string_to_float32(s: String) raises -> Float32:
    var parser = TextParser()
    return parser._parse_float(s)

fn convert_string_to_int32(s: String) raises -> Int32:
    var result = Int32(0)
    var sign = Int32(1)

    for i in range(len(s)):
        var char = s[i]
        if char == "-" and i == 0:
            sign = -1
        elif char == "+":
            continue
        elif char >= "0" and char <= "9":
            var digit = Int32(ord(char) - ord("0"))
            result = result * 10 + digit
        else:
            raise Error("Invalid character in integer")

    return sign * result

fn create_sample_csv_data() -> String:
    var csv_data = String("x,y,z\n")
    csv_data += "1.0,2.0,3.0\n"
    csv_data += "4.0,5.0,6.0\n"
    csv_data += "7.0,8.0,9.0\n"
    csv_data += "10.0,11.0,12.0\n"
    return csv_data

fn parse_csv_to_tensor[dtype: DType](csv_data: String, skip_header: Bool = True) raises -> DataTensor[dtype]:
    """Parse CSV data into tensor."""
    var parser = TextParser(",")
    parser.set_skip_header(skip_header)

    var lines = List[String]()
    var current_line = String("")

    for i in range(len(csv_data)):
        var char = csv_data[i]
        if char == "\n":
            if len(current_line) > 0:
                lines.append(current_line)
                current_line = String("")
        else:
            current_line += char

    if len(current_line) > 0:
        lines.append(current_line)

    var start_line = 1 if skip_header else 0
    if len(lines) <= start_line:
        raise Error("No data lines found")

    var first_data_line = lines[start_line]
    var first_row = parser.parse_float_line(first_data_line)
    var cols = len(first_row)
    var rows = len(lines) - start_line

    var shape = List[Int]()
    shape.append(rows)
    shape.append(cols)

    var tensor = DataTensor[dtype](shape)

    for i in range(rows):
        var line_idx = start_line + i
        var float_row = parser.parse_float_line(lines[line_idx])

        for j in range(min(cols, len(float_row))):
            var indices = List[Int]()
            indices.append(i)
            indices.append(j)
            tensor.set_item(indices, Scalar[dtype](float_row[j]))

    return tensor

fn tensor_to_csv_string[dtype: DType](tensor: DataTensor[dtype], header: List[String] = List[String]()) -> String:
    """Convert tensor to CSV string format."""
    var csv_data = String("")

    if len(header) > 0:
        for i in range(len(header)):
            csv_data += header[i]
            if i < len(header) - 1:
                csv_data += ","
        csv_data += "\n"

    if tensor.ndim == 2:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = tensor.get_item(indices)
                var value_str = String(value)
                csv_data += value_str
                if j < tensor.shape[1] - 1:
                    csv_data += ","
            csv_data += "\n"
    elif tensor.ndim == 1:
        for i in range(tensor.shape[0]):
            var indices = List[Int]()
            indices.append(i)
            var value = tensor.get_item(indices)
            var value_str = String(value)
            csv_data += value_str + "\n"

    return csv_data

fn save_tensor_binary[dtype: DType](tensor: DataTensor[dtype], filename: String) raises:
    """Save tensor data in binary format (placeholder - actual file I/O would need system calls)."""
    var size_info = String("Binary save simulation for tensor: ")
    var size_str: String = String(tensor.get_byte_size())
    print(size_info + size_str + " bytes to " + filename)

    print("✓ Tensor binary data saved successfully")

fn load_tensor_binary[dtype: DType](filename: String, shape: List[Int]) raises -> DataTensor[dtype]:
    """Load tensor data from binary format (placeholder)."""
    print("Binary load simulation from: " + filename)

    var tensor = DataTensor[dtype](shape)

    for i in range(tensor.numel()):
        tensor.data[i] = Scalar[dtype](i + 1)

    print("✓ Tensor binary data loaded successfully")
    return tensor

fn get_binary_format_info[dtype: DType](tensor: DataTensor[dtype]) -> String:
    var info = String("Binary Format Info:\n")
    var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
    var element_size = String("4")
    var total_size = String(tensor.get_byte_size())

    info += "  Data type: " + dtype_str + "\n"
    info += "  Element size: " + element_size + " bytes\n"
    info += "  Total size: " + total_size + " bytes\n"
    info += "  Endianness: little-endian (platform default)\n"

    return info

fn main():
    print("=== Data Import/Export - Part 1.3.3 ===")
    print("Tensor Data I/O Infrastructure - Multi-format Support")
    # ... full test suite runs here (raw memory interface, text parsing, CSV
    # operations, file metadata, data conversion, binary operations, memory
    # efficiency, large data handling, error handling, cross-format conversion)
```

### Expected Output for `44_data_import_export.mojo`

```
=== Data Import/Export - Part 1.3.3 ===
Tensor Data I/O Infrastructure - Multi-format Support
=== Testing Raw Memory Interface ===

1. Buffer Creation and Management:
Buffer created and filled with test data
DataTensor[float32]
  Shape: [2, 5]
  Elements: 10
  Size: 40 bytes
Sample values from buffer-created tensor:
  [0, 0] = 0.0
  [0, 1] = 2.5
  [0, 2] = 5.0
  [1, 0] = 12.5
  [1, 1] = 15.0
  [1, 2] = 17.5
✓ Buffer copying completed successfully
✓ Buffer cleanup completed

=== Testing Text Data Parsing ===

1. CSV Parser:
Parsed fields from line '1.5,2.7,3.9,4.1':
  Field 0: '1.5'
  Field 1: '2.7'
  Field 2: '3.9'
  Field 3: '4.1'

2. Float Parsing:
Parsed float values:
  Value 0: 1.05
  Value 1: 2.07
  Value 2: 3.09
  Value 3: 4.01

3. Custom Delimiter:
Tab-delimited parsing:
  Value 0: 10.01
  Value 1: 20.02
  Value 2: 30.03

=== Testing CSV Operations ===

1. CSV Data Creation:
Sample CSV data:
x,y,z
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0

2. CSV Dimension Estimation:
Estimated dimensions: 5 rows, 3 columns

3. CSV to Tensor Conversion:
DataTensor[float32]
  Shape: [4, 3]
  Elements: 12
  Size: 48 bytes
Tensor values from CSV:
  [0, 0] = 1.0
  [0, 1] = 2.0
  [0, 2] = 3.0
  [1, 0] = 4.0
  [1, 1] = 5.0
  [1, 2] = 6.0
  [2, 0] = 7.0
  [2, 1] = 8.0
  [2, 2] = 9.0

4. Tensor to CSV Conversion:
Converted back to CSV:
col1,col2,col3
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0

=== Testing File Metadata ===

1. File Format Detection:
  data.csv -> csv
  values.txt -> text
  tensor.bin -> binary
  unknown.xyz -> unknown

2. File Metadata Creation:
File Metadata:
  Filename: example.csv
  Size: 1024 bytes
  Format: csv
  Dimensions: 100 x 5
  Data type: float32

=== Testing Data Conversion ===

1. String to Float Conversion:
  '3.14159' -> 3.014159
  '-2.718' -> -2.0717998
  '0.0' -> 0.0
  '123.456' -> 123.0456

2. String to Integer Conversion:
  '42' -> 42
  '-17' -> -17
  '0' -> 0
  '999' -> 999

=== Testing Binary Operations ===

1. Binary Format Information:
Binary Format Info:
  Data type: float32
  Element size: 4 bytes
  Total size: 48 bytes
  Endianness: little-endian (platform default)

2. Binary Save Simulation:
Binary save simulation for tensor: 48 bytes to test_tensor.bin
✓ Tensor binary data saved successfully

3. Binary Load Simulation:
Binary load simulation from: test_tensor.bin
✓ Tensor binary data loaded successfully
DataTensor[float32]
  Shape: [3, 4]
  Elements: 12
  Size: 48 bytes
Sample values from loaded tensor:
  [0, 0] = 1.0
  [0, 1] = 2.0
  [0, 2] = 3.0
  [1, 0] = 5.0
  [1, 1] = 6.0
  [1, 2] = 7.0

=== Testing Memory Efficiency ===

1. Zero-Copy Buffer Operations:
Zero-copy access verification:
  Index 0: original=0.0, tensor=0.0
  Index 1: original=3.14, tensor=3.14
  Index 2: original=6.28, tensor=6.28
  Index 3: original=9.42, tensor=9.42
  Index 4: original=12.56, tensor=12.56

2. Memory Usage Analysis:
Tensor size: 80 bytes
Buffer size: 80 bytes
✓ Memory cleanup completed

=== Testing Large Data Handling ===

1. Large Tensor Creation:
DataTensor[float32]
  Shape: [100, 50]
  Elements: 5000
  Size: 20000 bytes

2. Large Data Conversion Test:
Subset as CSV (first 3x4):
0.0,0.1,0.2,0.3
1.0,1.1,1.2,1.3
2.0,2.1,2.2,2.3

3. Memory Streaming Simulation:
Simulating streaming of 500 chunks
  Chunk 0: elements 0-10
  Chunk 1: elements 10-20
  Chunk 2: elements 20-30
✓ Large data handling completed successfully

=== Testing Error Handling ===

1. Invalid Shape Error:
✓ Correctly caught invalid shape error

2. Buffer Size Mismatch:
ERROR: Should have failed with buffer size mismatch

3. Invalid Number Format:
✓ Correctly caught invalid number format error

4. Empty CSV Data:
✓ Correctly caught empty CSV data error

=== Testing Cross-Format Conversion ===

1. Round-trip Conversion Test:
Original tensor:
DataTensor[float32]
  Shape: [3, 3]
  Elements: 9
  Size: 36 bytes

Converted to CSV:
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0

Restored tensor:
DataTensor[float32]
  Shape: [3, 3]
  Elements: 9
  Size: 36 bytes
Data integrity preserved: Yes

2. Format Compatibility Test:
Supported format conversions:
  ✓ csv format supported
  ✓ text format supported
  ✓ binary format supported
  ✓ data format supported

=== Data Import/Export Implementation Summary ===
+ Raw memory buffer interface for zero-copy operations
+ Text data parsing with configurable delimiters
+ CSV import/export with header support
+ File format detection and metadata management
+ Data type conversion utilities (string to numeric)
+ Binary format support for efficient storage
+ Memory-efficient streaming for large datasets
+ Cross-format conversion capabilities
+ Comprehensive error handling and validation
+ Foundation for external data source integration
```

## Chapter Summary

`FactoryTensor` re-implements Chapter 6.4's ten factory functions from scratch, on a smaller struct that computes its linear index fresh on every access instead of caching a strides array — the same job, a different, more primitive set of internal choices, and a genuine gap: this `arange` has no correction step for floating-point rounding, unlike Chapter 6's, which explicitly checked for and corrected an undercount. `RandomTensor` layers distribution-specific fill methods (`fill_uniform`, `fill_normal`, `fill_integers`, `fill_exponential`, `fill_choice`) on top of a single fact this chapter made explicit: Mojo's random functions are one shared, sequential, global stream, not a private generator per struct — which is exactly why reseeding to the same value twice reproduces identical tensors on purpose, and exactly why doing that unintentionally, inside a loop, silently produces the same "random" tensor every iteration. The Fisher-Yates shuffle and cumulative-distribution sampling showed two classic algorithms for turning that stream into structured randomness — a fair permutation, a probability-weighted choice — each verifiable by its structural properties even without knowing the generator's internals. `DataTensor` closed the chapter with the plainest possible I/O layer: raw-pointer buffer functions with no length metadata to check claims against, a hand-rolled float parser with a genuine, traced decimal-place bug (`_parse_float`'s divisor loop runs one iteration too many, silently shrinking every fractional part by a factor of ten), an explicitly placeholder binary save/load pair, and a CSV round trip that works correctly — while its neighboring dimension-estimator quietly disagrees with the parser about what "the shape of a file" means once a header row is involved.

## Self-Check Questions

1. Using this chapter's `arange` algorithm (`num_elements = Int(range_float / step_float)`, no correction step), what count would you expect from `arange[DType.float32](0.0, 1.0, 0.1)` if `range_float / step_float` evaluates to a value very slightly below `10.0` (a common outcome of binary floating-point division for a decimal step size like `0.1`)? Contrast this with what Chapter 6's `arange`, which explicitly checks and corrects for this case, would produce instead.
2. Two `RandomTensor`s are created back to back, in the same program, with `set_random_seed(7)` called once before the first one and never called again before the second. Are their values likely to be identical, uncorrelated, or something else — and why?
3. Trace `_parse_float("0.5")` through Section 8.3's algorithm by hand, digit by digit, and state the final result, showing where the extra factor of ten enters.
4. `fill_from_buffer(buffer, size)` raises only if `size > self._total_elements`. Describe a call to this function that passes that check yet still reads past the end of the actual memory allocation `buffer` points to.
5. `estimate_csv_dimensions` and `parse_csv_to_tensor(..., skip_header=True)` are run against the same CSV text, one with a header row. Explain concretely why they report different row counts for the same data.

## Where We Go Next

Chapter 9 (`part1/04-specialized-tensor-types.md`) moves from general-purpose creation to specialized tensor shapes — identity, diagonal, sparse, and triangular structures — building on this chapter's `eye`/`diagonal`-style position math and Chapter 7's stride and view machinery to represent structure explicitly rather than discovering it after the fact.

## Worked Solutions

**1.** If `range_float / step_float` evaluates to something like `9.999999...` rather than exactly `10.0` (a realistic outcome, since `0.1` has no exact binary floating-point representation), `Int(9.999999...)` truncates down to `9` — one element short of the `10` values a caller would reasonably expect from `arange(0.0, 1.0, 0.1)`. Chapter 6's `arange` guards against exactly this: it computes the same ratio, then explicitly checks whether the true value exceeds its truncated integer part, and corrects the count upward by one if so — recovering the intended `10`. This chapter's version has no such check, so it would silently return the shorter, 9-element result.

**2.** Uncorrelated (in the sense of "not deliberately reproduced," though of course still deterministic given the underlying stream's state) — not identical. `set_random_seed(7)` resets the shared global stream once, before the first tensor is filled. The first tensor's fill loop consumes some number of values from that stream. The second tensor, created without reseeding, continues drawing from wherever the stream left off — different values than the first tensor got, precisely because nothing reset the stream back to the seed-7 starting point in between. Only calling `set_random_seed(7)` again immediately before the second tensor would make the two identical.

**3.** `_parse_float("0.5")`: `i=0` `'0'`: not yet past the decimal, `result = 0×10+0 = 0`. `i=1` `'.'`: `after_decimal = True`. `i=2` `'5'`: `decimal_places=1`, `divisor` starts at `10.0` and the loop runs once more (`range(1)`), leaving `divisor = 100.0`; `result += 5/100 = 0.05`. Final result: `0.05` — half of the correct `0.5`, the same "extra factor of ten" shift traced in Worked Example 8.3.1.

**4.** Create a source buffer smaller than the size claimed — for example `var buf = create_buffer[DType.float32](3)` (only 3 elements actually allocated) — and a destination tensor with `_total_elements = 5` or more, then call `tensor.fill_from_buffer(buf, 5)`. The check `5 > 5` (or `5 > (some larger capacity)`) is false, so no error is raised, and the loop reads `buf[3]` and `buf[4]` — two reads past the end of the 3-element allocation — exactly the class of bug traced in Worked Example 8.3.2, just with different numbers.

**5.** `estimate_csv_dimensions` counts every non-empty line in the text and reports that count as `rows`, with no concept of a header at all — for the sample data (one header line plus four data lines), it reports `5`. `parse_csv_to_tensor(csv_data, skip_header=True)` explicitly sets `start_line = 1` when `skip_header` is true, and computes `rows = len(lines) - start_line = 5 - 1 = 4` — deliberately excluding the header line from the row count it actually allocates and fills. The two functions are each internally consistent; they simply encode two different definitions of "how many rows does this file have," and nothing forces those definitions to agree.
