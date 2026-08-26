# 1.3 Tensor Creation Functions

Factories are part of the tensor's correctness boundary. They validate shape once, allocate the exact number of elements, initialize every readable value, and return a tensor whose metadata already satisfies the invariants from Section 1.1.

## 1.3.1 Zeros, ones, and full share one implementation

One `full` factory prevents repeated allocation logic. `zeros` and `ones` become descriptive wrappers rather than separate constructors that can drift.

```mojo
def full(shape: Shape, value: Float32) raises -> Tensor:
    var tensor = Tensor(shape)
    for i in range(len(tensor.data)):
        tensor.data[i] = value
    return tensor

def zeros(shape: Shape) raises -> Tensor:
    return full(shape, 0)

def ones(shape: Shape) raises -> Tensor:
    return full(shape, 1)
```

**Manual worked example.** `full([2,3],7)` calculates six elements, allocates six slots, and writes 7 into indices 0–5. The flat result is `[7,7,7,7,7,7]`; `zeros` and `ones` produce the same length with 0 or 1.

## 1.3.2 Ranges derive length before allocation

For a positive integer step, ceiling division gives the number of values in the half-open interval `[start,stop)`. Rejecting zero and negative steps keeps this teaching implementation's contract narrow and unambiguous.

```mojo
def arange(start: Int, stop: Int, step: Int = 1) raises -> Tensor:
    if step <= 0 or stop < start:
        raise Error("arange requires stop>=start and step>0")
    var count = (stop - start + step - 1) // step
    var result = Tensor(Shape([count]))
    for i in range(count):
        result.data[i] = Float32(start + i * step)
    return result
```

**Manual worked example.** `arange(1,8,2)` computes count `(7+1)//2=4` and writes `1+0×2`, `1+1×2`, `1+2×2`, `1+3×2`: `[1,3,5,7]`. The next value 9 is outside the half-open stop 8.

## 1.3.3 Linspace includes both endpoints

`linspace` differs from `arange`: the caller chooses the number of points, and for more than one point the spacing divides by `count-1` so both endpoints are represented.

```mojo
def linspace(start: Float32, stop: Float32, count: Int) raises -> Tensor:
    if count <= 0:
        raise Error("linspace count must be positive")
    var result = Tensor(Shape([count]))
    if count == 1:
        result.data[0] = start
        return result
    var step = (stop - start) / Float32(count - 1)
    for i in range(count):
        result.data[i] = start + Float32(i) * step
    return result
```

**Manual worked example.** `linspace(0,1,5)` uses step `(1-0)/(5-1)=0.25` and yields `[0,0.25,0.5,0.75,1]`. With count 1, returning only `start` avoids division by zero and makes the edge case explicit.

## 1.3.4 Seeded random creation separates reproducibility from distribution

A factory should accept or construct an explicit random state. This excerpt shows the fill contract; the generator's statistical quality remains the responsibility of the standard-library RNG.

```mojo
from std.random import random_float64

def random_uniform(shape: Shape, low: Float32, high: Float32) raises -> Tensor:
    if not low < high:
        raise Error("uniform range must satisfy low < high")
    var result = Tensor(shape)
    for i in range(len(result.data)):
        var unit = Float32(random_float64())
        result.data[i] = low + (high - low) * unit
    return result
```

**Manual worked example.** For low 10, high 20, and a unit sample 0.25, the affine map gives `10+(20-10)×0.25=12.5`. A `[2,3]` shape consumes exactly six unit samples, and every output must satisfy `10≤x<20`.

## 1.3.5 Identity creation writes only its diagonal

The dense identity factory starts from known zeros and changes exactly `n` offsets. Use the specialized `IdentityTensor` from Section 1.3.4 when a dense buffer is unnecessary.

```mojo
def eye(size: Int) raises -> Tensor:
    if size < 0:
        raise Error("identity size must be nonnegative")
    var result = zeros(Shape([size, size]))
    for i in range(size):
        result.data[i * size + i] = 1
    return result
```

**Manual worked example.** For size 3, diagonal offsets are `0×3+0=0`, `1×3+1=4`, and `2×3+2=8`. Writing ones there produces `[[1,0,0],[0,1,0],[0,0,1]]`.

## 1.3.6 Binary I/O needs a schema

Raw bytes alone cannot recover dtype or shape. A durable format writes a versioned header containing magic bytes, dtype, rank, extents, and byte order before payload data, then validates the declared payload length while loading.

```text
header = magic | version | dtype | rank | extents | byte_order
payload_bytes = product(extents) * bytes_per_element(dtype)
```

**Manual worked example.** A Float32 tensor of shape `[2,3]` declares six elements and therefore 24 payload bytes. If the file contains only 20 payload bytes, loading must fail as truncated; inferring five elements and changing the shape would silently corrupt the model.

The factory layer now has one allocation path, explicit edge cases, reproducible random-state requirements, and a self-describing persistence contract.
