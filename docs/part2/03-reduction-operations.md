# Chapter 5: Reduction Operations

A reduction combines many values into fewer values. Order now matters for floating-point arithmetic, so the reference algorithm states its initial value, empty-input policy, and numerical method explicitly.

## 5.1 Sum and mean

Compensated summation tracks low-order bits lost when a small value is added to a much larger running total. Mean divides only after a non-empty sum.

```mojo
def compensated_sum(values: List[Float64]) -> Float64:
    var total = 0.0
    var correction = 0.0
    for value in values:
        var adjusted = value - correction
        var next = total + adjusted
        correction = (next - total) - adjusted
        total = next
    return total
```

**Manual worked example.** `[1,4,9,16]` accumulates to 30, and mean is `30/4=7.5`. The correction is zero for these exact integers; it becomes useful when magnitudes differ sharply.

## 5.2 Min, max, and argmax

Initialize from the first element rather than an arbitrary sentinel, and reject empty input. A tie policy should be deterministic; this version keeps the first maximum.

```mojo
def argmax(values: List[Float32]) raises -> Int:
    if len(values) == 0:
        raise Error("argmax of empty input")
    var best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best
```

**Manual worked example.** For `[3,9,9,2]`, index 1 becomes best when 9 exceeds 3. Index 2 ties but does not exceed it, so the result remains the first maximum, index 1.

## 5.3 Norms and gradient clipping

The L2 norm is the square root of the sum of squares. Global norm clipping scales every gradient by the same factor only when the norm exceeds the limit.

```mojo
def clip_scale(norm: Float32, maximum: Float32) raises -> Float32:
    if maximum <= 0:
        raise Error("maximum norm must be positive")
    return maximum / norm if norm > maximum else 1
```

**Manual worked example.** Gradient `[3,4]` has norm `sqrt(9+16)=5`. With maximum 2.5, scale is `2.5/5=0.5`, giving `[1.5,2]` whose norm is 2.5. With maximum 10, scale remains 1.

## 5.4 Variance with Welford's algorithm

One-pass `mean(x²)-mean(x)²` can lose precision when values are large and tightly clustered. Welford updates mean and squared deviations stably in one pass.

```mojo
def population_variance(values: List[Float64]) raises -> Float64:
    if len(values) == 0:
        raise Error("variance of empty input")
    var mean = 0.0
    var m2 = 0.0
    var count = 0
    for value in values:
        count += 1
        var delta = value - mean
        mean += delta / Float64(count)
        m2 += delta * (value - mean)
    return m2 / Float64(count)
```

**Manual worked example.** For `[1,2,3]`, the mean becomes 1, then 1.5, then 2. The accumulated `m2` becomes 0, 0.5, then 2. Population variance is `2/3≈0.6667`; standard deviation is `sqrt(2/3)≈0.8165`.

GPU tree reductions must reproduce these policies, allowing only the floating-point differences justified by their changed summation order.
