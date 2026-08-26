# Chapter 5: Reduction Operations

Every loss function in Part 6 ends the same way: a tensor of per-example errors collapses into a single scalar. Reductions are how that collapse happens — a reduction takes many values and combines them into fewer, usually one, using an operation like sum, max, or average. This is the one class of operation in this book where a naive parallel implementation is *wrong*, not just slow: summing floats in a different order changes the result (floating-point addition isn't perfectly associative), and letting a thousand threads all write to the same output simultaneously without a real reduction algorithm is a race condition, not an optimization.

## 5.1 Sum and Mean Reductions

The safe pattern is *tree reduction*: pair up elements and combine each pair, then pair up the results and combine again, halving the array's length each round until one value remains. Work it by hand on `[1, 4, 9, 16]`. Round 1 pairs `(1,4)` and `(9,16)`, giving `[5, 25]`. Round 2 pairs `(5,25)`, giving `[30]`. Three additions total, same as a straight-line loop — but every pair *within* a round is independent, so a GPU can do all of round 1's additions simultaneously, then all of round 2's:

```
Round 0: [ 1,  4,  9, 16]
           \  /    \  /
Round 1: [  5   ,   25  ]
             \       /
Round 2: [     30      ]
```

```mojo
fn sum_reduce_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    """One round of pairwise reduction: size elements -> ceil(size/2)."""
    var tid = Int(thread_idx.x)
    if tid * 2 + 1 < size:
        output[tid] = input[tid * 2] + input[tid * 2 + 1]
    elif tid * 2 < size:
        output[tid] = input[tid * 2]      # odd leftover, pass through
    else:
        output[tid] = Float32(0.0)

fn tensor_sum(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    """Repeated halving until a single scalar remains."""
    var current = input
    var current_size = size
    var scratch = UnsafePointer[Scalar[DType.float32]].alloc(size)
    while current_size > 1:
        var next_size = (current_size + 1) // 2
        # launch sum_reduce_kernel with current_size threads, writing into scratch
        current = scratch
        current_size = next_size
    var result = current[0]
    scratch.free()
    return result

fn tensor_mean(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    return tensor_sum(input, size) / Float32(size)
```

`tensor_sum([1,4,9,16], 4)` returns `30`, matching the hand trace exactly; `tensor_mean` divides that by the count, `30 / 4 = 7.5`. This is the identical reduction structure already used to total a bond portfolio's present value in [Part 7's `sum_reduce_kernel`](../part7/01-quantitative-finance-examples.md#step-1-computing-bond-prices) — the same primitive that sums squared errors for a loss function sums discounted cash flows for a bond book, one pairwise round at a time.

## 5.2 Min/Max Operations

Min/max reductions use the identical tree pattern with the combine step swapped from addition to a comparison. Trace `[3, 7, 2, 9]` for max: round 1 compares `(3,7)→7` and `(2,9)→9`; round 2 compares `(7,9)→9`. The answer, `9`, is unsurprising — what matters is that the reduction also needs to remember *which position* produced it, `index 3`, because `d(max(x))/dx_i` is `1` for the winning index and `0` everywhere else, and the backward pass needs to know which single index gets that `1`.

```mojo
fn max_reduce_kernel(
    output_val: UnsafePointer[Scalar[DType.float32]],
    output_idx: UnsafePointer[Int],
    input: UnsafePointer[Scalar[DType.float32]],
    in_idx: UnsafePointer[Int],
    size: Int,
):
    var tid = Int(thread_idx.x)
    if tid * 2 + 1 < size:
        var left = input[tid * 2]
        var right = input[tid * 2 + 1]
        if left >= right:
            output_val[tid] = left
            output_idx[tid] = in_idx[tid * 2]
        else:
            output_val[tid] = right
            output_idx[tid] = in_idx[tid * 2 + 1]
    elif tid * 2 < size:
        output_val[tid] = input[tid * 2]
        output_idx[tid] = in_idx[tid * 2]
```

The `argmax` this produces (`index 3` in the trace above) feeds directly into classification metrics — [Chapter 11.4](../part6/01-neural-network-layers.md#114-optimizer-framework)'s `PerformanceMetrics` compares the argmax of a two-unit output layer against the true label exactly this way — and into the sparse gradient the backward pass scatters back through only the winning path.

## 5.3 Norm Calculations

The L2 norm collapses a vector into a single measure of its size: sum the squares of every entry, then take one square root. Worked on `[3, 4]`: squares are `[9, 16]`, sum is `25`, square root is `5` — the familiar 3-4-5 triangle, and a useful sanity check that `l2_norm` is implemented correctly before trusting it on real gradients.

```mojo
fn l2_norm(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    var squared = UnsafePointer[Scalar[DType.float32]].alloc(size)
    for i in range(size):
        squared[i] = input[i] * input[i]
    var sum_sq = tensor_sum(squared, size)
    squared.free()
    return sqrt(sum_sq)

fn clip_grad_norm(mut grad: UnsafePointer[Scalar[DType.float32]], size: Int, max_norm: Float32):
    var norm = l2_norm(grad, size)
    if norm > max_norm:
        var scale = max_norm / norm
        for i in range(size):
            grad[i] *= scale
```

The norm shows up twice in this book under different names. As a loss, it's the regression error itself. As **gradient-norm clipping**, it protects a training step: if a gradient vector's norm is `5.0` (the `[3,4]` example above) and `max_norm = 2.0`, `clip_grad_norm` computes `scale = 2.0 / 5.0 = 0.4` and rescales every entry — `[3, 4] → [1.2, 1.6]`, whose norm is now exactly `2.0` — preventing one unusually large gradient from taking an enormous, destabilizing optimizer step.

## 5.4 Statistical Functions

Variance and standard deviation are sum-and-mean built twice — once for the mean, once for the mean of squared deviations from it — which is why they're placed last in this chapter rather than derived independently. Work through `[2, 4, 4, 4, 5, 5, 7, 9]`, the textbook example precisely because its answer is a round number. Mean: `(2+4+4+4+5+5+7+9)/8 = 40/8 = 5`. Squared deviations from that mean: `[9, 1, 1, 1, 0, 0, 4, 16]`. Mean of *those*: `32/8 = 4` — the variance. Standard deviation is `√4 = 2`.

```mojo
fn tensor_variance(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    var mean = tensor_mean(input, size)
    var sq_dev = UnsafePointer[Scalar[DType.float32]].alloc(size)
    for i in range(size):
        var d = input[i] - mean
        sq_dev[i] = d * d
    var variance = tensor_mean(sq_dev, size)
    sq_dev.free()
    return variance

fn tensor_std(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    return sqrt(tensor_variance(input, size))
```

`tensor_variance` on the worked array returns `4.0`, and `tensor_std` returns `2.0`, matching the hand calculation exactly. Batch normalization layers in [Chapter 11](../part6/01-neural-network-layers.md) call exactly this pair, per-feature, to normalize activations before every hidden layer — subtracting the mean and dividing by the standard deviation is what keeps a network's internal activations from drifting to extreme values as training progresses — and Part 7's risk analytics call the same pair across a bond portfolio's yields to size a Value-at-Risk estimate.

With reductions in place, Part 2's tensor operations are complete: element-wise, matrix, and reduction operations cover every arithmetic primitive the rest of the book composes. Part 3 turns to *recording* those compositions as a graph, so the framework knows how to run them backward.
