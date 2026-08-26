# Chapter 5: Reduction Operations

Every loss function in Part 6 ends the same way: a tensor of per-example errors collapses into a single scalar. Reductions are how that collapse happens, and they're the one class of operation in this book where a naive parallel implementation is *wrong*, not just slow — summing floats in a different order changes the result, and summing them on a thousand threads simultaneously without a real reduction algorithm is a race condition.

## 5.1 Sum and Mean Reductions

The safe pattern is tree reduction: each thread combines two elements, then half as many threads combine those partial results, halving again each round until one value remains.

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

This is exactly the reduction structure already used to total a bond portfolio's present value in the `sum_reduce_kernel` reused in [Part 7](../part7/01-quantitative-finance-examples.md#step-1-computing-bond-prices) — the same primitive that sums squared errors for a loss function sums discounted cash flows for a bond book.

## 5.2 Min/Max Operations

Min/max reductions use the identical tree pattern with the combine step swapped from `+` to a comparison, and they carry an extra piece of state the backward pass needs: *which* index produced the extreme value, since `d(max(x))/dx_i` is 1 for the winning index and 0 everywhere else.

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

The `argmax` this produces feeds directly into classification metrics (predicted class = `argmax` of the output layer) and into the sparse gradient the backward pass scatters back through only the winning path.

## 5.3 Norm Calculations

The L2 norm is a sum-of-squares reduction followed by one square root: `‖x‖₂ = sqrt(Σ xᵢ²)`. It shows up twice in this book under different names — as the "loss" itself for a regression model, and as gradient-norm clipping, where a training step rescales every gradient tensor so `‖g‖₂` never exceeds a threshold, preventing a single bad batch from taking an enormous, destabilizing optimizer step.

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

## 5.4 Statistical Functions

Variance and standard deviation are sum-and-mean built twice — once for the mean, once for the mean of squared deviations from it — which is why they're placed last in this chapter rather than derived independently:

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

Batch normalization layers in [Chapter 11](../part6/01-neural-network-layers.md) call exactly this pair, per-feature, to normalize activations before every hidden layer — and Part 7's risk analytics call the same pair across a bond portfolio's yields to size a Value-at-Risk estimate.

With reductions in place, Part 2's tensor operations are complete: element-wise, matrix, and reduction operations cover every arithmetic primitive the rest of the book composes. Part 3 turns to *recording* those compositions as a graph, so the framework knows how to run them backward.
