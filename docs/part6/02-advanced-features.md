# Chapter 12: Advanced Features

Advanced features should extend a correct core without weakening its contracts. Custom derivatives state their mathematics, higher-order differentiation states whether backward itself is recorded, serialization validates a schema, and debugging compares analytic gradients with independent numerical evidence.

## 12.1 Custom autograd operations

A custom operation packages a forward solver and a derived vector-Jacobian rule. Its backward contract includes every saved value and domain assumption it needs.

```mojo
def implicit_root_backward(df_dx: Float64, df_dc: Float64, upstream: Float64) raises -> Float64:
    if abs(df_dx) < 1e-12:
        raise Error("implicit derivative is singular")
    return upstream * (-df_dc / df_dx)
```

**Manual worked example.** For `f(x,c)=x²-c=0` at `c=2`, `x=sqrt(2)`, `df/dx=2x≈2.82843` and `df/dc=-1`. With upstream 1, the result is `1/2.82843≈0.353553`, matching the centered finite difference from Chapter 7.

## 12.2 Higher-order derivatives require a differentiable backward graph

The scalar tape in Chapters 6–8 computes numerical first derivatives but does not record the operations executed by `local_backward`. Therefore it cannot truthfully promise second derivatives. Higher-order AD requires backward rules expressed in graph operations with `create_graph=True`, or a separate forward-over-reverse implementation.

```text
first derivative:  forward graph --reverse--> gradient values
second derivative: backward operations must themselves form a graph
```

**Manual worked example.** For `g(x)=x³` at `x=2`, `g'=3x²=12` and `g''=6x=12`. A first-order tape returns the first 12 as an ordinary number. Differentiating that stored number gives zero; only a recorded computation `3×x×x` preserves the path needed to obtain the second 12.

## 12.3 Serialization is versioned and transactional

Write a magic value, format version, dtype, rank, extents, and payload checksum. Load into temporary storage, validate everything, and publish weights only after the complete file passes.

```text
model header = magic | version | tensor_count
tensor entry = name | dtype | rank | extents | byte_count | checksum | payload
```

**Manual worked example.** A Float32 weight of shape `[2,3]` declares six elements and 24 payload bytes. If the entry declares 24 but only 20 bytes remain, loading fails before replacing the live model. A `[3,2]` tensor has the same byte count but still fails the expected-shape check.

## 12.4 Centered gradient checking

Centered differences cancel first-order truncation error and are usually more accurate than a one-sided check. Compare with a scale-aware denominator and choose epsilon appropriate to dtype.

```mojo
def centered_difference(f_plus: Float64, f_minus: Float64, epsilon: Float64) -> Float64:
    return (f_plus - f_minus) / (2 * epsilon)

def relative_error(analytic: Float64, numeric: Float64) -> Float64:
    return abs(analytic - numeric) / max(1.0, max(abs(analytic), abs(numeric)))
```

**Manual worked example.** For `f(x)=x²` at 3 with epsilon 0.001, `f(3.001)=9.006001` and `f(2.999)=8.994001`. Difference 0.012 divided by 0.002 gives 6, exactly the analytic derivative `2×3`.

## 12.5 Profiling preserves synchronization semantics

GPU timing must include an explicit completion event around the measured region; host enqueue time alone measures dispatch overhead. Warm up compilation and allocation before collecting repeated samples.

```text
warm up → synchronize → start event → enqueue kernels → stop event → synchronize → report median
```

**Manual worked example.** If enqueue returns in 20 microseconds but the GPU finishes 400 microseconds later, host wall time without synchronization reports roughly 20 microseconds and is wrong by 20×. Device events or synchronized boundaries measure the actual 400-microsecond region.

These features are advanced because they strengthen explicit contracts, not because they add opaque magic to the framework.
