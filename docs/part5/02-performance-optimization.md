# Chapter 10: Performance Optimization

Optimization begins after correctness and ends with measured evidence. Use a scalar oracle, define the workload, warm up compilation and allocation, synchronize the measured device work, collect multiple samples, and report hardware and compiler versions beside the result.

## 10.1 SIMD keeps a scalar tail

Process full vectors with the target's natural width and finish the remainder scalarly. The width changes instruction grouping but not the element-wise contract.

```mojo
from std.sys.info import simdwidthof

def chunk_boundary[dtype: DType](size: Int) -> Int:
    comptime width = simdwidthof[dtype]()
    return (size // width) * width
```

**Manual worked example.** If width is 4 and size is 10, the boundary is `(10//4)×4=8`. Two vector operations cover indices 0–7 and a scalar tail covers 8–9.

## 10.2 Fusion reduces memory traffic

`relu(a*b+c)` can run in one pass rather than writing multiplication and addition intermediates. Fusion changes saved-value needs, so its custom backward must be checked against the unfused graph.

```mojo
def fused_scalar(a: Float32, b: Float32, c: Float32) -> Float32:
    var z = a * b + c
    return z if z > 0 else 0
```

**Manual worked example.** For `a=2`, `b=-3`, `c=7`, the fused intermediate is `2×-3+7=1`, so output is 1. For `c=5`, intermediate is -1 and ReLU returns 0. The fused and three-operation references must agree in both branches.

## 10.3 Compile-time specialization has a budget

Parameters such as SIMD width and small tile shape can remove dynamic checks. Specializing every dynamic dimension causes code-size growth and longer compilation, so specialize only a small measured set.

```mojo
def dot_fixed[width: Int](a: SIMD[DType.float32, width], b: SIMD[DType.float32, width]) -> Float32:
    return (a * b).reduce_add()
```

**Manual worked example.** Width 4 with `[1,2,3,4]` and `[5,6,7,8]` produces products `[5,12,21,32]` and sum 70. Width is absent from runtime inputs because it is baked into the specialization.

## 10.4 Benchmark medians after warm-up

First-call compilation and cache effects do not belong in steady-state kernel time. Warm up, synchronize, take repeated samples, and report the median plus a dispersion measure.

```text
for 5 warm-ups: run workload
synchronize
for 30 samples:
    record start event
    run workload
    record stop event
    synchronize stop event
report median, p10, p90
```

**Manual worked example.** Samples `[0.98,1.00,1.01,1.02,4.50]` ms have median 1.01 ms; mean is 1.702 ms because one interruption dominates it. Reporting the median and range makes that outlier visible without letting it define typical performance.

## 10.5 Roofline reasoning separates bandwidth and compute limits

Arithmetic intensity is operations per byte transferred. Compare it with hardware's compute-to-bandwidth ratio before spending time on instruction-level tuning.

```mojo
def arithmetic_intensity(flops: Float64, bytes_moved: Float64) -> Float64:
    return flops / bytes_moved
```

**Manual worked example.** Vector add performs one add while reading two Float32 values and writing one: 1 flop over 12 bytes, or about 0.083 flop/byte. That is strongly bandwidth-bound; reducing memory traffic matters more than shaving one arithmetic instruction.

No benchmark numbers are hard-coded in this chapter because credible results require the runnable source, input size, Mojo version, optimization flags, device, and measurement command.
