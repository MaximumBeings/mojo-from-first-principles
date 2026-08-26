# 0.5 SIMD and Vectorization

SIMD executes one instruction across several lanes. It changes how values are grouped, not the mathematical result, so every vectorized routine keeps a scalar reference and handles any tail that does not fill a complete vector.

## 0.5.1 SIMD values operate lane by lane

The width is a compile-time parameter because it changes the register type and generated instructions.

```mojo
def add_four(
    a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]
) -> SIMD[DType.float32, 4]:
    return a + b
```

**Manual worked example.** `[1,2,3,4]+[10,20,30,40]` produces `[11,22,33,44]`. No lane interacts with another during addition.

## 0.5.2 Reduction combines lanes

After lane-wise work, a reduction collapses the vector to one scalar. A dot product is multiply followed by sum.

```mojo
def dot_four(
    a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]
) -> Float32:
    return (a * b).reduce_add()
```

**Manual worked example.** Products of `[1,2,3,4]` and `[5,6,7,8]` are `[5,12,21,32]`; reduction gives `5+12+21+32=70`.

## 0.5.3 Vector loops need a tail

Process complete SIMD chunks first, then finish remaining elements scalarly. The tail is part of correctness, not an optional slow path.

```mojo
def chunk_and_tail(length: Int, width: Int) -> Tuple[Int, Int]:
    var vectorized = (length // width) * width
    return (vectorized, length - vectorized)
```

**Manual worked example.** Length 10 with width 4 has `floor(10/4)×4=8` vectorized elements and a tail of 2. Two four-lane operations cover indices 0–7; scalar work covers 8–9.

## 0.5.4 Alignment and bounds are separate facts

A pointer may be aligned yet too short for a full vector load, or long enough but misaligned. Prove both conditions or use APIs that state the weaker alignment safely.

```text
safe vector load requires:
    start + width <= logical_length
    and the load API's alignment contract is satisfied
```

**Manual worked example.** At start 8 in a ten-element buffer, a width-4 load would need indices 8–11 and is out of bounds even if address 8 is perfectly aligned. The tail loop must load indices 8 and 9 individually.

## 0.5.5 Benchmark against the scalar result

Measure only after the vectorized output matches a scalar baseline element by element within a justified floating-point tolerance.

```mojo
def close(a: Float32, b: Float32, tolerance: Float32 = 1e-5) -> Bool:
    return abs(a - b) <= tolerance * max(Float32(1), max(abs(a), abs(b)))
```

**Manual worked example.** Results 100.00000 and 100.00008 differ by 0.00008. The allowed error is `1e-5×100.00008≈0.001`, so they pass. Results 100 and 100.01 differ by 0.01 and fail.

SIMD is now grounded in lane arithmetic, reductions, tails, alignment, and validation—the same sequence used for tensor kernels later.
