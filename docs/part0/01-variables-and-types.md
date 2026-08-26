# 0.1 Variables and Types

Mojo's type system makes a crucial performance distinction visible: some facts are known only while the program runs, while others are compile-time parameters used to specialize code. Start with ordinary values, then introduce specialization only where it changes representation or generated instructions.

## 0.1.1 `var` creates mutable runtime state

Use `var` when a binding will change. An unannotated literal may be inferred, but explicit numerical types make precision and device ABI choices reviewable.

```mojo
def main():
    var principal = Float64(1000)
    var rate = Float64(0.05)
    principal *= 1 + rate
    print(principal)
```

**Manual worked example.** Growth is `1000×(1+0.05)=1000×1.05=1050`, so the program prints 1050. The mutation updates `principal`; `rate` remains 0.05.

## 0.1.2 `comptime` records a compile-time fact

A SIMD width or tile size may be a compile-time value because it changes generated code. It is not ordinary mutable state.

```mojo
def scaled_count(count: Int) -> Int:
    comptime lanes = 8
    return count * lanes
```

**Manual worked example.** With `count=3`, the function returns `3×8=24`. The compiler sees `lanes=8` while specializing the function; only `count` varies at runtime.

## 0.1.3 Parameters use square brackets

Mojo calls compile-time inputs parameters and runtime inputs arguments. A parameterized function can produce a distinct specialization for each parameter value.

```mojo
def dot[width: Int](a: SIMD[DType.float32, width], b: SIMD[DType.float32, width]) -> Float32:
    return (a * b).reduce_add()
```

**Manual worked example.** For width 4, `a=[1,2,3,4]` and `b=[5,6,7,8]`, lane products are `[5,12,21,32]`; reduction gives `5+12+21+32=70`. Width 8 would compile a different specialization.

## 0.1.4 Argument conventions state ownership

The default argument is an immutable read reference. `mut` exposes caller-visible mutation, `var` receives an owned local value, and `deinit` consumes and destroys a value. Use the weakest convention the function needs.

```mojo
def add_one(mut values: List[Int]):
    for i in range(len(values)):
        values[i] += 1
```

**Manual worked example.** Calling `add_one` on `[2,4,6]` mutates the caller's list to `[3,5,7]`. If the argument were the default read convention, assigning to `values[i]` would be rejected.

## 0.1.5 Fixed-width types cross device boundaries

Mojo 1.0 does not allow platform-sized `Int` or `UInt` as GPU kernel arguments because host and device widths may differ. Use `Int32`, `UInt32`, `Int64`, or `UInt64` at the boundary, then convert internally when needed.

```mojo
def kernel_length_to_index(length: Int32) -> Int:
    return Int(length)
```

**Manual worked example.** `Int32(1000)` has the same 32-bit representation on host and device. Converting it to `Int` inside the kernel yields the platform index 1000 without an ambiguous cross-device ABI.

These rules—runtime values, compile-time parameters, and explicit ownership—are the vocabulary used by every later tensor and kernel definition.
