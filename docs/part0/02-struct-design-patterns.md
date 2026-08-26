# 0.2 Struct Design Patterns

Mojo structs are value types with declared fields and compiler-checked lifetimes. A good struct makes invalid states hard to construct, states its copy behavior, and owns resources through a destructor only when a higher-level owner cannot do the job.

## 0.2.1 Fieldwise initialization for transparent data

Use `@fieldwise_init` when every field is already valid independently and construction needs no extra checks.

```mojo
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

    def squared_norm(self) -> Float32:
        return self.x * self.x + self.y * self.y
```

**Manual worked example.** `Point(3,4)` computes `3²+4²=9+16=25`. The squared norm avoids a square root and is enough for comparisons.

## 0.2.2 Custom initialization enforces invariants

When fields depend on one another, write `__init__` and reject invalid inputs before assigning a complete value.

```mojo
struct Interval:
    var low: Float64
    var high: Float64

    def __init__(out self, low: Float64, high: Float64) raises:
        if high < low:
            raise Error("interval endpoints are reversed")
        self.low = low
        self.high = high
```

**Manual worked example.** `Interval(2,5)` is valid and has width 3. `Interval(5,2)` raises rather than creating a negative-width interval that every later method would need to defend against.

## 0.2.3 Copy behavior is explicit

Mojo 1.0 copy initializers use `__init__(out self, *, copy: Self)`. For reference-semantic storage, copy the smart pointer; for value semantics, copy the underlying fields.

```mojo
struct Counter(ImplicitlyCopyable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value
```

**Manual worked example.** Copy a counter holding 7, then increment only the copy to 8. The original remains 7 because the integer field was duplicated; this is value semantics.

## 0.2.4 Traits specify behavior, not storage

A trait lists required operations. Concrete structs choose their own representation while generic functions depend only on the promised behavior.

```mojo
trait Area:
    def area(self) -> Float64: ...

@fieldwise_init
struct Rectangle(Area):
    var width: Float64
    var height: Float64

    def area(self) -> Float64:
        return self.width * self.height
```

**Manual worked example.** A rectangle of width 3 and height 5 returns `3×5=15`. Another shape can conform to `Area` with different fields without changing callers that need only `area()`.

## 0.2.5 Destructors consume `self`

Mojo 1.0 spells the destructor `__deinit__` and uses `deinit self`. Prefer automatic owners such as `List`, `OwnedPointer`, `ArcPointer`, and `DeviceBuffer`; custom destruction is for genuinely owned low-level resources.

```mojo
struct TracedResource:
    var id: Int

    def __deinit__(deinit self):
        print("released", self.id)
```

**Manual worked example.** A resource with ID 42 prints `released 42` once at its last use. Copying resource owners without an explicit shared-ownership policy would risk double release, which is why ownership choice precedes destructor code.

The tensor framework follows these patterns: validated constructors, explicit copy/reference semantics, small behavioral traits, and automatic owners wherever possible.
