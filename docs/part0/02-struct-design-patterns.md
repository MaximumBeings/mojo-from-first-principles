# Chapter 2: Structs — Building Your Own Types

> "Every type you didn't have to invent — `Int`, `Float32`, `SIMD[dtype, width]` — was built by someone else out of the same tool this chapter hands you: a fixed, named layout of memory, plus the functions that know what to do with it. A struct is not a container for data. It is a promise about a layout, exactly the way `Int` was a promise about eight bytes — except now you get to write the promise yourself."

**What you will understand by the end of this chapter:**

- What a `struct` actually is in memory — a fixed, contiguous, compile-time-known layout of named fields, and why that's a direct descendant of the C `struct`, not an invention of Mojo's
- Why Python has no real equivalent, and what Python programs use instead (a `dict`, or a `class` whose instances are secretly dictionaries under the hood) — and the real memory and speed cost of that substitution
- How constructor overloading (`__init__` written more than once) lets one struct offer several ways to be built, resolved entirely at compile time
- How a *parametric* struct (`Vector[dtype: DType, size: Int]`) gets compiled into a separate, fully specialized version of itself for every combination of parameters it's used with — Mojo's version of a C++ template
- How `__init__`/`__del__` pairs give a struct **RAII** — automatic, scope-tied cleanup of any heap memory it owns, closing the exact gap Chapter 1.5 left open by hand
- How a `trait` defines a required set of methods that any struct can satisfy, checked at compile time, without inheritance and without Python's runtime duck-typing gamble

**What you need to know first:**

- Chapter 1 in full — this chapter reuses its stack-layout reasoning (Section 1.1), its `alias` vs `var` distinction (Section 1.3), its `SIMD` lane model (Section 1.4), and its `UnsafePointer.alloc()`/`.free()` heap model (Section 1.5) without re-explaining them

## 2.1 What a Struct Actually Is `[FOUNDATIONAL]`

### Intuition

An architectural blueprint for a house doesn't describe *a* house — it describes the layout every house built from it will share. The blueprint fixes that the front door is at the north wall, the kitchen is 4 meters from it, the bedroom is behind the kitchen. Every house built from that exact blueprint has its kitchen in exactly the same relative position, and a contractor who has the blueprint can tell you "the kitchen sink" is 4 meters and 2 turns from the front door without ever having stepped inside *that particular* house — the blueprint alone is enough, because the layout is fixed and identical for every house built from it.

A `struct` is that blueprint, applied to a region of memory instead of a plot of land. `struct Point2D { var x: Float64; var y: Float64 }` fixes that field `x` sits at byte offset 0 and field `y` sits at byte offset 8, for *every* `Point2D` that will ever exist in the program. The compiler, like the contractor, never has to ask "where is `y` in this particular instance?" — the blueprint already answered that question once, for all instances, at compile time.

### Background

In C, a struct declaration is exactly this: a fixed, named, contiguous memory layout, with each field's byte offset computed once at compile time from the fields declared before it.

```c
struct Point2D {
    double x;   /* offset 0,  8 bytes */
    double y;   /* offset 8,  8 bytes */
};              /* total size: 16 bytes */
```

Mojo's `struct` is the direct descendant of this — same fixed-offset, contiguous-layout guarantee — with methods attached directly to the type (something C requires you to fake with free functions taking a `struct Point2D*` as their first argument) and Mojo's ownership rules (Chapter 1's stack/heap distinction) governing how instances are passed around.

```mojo
struct Point2D:
    """Basic struct representing a 2D point."""
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

    fn distance_from_origin(self) -> Float64:
        return pow(self.x * self.x + self.y * self.y, 0.5)
```

| | C `struct` | Mojo `struct` | Python `class` |
|---|---|---|---|
| Field layout | Fixed offsets, known at compile time | Fixed offsets, known at compile time | No fixed offsets — a per-instance `dict` |
| Methods | Not supported (free functions instead) | Attached directly to the type | Attached directly to the type |
| Field access cost | One offset add, no lookup | One offset add, no lookup | A hash-table lookup by attribute name string |
| Total instance size | Fixed, known at compile time | Fixed, known at compile time | Not fixed — depends on the dict's internal state |

### Worked Example 2.1.1 — The exact byte layout of `Point2D`

Construct `var point1 = Point2D(3.0, 4.0)` and trace what the compiler does, reusing Chapter 1.1's stack-frame reasoning directly. `Point2D` has two `Float64` fields, and Chapter 1.1 already established that a `Float64` occupies 8 bytes. The compiler lays the struct out as one contiguous 16-byte region:

```
Point2D instance for point1 = Point2D(3.0, 4.0):

 offset +0..+7   (field x, Float64):  bit pattern for 3.0
 offset +8..+15  (field y, Float64):  bit pattern for 4.0

 total size of one Point2D instance: 16 bytes
```

`point1.x` is not a search — it is "read 8 bytes starting at this instance's base address plus 0." `point1.y` is "read 8 bytes starting at this instance's base address plus 8." Both are a single fixed-offset memory read, decided once, at compile time, exactly like reading a plain `var y: Float64` from a stack frame in Chapter 1.1 — a struct field is not fundamentally different from a named stack slot, it's just a stack slot living *inside* another, larger stack slot.

Now trace the method call `point1.distance_from_origin()` with real numbers: `self.x * self.x = 3.0 × 3.0 = 9.0`, `self.y * self.y = 4.0 × 4.0 = 16.0`, sum `= 25.0`, and `pow(25.0, 0.5) = 5.0` — the classic 3-4-5 right triangle, so `point1`'s distance from the origin is exactly `5.0`.

### Worked Example 2.1.2 — Distance between two points

```mojo
var point1 = Point2D(3.0, 4.0)
var point2 = Point2D(1.0, 1.0)
var dist_between_points = point1.distance_to(point2)
```

`distance_to` computes `dx = self.x - other.x = 3.0 - 1.0 = 2.0` and `dy = self.y - other.y = 4.0 - 1.0 = 3.0`, then `pow(dx*dx + dy*dy, 0.5) = pow(4.0 + 9.0, 0.5) = pow(13.0, 0.5) ≈ 3.6056`. Check the shape of the computation, not just the arithmetic: both `point1` and `point2` are 16-byte `Point2D` instances with an identical field layout, which is exactly why `self.x - other.x` is meaningful at all — the compiler knows, at compile time, that `other.x` sits at the same relative offset within `other`'s 16 bytes that `self.x` sits within `self`'s.

### ASCII Diagram — two instances, identical layout, different bytes

```
point1 (Point2D):              point2 (Point2D):
 +0..+7:  3.0   (x)              +0..+7:  1.0   (x)
 +8..+15: 4.0   (y)              +8..+15: 1.0   (y)

Same blueprint (offsets +0 and +8 for x and y in both),
different bytes filled in at each offset -- this is what
"two instances of the same struct" means at the machine level.
```

### But what about Python?

Python has no built-in equivalent of a fixed-layout struct at all. There are two idioms Python programs reach for instead, and both give up the fixed-offset guarantee entirely.

**A `dict`:**
```python
point1 = {"x": 3.0, "y": 4.0}
dist = (point1["x"]**2 + point1["y"]**2) ** 0.5   # 5.0
```
`point1` here is a hash table. `point1["x"]` is not "read offset 0" — it is "compute a hash of the string `"x"`, use that hash to find a bucket, and look inside that bucket for a matching key," every single time the field is read. Two different `dict`s built from the same literal pattern aren't guaranteed to store their entries in the same internal slot order at all — there is no blueprint, only whatever bucket each key happened to land in.

**A `class`:**
```python
class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

point1 = Point2D(3.0, 4.0)
dist = (point1.x**2 + point1.y**2) ** 0.5   # 5.0
```
This *looks* like the Mojo struct above — same `__init__`, same dotted-attribute access — but under the hood, `point1.x = 3.0` in `__init__` doesn't write to a fixed offset; it inserts the key `"x"` into `point1.__dict__`, a hash table Python attaches to every instance automatically. `point1.x` in the distance calculation is, again, a hashed dictionary lookup by the string `"x"` — functionally the dict example above, wearing dot-syntax instead of bracket-syntax. The instance's total size isn't fixed at 16 bytes either; it depends on how the dictionary is currently sized internally, and can grow if more attributes are added later — something a Mojo or C struct's declaration explicitly forbids.

This is the practical cost of dynamic typing (Chapter 1.2) applied to compound data: every field access anywhere in a Python program pays for a hash and a lookup that a Mojo or C struct resolved once, permanently, at compile time.

> `[COMMON TRAP]` Because Mojo's `struct` syntax and Python's `class` syntax look almost identical — `__init__`, `self`, dotted access — it's easy to assume they compile down to the same thing with a different accent. They don't. A Mojo `struct` has no `__dict__`, no hash table, and no ability to gain a new field after it's declared; a Python `class` instance is a hash table wearing a name, and can grow new attributes at runtime that were never mentioned in `__init__` at all (`point1.z = 9.0` is perfectly legal Python, and would be a compile error in Mojo).

## 2.2 Multiple Constructors and Value Semantics `[FOUNDATIONAL]`

### Intuition

Ordering coffee at a counter, you can say "a medium latte" and get a fully specified drink, or just say "a coffee" and get whatever the shop's default is — same drink category, two different amounts of detail supplied, and the barista (not you) decides what to fill in when you under-specify. Multiple constructors work the same way: `Point2D(3.0, 4.0)` fully specifies the point, while `Point2D()` supplies nothing and lets the struct's own default apply.

### Background

**Constructor overloading** — writing `__init__` more than once with different parameter lists — is resolved entirely at compile time, by matching the arguments in a call against each overload's declared parameter types, the same mechanism C++ has used for function overloading since its earliest versions. Python cannot do this at all: a Python class has exactly one `__init__` method, and any "flexibility" in how it's called has to be faked *inside* that single method with default argument values or `*args`/`**kwargs`, checked with runtime `if` statements — there is no language-level dispatch on argument shape the way Mojo and C++ have.

```mojo
struct Point2D:
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        """Initialize a 2D point with x and y coordinates."""
        self.x = x
        self.y = y

    fn __init__(out self):
        """Default constructor - initialize to origin."""
        self.x = 0.0
        self.y = 0.0
```

| | Mojo / C++ overloaded `__init__`/constructor | Python single `__init__` |
|---|---|---|
| Number of `__init__` definitions | As many as needed, one per call shape | Exactly one |
| Which one runs | Chosen by the compiler, at compile time, from argument types | Always the same one — must branch internally |
| A call that matches no overload | Compile error | `TypeError` at runtime, or silently wrong if defaults mask it |

### Worked Example 2.2.1 — Two constructors, traced side by side

```mojo
var origin = Point2D()
var point1 = Point2D(3.0, 4.0)
```

`Point2D()` supplies zero arguments, so the compiler matches it against the no-argument `__init__` overload and runs `self.x = 0.0; self.y = 0.0` — `origin` ends up with the 16-byte layout from Section 2.1, with `0.0` at both offsets. `Point2D(3.0, 4.0)` supplies two `Float64` arguments, matching the two-argument overload instead, and produces the `[3.0, 4.0]` layout already traced in Worked Example 2.1.1. Both calls produce the same *type* of value — a 16-byte `Point2D` — through two entirely different, compile-time-selected code paths.

### Worked Example 2.2.2 — `distance_to`, computed and cross-checked

```mojo
var point2 = Point2D(1.0, 1.0)
var dist_origin_to_point2 = origin.distance_to(point2)
```

`origin` is `(0.0, 0.0)`, `point2` is `(1.0, 1.0)`. `dx = 0.0 - 1.0 = -1.0`, `dy = 0.0 - 1.0 = -1.0`. `distance_from_origin`'s formula and `distance_to`'s formula must agree here, since the distance from the origin to `point2` is by definition the same number either way: `pow((-1.0)² + (-1.0)², 0.5) = pow(2.0, 0.5) ≈ 1.41421` — and `point2.distance_from_origin()` computes `pow(1.0² + 1.0², 0.5)`, the identical `pow(2.0, 0.5) ≈ 1.41421`. Two different methods, two different call sites, one correct shared answer — a cheap but real consistency check on any new geometric method added to a struct like this one.

> `[COMMON TRAP]` Seeing `Point2D()` succeed with zero arguments can read as "Mojo structs have optional fields, like a Python `dataclass` with defaults on every field." They don't — `Point2D()` only works because a specific, separately-written `__init__(out self):` overload exists to handle exactly that call shape. Delete that overload, and `Point2D()` becomes a compile error, not a struct silently missing some fields; every overload is a complete, independent recipe for producing a fully-initialized instance, not a partial one.

## 2.3 Parametric Structs and Compile-Time Specialization `[FOUNDATIONAL]`

### Intuition

A cookie cutter mold that comes pre-manufactured in a fixed set of sizes — 4-inch, 6-inch, 8-inch — bakes each size as its own solid, dedicated piece of metal; there is no single mold that "adjusts" at baking time, each size is simply a separate object, made once. Compare that with a stretchy silicone cutter that's reshaped by hand for every batch — one physical object, reconfigured over and over, at the moment it's used.

A Mojo parametric struct like `Vector[dtype: DType, size: Int]` is the fixed-size mold: `Vector[DType.float32, 4]` and `Vector[DType.float64, 2]` are two entirely separate, independently compiled pieces of machine code, generated once at compile time — not one flexible piece of code that branches on `dtype` and `size` while the program runs.

### Background

Mojo's square-bracket parameters (`[dtype: DType, size: Int]`) are resolved at compile time, and every distinct combination of parameter values used anywhere in the program causes the compiler to generate its own fully specialized copy of the struct's code — this is called **monomorphization**, and it is exactly the mechanism behind C++ templates (`template<typename T, int N> struct Vector`). Python's answer to "generic" containers, `typing.Generic[T]`, is a purely cosmetic annotation read only by external static-analysis tools like `mypy`; at runtime, a Python class marked `Generic[T]` produces exactly the same single dict-based object regardless of what `T` was — there is no specialization, no separate compiled code path, and no `T` information left anywhere once the program is actually running.

| | Mojo `Vector[dtype, size]` | C++ `template<typename T, int N>` | Python `Generic[T]` |
|---|---|---|---|
| When is the parameter resolved? | Compile time | Compile time | Never — erased before runtime |
| Result of using two different parameter sets | Two separate compiled types | Two separate compiled types | One class, unchanged |
| Runtime cost of the parameterization itself | Zero | Zero | Zero benefit either — there was nothing to specialize |

### Worked Example 2.3.1 — Two specializations, traced independently

```mojo
alias Float32Vec4 = Vector[DType.float32, 4]
var vec1 = Float32Vec4(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))
var vec2 = Float32Vec4(SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))
```

`alias Float32Vec4 = Vector[DType.float32, 4]` is Section 1.3's compile-time substitution again: `Float32Vec4` is not a new type, it is a compile-time name for the *already-fully-specified* type `Vector[DType.float32, 4]`. Reusing Chapter 1.4's lane-wise reasoning for `vec_sum = vec1.add(vec2)`:

```
lane 0: 1.0 + 5.0 =  6.0
lane 1: 2.0 + 6.0 =  8.0
lane 2: 3.0 + 7.0 = 10.0
lane 3: 4.0 + 8.0 = 12.0
```

`vec_sum.data = [6.0, 8.0, 10.0, 12.0]`. `vec_product = vec1.multiply(vec2)` follows the same lane-wise pattern with multiplication: `1.0×5.0=5.0`, `2.0×6.0=12.0`, `3.0×7.0=21.0`, `4.0×8.0=32.0`, giving `[5.0, 12.0, 21.0, 32.0]`. `vec1.sum()` walks all four lanes and accumulates: `1.0 + 2.0 + 3.0 + 4.0 = 10.0`.

### Worked Example 2.3.2 — A second specialization, entirely separate code

```mojo
alias Float64Vec2 = Vector[DType.float64, 2]
var vec3 = Float64Vec2(SIMD[DType.float64, 2](1.5, 2.5))
```

`Vector[DType.float64, 2]` is a *different* instantiation from `Vector[DType.float32, 4]` above — different `dtype`, different `size` — so the compiler generates a second, independent version of every method in `Vector`, this one operating on two `Float64` lanes instead of four `Float32` lanes. `vec3.data` holds `[1.5, 2.5]`; nothing about `vec3`'s compiled code shares so much as an instruction with `vec1`'s compiled code, even though both came from the exact same `struct Vector[dtype: DType, size: Int]:` source text. This is the concrete meaning of "zero-cost abstraction" invoked in Chapter 1.4: the generic-looking source produces exactly the specialized machine code a programmer would have hand-written separately for `float32×4` and `float64×2`, with the parameterization itself costing nothing at runtime.

### ASCII Diagram — one source, two compiled types

```
struct Vector[dtype: DType, size: Int]:   <- one generic source definition
   ...

           compiled separately, once each, at compile time
                    /                              \
                   v                                v
 Vector[DType.float32, 4]              Vector[DType.float64, 2]
 - 4 lanes of Float32 (16 bytes)       - 2 lanes of Float64 (16 bytes)
 - its own .add(), .multiply(), ...    - its own .add(), .multiply(), ...
 - used by vec1, vec2                  - used by vec3

No shared machine code between the two boxes above, despite
sharing one struct definition in the source file.
```

## 2.4 RAII: Constructors, Destructors, and Manual Memory in a Struct `[FOUNDATIONAL]`

### Intuition

Chapter 1.5 compared heap memory to a hotel room you must remember to check out of yourself, and pointed out that forgetting the checkout leaves the room permanently, uselessly occupied. Now imagine a smarter hotel room: one wired so that the moment your reserved stay (the scope you booked it for) ends, the door automatically unlocks, the room reports itself vacant, and housekeeping is notified — with no action required from the guest at all. You still had to explicitly check *in* (the constructor still runs), but checking *out* is no longer something you can forget, because it's tied directly to the reservation ending rather than to your memory.

### Background

**RAII** (Resource Acquisition Is Initialization) is the pattern of tying a resource's cleanup to an object's own lifetime, rather than requiring a separate, easy-to-forget cleanup call. A Mojo struct's `__init__` can allocate heap memory the same way Chapter 1.5's standalone code did, and its `__del__` — called automatically when the instance's lifetime ends — frees that memory, closing the loop that Chapter 1.5 left the programmer to close by hand. This is the same pattern C++ constructors/destructors have always provided; raw C has neither piece automated at all, requiring every `malloc` to be paired with a `free` purely by programmer discipline. Python doesn't need this pattern for the memory case, because every Python object is already heap-allocated and reference-counted automatically (Chapter 1.5's Background table) — Python does have a `__del__` method, but it fires only when the reference count happens to reach zero, at a time the garbage collector chooses, not deterministically at the end of a particular scope the way Mojo's and C++'s destructors do.

```mojo
struct DynamicArray[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var capacity: Int

    fn __init__(out self, initial_capacity: Int):
        self.capacity = initial_capacity
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]].alloc(initial_capacity)   # check-in

    fn __del__(owned self):
        self.data.free()   # automatic check-out, tied to self's lifetime
```

| | Raw C (`malloc`/`free`, Ch. 1.5) | Mojo struct `__init__`/`__del__` | Python object refcounting |
|---|---|---|---|
| Who calls the cleanup | The programmer, by hand, every time | The compiler, automatically, at scope end | The interpreter, automatically, when refcount hits 0 |
| Forgetting it | Silent memory leak | Not possible to forget — it's automatic | Not applicable — always automatic |
| Timing of cleanup | Whenever the programmer wrote `free()` | Deterministic — exactly at scope end | Non-deterministic in general (GC-dependent) |

### Worked Example 2.4.1 — Tracing a resize by hand

```mojo
var int_array = DynamicArray[DType.int32](4)
for i in range(10):
    int_array.append(i * i)
```

`DType.int32` is 4 bytes per element (half of the 8-byte `Int` from Chapter 1.1, since `int32` is explicitly 32-bit). `DynamicArray[DType.int32](4)` allocates `4 × 4 = 16` bytes for the initial capacity of 4. Trace the ten `append` calls against `_resize`'s trigger condition, `self.size >= self.capacity`:

```
i=0: append(0)   size 0->1   (0 >= 4? no)   capacity stays 4
i=1: append(1)   size 1->2   (1 >= 4? no)   capacity stays 4
i=2: append(4)   size 2->3   (2 >= 4? no)   capacity stays 4
i=3: append(9)   size 3->4   (3 >= 4? no)   capacity stays 4
i=4: append(16)  (4 >= 4? YES) -> resize: new_capacity = 4*2 = 8
                 copy 4 existing elements, free the old 16-byte block,
                 allocate a new 32-byte (8*4) block
                 size 4->5
i=5: append(25)  size 5->6   (5 >= 8? no)
i=6: append(36)  size 6->7   (6 >= 8? no)
i=7: append(49)  size 7->8   (7 >= 8? no)
i=8: append(64)  (8 >= 8? YES) -> resize: new_capacity = 8*2 = 16
                 copy 8 existing elements, free the old 32-byte block,
                 allocate a new 64-byte (16*4) block
                 size 8->9
i=9: append(81)  size 9->10  (9 >= 16? no)
```

After all ten appends: `size = 10`, `capacity = 16`, and the array holds `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]` — each entry is `i*i` for `i` from 0 to 9, exactly as the loop specifies. Two resizes happened (capacity `4→8→16`), each one freeing its old block only *after* copying every existing element into the new, larger one — free the old 16-byte block only once its 4 values are safely sitting in the new 32-byte block, otherwise the copy would be reading already-freed memory (Chapter 1.5's use-after-free trap, avoided here by ordering).

### ASCII Diagram — capacity growth over the ten appends

```
size:      0  1  2  3  4  5  6  7  8  9  10
capacity:  4  4  4  4  8  8  8  8 16 16  16
                       ^              ^
                  resize here    resize here
                 (4 elems copied) (8 elems copied)
                 old 16B freed    old 32B freed
                 new 32B alloc'd  new 64B alloc'd
```

### Worked Example 2.4.2 — When `__del__` fires

```mojo
fn demo():
    var int_array = DynamicArray[DType.int32](4)
    int_array.append(9)
    # ... int_array used here ...
# <- int_array's scope ends exactly here: __del__ runs automatically,
#    calling self.data.free() on whatever block int_array currently owns
```

No line in `demo()` calls `.free()` directly, and none needs to — `int_array` is a local variable, so the compiler places a call to `DynamicArray.__del__` at every point `int_array`'s scope can end, including a normal fall-through *and* an early `return`, guaranteeing the underlying heap block (whatever size it grew to through however many resizes) is freed exactly once. This is the direct fix for the exact risk Section 1.5 flagged by hand with `heap_ptr` — wrapping the raw pointer in a struct with a destructor removes the "did I remember to call `.free()`" question entirely, for every code path, not just the ones a programmer happened to test.

> `[COMMON TRAP]` A struct that owns an `UnsafePointer` and *doesn't* define `__del__` leaks exactly as surely as forgetting `.free()` did in Chapter 1.5 — RAII isn't automatic because the field is a pointer, it's automatic because a destructor was written to act on that pointer. It is easy, coming from Python, to assume any struct's memory is "handled" simply because Mojo has an ownership system; ownership tracks *who is allowed to use* a value and *when*, but it does not, by itself, know to call `.free()` on a raw pointer field unless a `__del__` method says so.

## 2.5 Traits: Interfaces Without Inheritance Hierarchies `[FOUNDATIONAL]`

### Intuition

A job posting can list required capabilities — "must be able to drive, must be able to lift 50 lbs" — without requiring the applicant to hold any particular job title or have come from any particular company. Anyone who can actually do those two things qualifies, whether their background is a delivery driver, a furniture mover, or a warehouse worker; the posting doesn't care about their history, only about what they can currently do. A `trait` is that job posting, applied to a type: it lists required method signatures, and any struct that implements them qualifies — regardless of what other traits or structs it's related to.

### Background

A Mojo `trait` declares a set of method signatures a conforming struct must implement; conformance is declared explicitly (`struct Rectangle(Drawable):`) and checked entirely at compile time, with no runtime "is this really a `Drawable`" check anywhere. This is close in spirit to a C++ abstract base class with pure virtual methods — but a C++ virtual call is resolved through a **vtable**, a table of function pointers looked up at runtime, adding one indirection per call. A generic Mojo function like `draw_shape[T: Drawable](shape: T)` is, like `Vector` in Section 2.3, monomorphized: called with a `Rectangle`, the compiler generates a `draw_shape` specialized for `Rectangle` with a direct call to `Rectangle.draw`; called with a `Circle`, it generates a second, separate specialization calling `Circle.draw` directly — no vtable, no runtime indirection, in either case. Python's version of "does this object support `.draw()` and `.area()`" is **duck typing**: nothing is declared or checked in advance at all, and a call to `shape.draw()` either works or raises an `AttributeError`, discovered only at the moment the call is actually made, at runtime, however deep in a program's execution that happens to be.

| | Mojo `trait` | C++ abstract base class | Python duck typing |
|---|---|---|---|
| Conformance checked | Compile time | Compile time (must inherit) | Runtime, at the call site, via `AttributeError` |
| Call mechanism | Direct call (monomorphized per type) | Vtable lookup (runtime indirection) | Attribute lookup, then call |
| Must declare an inheritance relationship | Yes, explicitly (`(Drawable)`) | Yes, explicitly (`: public Drawable`) | No relationship exists at all |

### Worked Example 2.5.1 — Two shapes, two areas, checked by hand

```mojo
var rect = Rectangle(5.0, 3.0)
var circle = Circle(2.5)
```

`Rectangle.area()` computes `width × height = 5.0 × 3.0 = 15.0`. `Circle.area()` computes `PI × radius² = 3.14159265359 × 2.5² = 3.14159265359 × 6.25 ≈ 19.6349540849`. Both `area()` implementations satisfy the same `Drawable` trait requirement (`fn area(self) -> Float64`), despite computing completely different formulas internally — the trait only constrains the *signature*, never the *implementation*.

### Worked Example 2.5.2 — Generic dispatch, resolved at compile time

```mojo
fn draw_shape[T: Drawable](shape: T):
    shape.draw()
    print("Area: " + String(shape.area()))

draw_shape(rect)     # T is inferred as Rectangle at this call site
draw_shape(circle)   # T is inferred as Circle at this call site
```

At the first call, the compiler infers `T = Rectangle` and generates a specialization of `draw_shape` whose body is, concretely, `rect.draw()` (printing `"Drawing rectangle: 5.0 x 3.0"`) followed by printing `"Area: 15.0"` — the value traced by hand above. At the second call, it infers `T = Circle` and generates a *second*, independent specialization calling `circle.draw()` and printing `"Area: 19.6349540849..."`. Two calls to the same generic source, two separately compiled, direct-call function bodies — the same monomorphization Section 2.3 traced for `Vector[dtype, size]`, now applied to a trait-bounded generic instead of a data-parameter generic.

### ASCII Diagram — one generic function, two direct-call specializations

```
fn draw_shape[T: Drawable](shape: T):     <- one generic source
    shape.draw()
    print("Area:", shape.area())

        compiled separately for each T actually used
              /                          \
             v                            v
 draw_shape[Rectangle]           draw_shape[Circle]
 - calls Rectangle.draw()        - calls Circle.draw()   directly,
   directly, no vtable             no vtable
 - calls Rectangle.area()        - calls Circle.area()
   directly                        directly
```

> `[COMMON TRAP]` `struct Rectangle(Drawable):` reads like C++'s `class Rectangle : public Drawable`, which invites the assumption that `Drawable` behaves like a base class — that a `Rectangle` "is a" `Drawable` in some stored, runtime-visible sense, the way a C++ object carries a vtable pointer identifying its dynamic type. It doesn't: trait conformance in Mojo (like in Rust, its closest relative here) is a compile-time-only fact used to decide which specialization of a generic function to generate. There is no runtime tag on a `Rectangle` instance saying "I am `Drawable`," and no runtime cost paid anywhere for the conformance — which is also exactly why a struct can be given a *new* trait conformance later, from outside its original declaration, without modifying or recompiling the struct's own definition, something a fixed C++ inheritance hierarchy cannot do after the fact.

## 2.6 Complete Runnable Code

The five files below are the chapter's full, standalone source — each one adds exactly one more pattern from Sections 2.1–2.5 on top of the last, so any file can be compiled and run on its own.

### File: `06_basic_structs.mojo` — Sections 2.1 and 2.2

**Execution:** `pixi run mojo 06_basic_structs.mojo`

```mojo
from memory import UnsafePointer

struct Point2D:
    """Basic struct representing a 2D point."""
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        """Initialize a 2D point with x and y coordinates."""
        self.x = x
        self.y = y

    fn __init__(out self):
        """Default constructor - initialize to origin."""
        self.x = 0.0
        self.y = 0.0

    fn distance_from_origin(self) -> Float64:
        """Calculate distance from origin using Pythagorean theorem."""
        return pow(self.x * self.x + self.y * self.y, 0.5)

    fn distance_to(self, other: Point2D) -> Float64:
        """Calculate distance to another point."""
        var dx = self.x - other.x
        var dy = self.y - other.y
        return pow(dx * dx + dy * dy, 0.5)

    fn __str__(self) -> String:
        """String representation of the point."""
        return "Point2D(" + String(self.x) + ", " + String(self.y) + ")"

fn basic_struct_demo():
    """Demonstrate basic struct usage patterns."""
    print("=== Basic Struct Patterns ===")

    var origin = Point2D()
    var point1 = Point2D(3.0, 4.0)
    var point2 = Point2D(1.0, 1.0)

    print("Origin:", origin.__str__())
    print("Point1:", point1.__str__())
    print("Point2:", point2.__str__())

    var dist_from_origin = point1.distance_from_origin()
    var dist_between_points = point1.distance_to(point2)

    print("Point1 distance from origin:", dist_from_origin)
    print("Distance between point1 and point2:", dist_between_points)

fn main():
    """Main function for basic struct demonstration."""
    basic_struct_demo()
```

### File: `07_parametric_structs.mojo` — Section 2.3

**Execution:** `pixi run mojo 07_parametric_structs.mojo`

```mojo
from memory import UnsafePointer

struct Vector[dtype: DType, size: Int]:
    """Parametric struct for SIMD-optimized vectors."""
    var data: SIMD[dtype, size]

    fn __init__(out self):
        """Initialize vector with zeros."""
        self.data = SIMD[dtype, size](0)

    fn __init__(out self, value: Scalar[dtype]):
        """Initialize vector with single value (broadcast)."""
        self.data = SIMD[dtype, size](value)

    fn __init__(out self, simd_data: SIMD[dtype, size]):
        """Initialize vector with SIMD data."""
        self.data = simd_data

    fn get(self, index: Int) -> Scalar[dtype]:
        """Get element at index."""
        return self.data[index]

    fn set(mut self, index: Int, value: Scalar[dtype]):
        """Set element at index."""
        self.data[index] = value

    fn add(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        """Add two vectors element-wise."""
        return Vector[dtype, size](self.data + other.data)

    fn multiply(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        """Multiply two vectors element-wise."""
        return Vector[dtype, size](self.data * other.data)

    fn sum(self) -> Scalar[dtype]:
        """Sum all elements in the vector."""
        var result: Scalar[dtype] = 0
        for i in range(size):
            result += self.data[i]
        return result

    fn print_vector(self, name: String):
        """Print vector elements without string concatenation."""
        print(name + ":", self.data)

fn parametric_struct_demo():
    """Demonstrate parametric struct usage."""
    print("=== Parametric Struct Patterns ===")

    alias Float32Vec4 = Vector[DType.float32, 4]
    alias Float64Vec2 = Vector[DType.float64, 2]

    var vec1 = Float32Vec4(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))
    var vec2 = Float32Vec4(SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))

    vec1.print_vector("Vector 1")
    vec2.print_vector("Vector 2")

    var vec_sum = vec1.add(vec2)
    var vec_product = vec1.multiply(vec2)

    vec_sum.print_vector("Sum")
    vec_product.print_vector("Product")
    print("Sum of all elements in vec1:", vec1.sum())

    var vec3 = Float64Vec2(SIMD[DType.float64, 2](1.5, 2.5))
    vec3.print_vector("Float64 Vector")

    var zero_vec = Float32Vec4()
    zero_vec.print_vector("Zero Vector")

    var broadcast_value: Scalar[DType.float32] = 3.14
    var broadcast_vec = Float32Vec4(broadcast_value)
    broadcast_vec.print_vector("Broadcast Vector")

fn main():
    """Main function for parametric struct demonstration."""
    parametric_struct_demo()
```

### File: `08_memory_management_structs.mojo` — Section 2.4

**Execution:** `pixi run mojo 08_memory_management_structs.mojo`

```mojo
from memory import UnsafePointer

struct DynamicArray[dtype: DType]:
    """Dynamic array with manual memory management."""
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var capacity: Int

    fn __init__(out self, initial_capacity: Int):
        """Initialize with given capacity."""
        self.capacity = initial_capacity
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]].alloc(initial_capacity)

    fn __init__(out self):
        """Initialize with default capacity."""
        self.capacity = 8
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]].alloc(8)

    fn __del__(owned self):
        """Destructor - free allocated memory."""
        self.data.free()

    fn append(mut self, value: Scalar[dtype]):
        """Add element to end of array."""
        if self.size >= self.capacity:
            self._resize()

        self.data[self.size] = value
        self.size += 1

    fn get(self, index: Int) -> Scalar[dtype]:
        """Get element at index."""
        return self.data[index]

    fn set(mut self, index: Int, value: Scalar[dtype]):
        """Set element at index."""
        self.data[index] = value

    fn _resize(mut self):
        """Private method to resize array when capacity is exceeded."""
        var new_capacity = self.capacity * 2
        var new_data = UnsafePointer[Scalar[dtype]].alloc(new_capacity)

        for i in range(self.size):
            new_data[i] = self.data[i]

        self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    fn print_elements(self):
        """Print all elements in the array."""
        print("DynamicArray[", self.size, "/", self.capacity, "]: ", end="")
        for i in range(self.size):
            print(self.data[i], end="")
            if i < self.size - 1:
                print(", ", end="")
        print("")

fn memory_management_demo():
    """Demonstrate memory management in structs."""
    print("=== Memory Management in Structs ===")

    var int_array = DynamicArray[DType.int32](4)
    var float_array = DynamicArray[DType.float32]()

    print("Adding elements to int array:")
    for i in range(10):
        int_array.append(i * i)
        int_array.print_elements()

    print("\nAdding elements to float array:")
    for i in range(5):
        var value = Float32(i) * 1.5
        float_array.append(value)
        float_array.print_elements()

    print("\nAccessing elements:")
    print("int_array[3] =", int_array.get(3))
    print("float_array[2] =", float_array.get(2))

fn main():
    """Main function for memory management demonstration."""
    memory_management_demo()
```

### File: `09_trait_structs.mojo` — Section 2.5

**Execution:** `pixi run mojo 09_trait_structs.mojo`

```mojo
from memory import UnsafePointer

trait Drawable:
    """Trait for objects that can be drawn."""
    fn draw(self):
        """Draw the object."""
        pass

    fn area(self) -> Float64:
        """Calculate area of the object."""
        pass

struct Rectangle(Drawable):
    """Rectangle that implements Drawable trait."""
    var width: Float64
    var height: Float64

    fn __init__(out self, width: Float64, height: Float64):
        self.width = width
        self.height = height

    fn draw(self):
        """Draw the rectangle."""
        print("Drawing rectangle: " + String(self.width) + " x " + String(self.height))

    fn area(self) -> Float64:
        """Calculate rectangle area."""
        return self.width * self.height

struct Circle(Drawable):
    """Circle that implements Drawable trait."""
    var radius: Float64

    fn __init__(out self, radius: Float64):
        self.radius = radius

    fn draw(self):
        """Draw the circle."""
        print("Drawing circle with radius: " + String(self.radius))

    fn area(self) -> Float64:
        """Calculate circle area."""
        alias PI = 3.14159265359
        return PI * self.radius * self.radius

fn draw_shape[T: Drawable](shape: T):
    """Generic function that works with any Drawable."""
    shape.draw()
    print("Area: " + String(shape.area()))

fn trait_demo():
    """Demonstrate trait usage with structs."""
    print("=== Trait-based Struct Patterns ===")

    var rect = Rectangle(5.0, 3.0)
    var circle = Circle(2.5)

    print("Rectangle:")
    draw_shape(rect)

    print("\nCircle:")
    draw_shape(circle)

    print("\nDirect method calls:")
    rect.draw()
    circle.draw()

fn main():
    """Main function for trait demonstration."""
    trait_demo()
```

### File: `10_struct_patterns_complete.mojo` — every pattern in one program

**Execution:** `pixi run mojo 10_struct_patterns_complete.mojo`

```mojo
from memory import UnsafePointer

# Basic struct pattern
struct Point2D:
    """Basic struct representing a 2D point."""
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0

    fn distance_from_origin(self) -> Float64:
        return pow(self.x * self.x + self.y * self.y, 0.5)

# Parametric struct pattern
struct Vector[dtype: DType, size: Int]:
    """Parametric struct for SIMD-optimized vectors."""
    var data: SIMD[dtype, size]

    fn __init__(out self, simd_data: SIMD[dtype, size]):
        self.data = simd_data

    fn add(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        return Vector[dtype, size](self.data + other.data)

# Memory management pattern
struct SimpleArray[dtype: DType]:
    """Simple array with manual memory management."""
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
        self.data = UnsafePointer[Scalar[dtype]].alloc(size)

    fn __del__(owned self):
        self.data.free()

    fn get(self, index: Int) -> Scalar[dtype]:
        return self.data[index]

    fn set(mut self, index: Int, value: Scalar[dtype]):
        self.data[index] = value

# Trait pattern
trait Printable:
    fn print_info(self):
        pass

struct Number(Printable):
    var value: Float64

    fn __init__(out self, value: Float64):
        self.value = value

    fn print_info(self):
        var value_str: String = String(self.value)
        var message: String = "Number value: " + value_str
        print(message)

fn complete_struct_demo():
    """Demonstrate all struct patterns together."""
    print("=== Complete Struct Patterns Demo ===")

    var point = Point2D(3.0, 4.0)
    print("Point distance from origin:", point.distance_from_origin())

    alias FloatVec4 = Vector[DType.float32, 4]
    var vec1 = FloatVec4(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))
    var vec2 = FloatVec4(SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))
    var vec_sum = vec1.add(vec2)
    print("Vector addition result: ", vec_sum.data)

    var array = SimpleArray[DType.float64](3)
    array.set(0, 1.1)
    array.set(1, 2.2)
    array.set(2, 3.3)
    print("Array elements:", array.get(0), array.get(1), array.get(2))

    var number = Number(42.5)
    number.print_info()

fn main():
    """Complete demonstration of all struct patterns."""
    complete_struct_demo()
```

### Expected Output for `10_struct_patterns_complete.mojo`

```
=== Complete Struct Patterns Demo ===
Point distance from origin: 5.0
Vector addition result: [6.0, 8.0, 10.0, 12.0]
Array elements: 1.1 2.2 3.3
Number value: 42.5
```

Every number here was hand-traced earlier in the chapter: `5.0` is the 3-4-5 triangle from Worked Example 2.1.1, `[6.0, 8.0, 10.0, 12.0]` is the lane-wise vector sum from Worked Example 2.3.1, and `Number value: 42.5` exercises the same trait-conformance mechanism traced in Section 2.5, just with a single-method `Printable` trait instead of `Drawable`.

## Chapter Summary

A `struct` is a fixed, contiguous, compile-time-known memory layout — the direct descendant of C's `struct`, with methods attached and Mojo's ownership rules governing how instances move around, but sharing C's core promise: every field sits at a fixed offset, decided once, for every instance that will ever exist. Python has no equivalent primitive; a Python `class` instance is a per-object hash table (`__dict__`) wearing dot-syntax, and every field access anywhere is a runtime hash-and-lookup rather than a fixed-offset read — the price of the dynamic typing Chapter 1.2 already introduced, now paid on every attribute of every object rather than just on every variable. Constructor overloading lets one struct offer several build recipes, chosen by the compiler from the shapes of the arguments at each call site, something Python's single `__init__` cannot do without hand-written runtime branching. Parametric structs like `Vector[dtype, size]` extend Chapter 1.4's SIMD story: each distinct combination of compile-time parameters becomes its own separately compiled type, exactly like a C++ template instantiation, with zero runtime cost for the genericity — unlike Python's `Generic[T]`, which is erased entirely before the program runs and specializes nothing. `__init__`/`__del__` pairs give a struct RAII, tying Chapter 1.5's manual `alloc`/`free` discipline to the struct's own lifetime so it can no longer be forgotten, the way C++ constructors and destructors have always worked and raw C never did. And a `trait` defines required method signatures that any struct can satisfy, checked at compile time and dispatched through direct, monomorphized calls with no vtable — a stronger guarantee than Python's duck typing, which discovers a missing method only when the program actually crashes trying to call it.

## Self-Check Questions

1. `Point2D` and a Python class with the same two fields both support `point.x` and `point.y`. Explain precisely what happens differently, at the machine level, when each of those two field accesses executes.
2. Why can Mojo choose which `__init__` overload to run for `Point2D()` vs `Point2D(3.0, 4.0)` entirely at compile time, while a Python class handling both call shapes must do so with a runtime `if` inside a single `__init__`?
3. `Vector[DType.float32, 4]` and `Vector[DType.float64, 2]` come from the same `struct Vector[dtype: DType, size: Int]:` source. In what concrete sense are they "two different types" rather than one flexible type?
4. A struct holds an `UnsafePointer` field, has a working `__init__` that allocates it, but no `__del__` at all. What happens when an instance of that struct goes out of scope, and why doesn't Mojo's ownership system catch this automatically?
5. `struct Rectangle(Drawable):` and C++'s `class Rectangle : public Drawable` look structurally similar. Name one concrete runtime difference between how a call to `shape.area()` executes in each.

## Where We Go Next

This chapter's structs — `Point2D`, `Vector[dtype, size]`, `DynamicArray[dtype]` — were built from scratch expressly to demonstrate each pattern in isolation. Chapter 3 puts the same tools to their real job in this book: designing the memory layout *inside* a single struct — Array-of-Structs versus Struct-of-Arrays — for the `Tensor` type every remaining chapter is built around, where the layout choice determines whether the GPU can read a whole batch of values in one coalesced memory transaction or in many slow, scattered ones.

## Worked Solutions

**1.** `point.x` on the Mojo `Point2D` is a single fixed-offset memory read: the compiler already knows, from the struct's declaration, that field `x` lives at byte offset 0 relative to the instance's base address, so the generated machine code is "read 8 bytes at base+0" with no search involved. `point.x` on the Python class is a runtime dictionary lookup: it hashes the string `"x"`, uses that hash to locate a bucket in `point.__dict__`, and reads the value stored there — work repeated on every single access, because Python never fixed `x`'s position the way the struct declaration did.

**2.** Overload resolution only needs to know the *type and number* of arguments at a call site, and in Mojo (as in C++) that information is fully available at compile time — the compiler can see, textually, that `Point2D()` supplies zero arguments and `Point2D(3.0, 4.0)` supplies two `Float64`s, and match each against the correspondingly-shaped `__init__`. Python cannot do this because it has no separate compile-time type-checking pass matching call shapes to multiple candidate functions; there is exactly one `__init__` per class, so any shape-dependent behavior has to be re-derived every time that one function actually runs, using ordinary runtime conditionals on whatever arguments happened to arrive.

**3.** They are two different types in the concrete sense that the compiler generates two entirely separate bodies of machine code for them — separate storage layout (`SIMD[float32,4]` is 16 bytes across 4 lanes; `SIMD[float64,2]` is also 16 bytes but across 2 lanes of double the width each), and separate compiled versions of every method (`add`, `multiply`, `sum`, and so on), with no shared instructions between the two despite both being generated from one `struct Vector[dtype: DType, size: Int]:` source text. This is monomorphization: the generic source is a template for generating types, not a single flexible type that branches internally.

**4.** When that instance's scope ends, nothing frees the `UnsafePointer` field's heap memory — it leaks, exactly as it would have leaked in Chapter 1.5's standalone example if `.free()` were never called. Mojo's ownership system tracks *who currently owns a value and when its lifetime ends* so it knows precisely when to invoke a `__del__`, if one exists — but it has no way to infer that a raw pointer field represents a resource that needs releasing; that knowledge only exists if a programmer writes it into a `__del__` method. Ownership tracking and automatic cleanup are related but separate: one decides *when* cleanup should happen, the other decides *what* cleanup means, and only a written `__del__` supplies the second part.

**5.** In Mojo, calling `shape.area()` inside a generic function like `draw_shape[T: Drawable]` is resolved entirely at compile time: for each concrete `T` the function is called with, the compiler generates a specialized version of `draw_shape` containing a direct, non-indirect call to that specific type's `area()` — there is no runtime lookup involved at all. In C++, calling `area()` through a `Drawable*` or `Drawable&` base-class reference (the idiomatic way to get the equivalent polymorphism) goes through a **vtable**: a per-object pointer to a table of function pointers, consulted at runtime to find the correct override — one extra memory indirection on every single virtual call that Mojo's compile-time-monomorphized trait dispatch never pays.
