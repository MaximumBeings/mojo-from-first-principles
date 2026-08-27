# Chapter 1: Variables, Types, and the Machine Underneath

> "A type is not a restriction the compiler imposes on you. It is a promise you make to the compiler about what a piece of memory means — and the whole reason compiled languages are fast is that the compiler is allowed to trust that promise completely, instead of checking it at every single use the way a dynamic language must."

**What you will understand by the end of this chapter:**

- What actually happens in memory when you write `var x: Int = 42` — not just "it creates a variable," but the stack slot, the byte width, and why the compiler needs to know the type *before* that line runs
- Why Mojo's type inference (`var a = 10`) is not the same thing as Python's dynamic typing, even though neither one has a visible type annotation
- The difference between a runtime variable (`var`) and a compile-time constant (`alias`), and why that distinction lets Mojo eliminate work that Python and even C must do at runtime
- Why `SIMD[DType.float32, 4]` is a genuine language-level numeric type in Mojo — not a library wrapper the way a NumPy array is in Python, or a set of manually-invoked CPU intrinsics the way vectorization is in C
- How heap memory obtained through `UnsafePointer` differs from both Python's automatically-managed objects and C's `malloc`/`free`, and why forgetting the Mojo equivalent of `free()` produces the exact same leak it would in C

**What you need to know first:**

- General programming experience in *any* language (the C/C++ and Python comparisons in this chapter assume only familiarity with basic syntax, not expertise)
- No prior Mojo exposure is assumed — this is the first technical chapter of the book

## 1.1 What a Type Actually Is `[FOUNDATIONAL]`

### Intuition

Think about shipping a package through a courier service. Before the courier will take it, you declare what's inside and how much it weighs — not because the courier is nosy, but because that declaration determines everything downstream: which truck it goes on, how it's stacked, whether it needs a fragile sticker, how much space to reserve. A package labeled "2 kg, books" gets handled completely differently from one labeled "40 kg, machinery," and the courier never has to *open the box* to find out which — the label already told them.

A type is that label, attached not to a package but to a region of computer memory. `Int` means "this is 8 bytes, interpret those bytes as a signed whole number." `Float32` means "this is 4 bytes, interpret them according to the IEEE-754 rules for a 32-bit floating point number." Once the compiler knows the label, it never has to inspect the memory itself to know how to handle it — every future operation on that variable already knows exactly how many bytes to read and how to interpret them.

### Background

A **statically typed** language requires the type of every variable to be fixed and known *before the program runs* — either because the programmer wrote it explicitly or because the compiler deduced it unambiguously from context. C, C++, and Mojo are all statically typed. A **dynamically typed** language, like Python, attaches the type to the *value* at runtime instead of to the variable: the name `x` doesn't have a type at all, it just currently points at some object, and that object carries its own type tag that Python checks on every single operation.

This distinction is the entire reason statically typed languages can be fast. When the Mojo compiler sees `var x: Int = 42`, it reserves exactly 8 bytes on the stack, and every later line that uses `x` is compiled into machine code that reads exactly those 8 bytes and treats them as an integer — no check, no branch, no lookup. When the Python interpreter sees `x = 42`, it allocates a full Python `int` object on the heap (which, because Python integers are arbitrary-precision, is considerably more than 8 bytes) and every later line that uses `x` must first read a type tag off that object at runtime to decide what "add" or "print" even means for it.

| Language | Type known at | Where the value lives | Type check per use |
|---|---|---|---|
| C / C++ | Compile time (mandatory) | Stack (usually) | None |
| Mojo | Compile time (explicit or inferred) | Stack (usually) | None |
| Python | Runtime (attached to the object) | Heap (always) | Every operation |

### Worked Example 1.1.1 — Three languages, one declaration

Take the same intention — "store the number 42 as an integer" — across three languages, and trace what each one actually does.

**C:**
```c
int x = 42;
```
The compiler reserves 4 bytes (a C `int` is typically 32-bit) at a fixed offset in the current stack frame, say offset `+16`. It writes the bit pattern for `42` there. No object, no header, no tag — just 4 raw bytes sitting between the other local variables.

**Mojo:**
```mojo
var x: Int = 42
```
The compiler reserves 8 bytes (Mojo's `Int` is 64-bit, matching the machine's native word size) at some stack offset, and writes the bit pattern for `42`. Functionally identical to the C case, just a wider integer type.

**Python:**
```python
x = 42
```
The interpreter allocates a heap object — conceptually, something like a small `struct` with a type-tag field pointing at the `int` type, a reference count field, and a field (or fields, for large numbers) holding the digits of `42`. `x` itself is not the number; `x` is a pointer to that heap object, stored in the enclosing scope's variable dictionary. Every later `x + 1` first dereferences the pointer, reads the type tag, confirms it's an `int`, *then* does the addition.

Same intention, three completely different amounts of machine work — and the C and Mojo columns are identical in kind, because both are statically typed. This is why the rest of this book routinely compares Mojo against C/C++ for *mechanism* and against Python for *ergonomics*: Mojo's performance model matches C's, and its syntax is written to feel closer to Python's.

### ASCII Diagram — one stack frame, three variables

```
Stack frame for a function containing:
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828

High address
 +--------------------------+
 | ... caller's frame ...   |
 +--------------------------+
 | z  (8 bytes, Float64)    |  <- offset +16..+23:  01000000000101...  (IEEE-754 bits for 2.71828)
 +--------------------------+
 | y  (4 bytes, Float32)    |  <- offset +8..+11:   01000000010010...  (IEEE-754 bits for 3.14159)
 +--------------------------+
 | x  (8 bytes, Int)        |  <- offset +0..+7:    0000000000101010   (two's-complement bits for 42)
 +--------------------------+
Low address                    <- stack pointer sits here
```

Nothing in this picture says "this is an Int" or "this is a Float32" — that information exists only in the compiler's own bookkeeping (its *symbol table*), which it uses once, at compile time, to decide how many bytes to reserve at each offset and which machine instructions to emit for each operation. By the time the program is actually running, the type has already done its job and vanished; only raw bytes remain.

### Worked Example 1.1.2 — Built-in operations are just typed dispatch

Mojo's built-in math functions look like ordinary function calls, but each one is really a compile-time decision about which machine instruction to emit, guided entirely by the argument's type:

```mojo
var sqrt_result = pow(4.0, 0.5)   # 4.0 is inferred as Float64 -> emits a floating-point pow instruction
var abs_result = abs(-42)         # -42 is Int -> emits an integer absolute-value sequence
var max_result = max(10, 20)      # both Int -> emits an integer compare-and-select
```

Trace `pow(4.0, 0.5)` by hand to confirm the answer the compiler's chosen instruction should produce: `4.0^0.5` is the square root of `4.0`, and `2.0 × 2.0 = 4.0`, so the result is `2.0`. `abs(-42)` flips the sign bit's effect: `-(-42) = 42`. `max(10, 20)` picks the larger of the two: `20`. These are trivial by hand precisely *because* the types were fixed before any of this ran — the compiler already knew "these are two `Int`s, emit an integer comparison" without checking anything at the moment `max` is called.

```mojo
fn basic_variables_demo():
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828
    print("Integer x:", x)
    print("Float32 y:", y)
    print("Float64 z:", z)

    alias PI = 3.14159265359
    alias MAX_SIZE = 1024

    var sqrt_result = pow(4.0, 0.5)
    var abs_result = abs(-42)
    var max_result = max(10, 20)
    var min_result = min(10, 20)
    print("sqrt(4):", sqrt_result, " abs(-42):", abs_result)
    print("max(10,20):", max_result, " min(10,20):", min_result)

fn main():
    basic_variables_demo()
```

> `[COMMON TRAP]` It's tempting to read `Float32` and `Float64` as "small float" and "big float" and assume Mojo will silently convert between them wherever convenient, the way Python happily promotes `int` to `float`. Mojo does not: mixing a `Float32` and a `Float64` in one expression is a type mismatch the compiler will reject unless you convert explicitly. The width is part of the type's identity, not a hint.

## 1.2 Type Inference vs Dynamic Typing `[FOUNDATIONAL]`

### Intuition

Imagine two tailors. The first takes your measurements once, at the very first fitting, and cuts a suit that is permanently your size — if you gain weight later, the suit doesn't stretch, it just stops fitting and the tailor will refuse to let you wear it. The second tailor doesn't measure you at all up front; instead, every single time you put the suit on, they re-measure you on the spot and adjust it to fit whatever your body is *that day*.

Mojo's `var a = 10` is the first tailor: it looks at the initializer once, at compile time, deduces "this is an `Int`," and locks that in forever — `a` can never hold anything but an `Int` for the rest of its life, even though you never typed the word `Int`. Python's `a = 10` is the second tailor: `a` has no fixed size at all, it's just a name that currently points at an `int` object, and nothing stops the very next line from pointing it at a `str` instead.

### Background

**Type inference** is a *compile-time* process: the compiler looks at the expression on the right-hand side of an assignment, determines its type through the same rules it would use if you'd written the annotation yourself, and then treats the variable as having that fixed type for the rest of its scope — exactly as if you had typed it explicitly. No inference happens while the program is running; by the time the program executes, every variable's type was already nailed down during compilation, identically to an explicitly-annotated variable.

**Dynamic typing** is fundamentally different: there is no compile-time type-fixing step at all, because the type isn't a property of the variable — it's a property of whatever object the variable currently references, and that reference can be repointed to an object of a completely different type at any later line.

| | Mojo `var a = 10` | Python `a = 10` |
|---|---|---|
| When is the type decided? | Once, at compile time | Never fixed — checked fresh on every use |
| Can `a` later hold a `String`? | No — compile error | Yes — `a = "ten"` is legal |
| Cost of "type inference" at runtime | Zero (already resolved) | N/A — there is no inference, only per-use checking |

### Worked Example 1.2.1 — The same-looking line, two different outcomes

```mojo
var a = 10        # compiler infers Int here, at compile time -- permanent
var b = 5.5       # compiler infers Float64 here -- permanent
var c = True      # compiler infers Bool here -- permanent
print("Inferred types - a:", a, "b:", b, "c:", c)
```

Trace what the compiler does at the moment it sees `var a = 10`: it evaluates the *type* of the literal `10` (an `Int`, by Mojo's default integer-literal rule), and from that point forward `a`'s type in the symbol table is `Int` — indistinguishable, for every purpose, from having written `var a: Int = 10`. If a later line in the same scope tried `a = "ten"`, the compiler would reject it at compile time, before the program ever runs, with a type mismatch — the same way it would reject `x = "ten"` for an explicitly-declared `var x: Int`.

```python
a = 10
print(type(a))    # <class 'int'>
a = "ten"
print(type(a))    # <class 'str'>  -- perfectly legal, no error anywhere
```

Here, nothing is fixed at any point. `type(a)` genuinely changes between the two `print` calls because `a` was never bound to a type — only ever to a value, and a new assignment simply points it at a new value of a new type.

### Worked Example 1.2.2 — Why this matters for a function signature

```mojo
fn double_it(n: Int) -> Int:
    return n * 2

var a = 10          # inferred Int
print(double_it(a)) # legal: a's fixed, compile-time type (Int) matches the parameter type
```

Because `a`'s type was resolved at compile time as `Int`, the call `double_it(a)` is checked at compile time too — the compiler confirms `Int` matches `Int` and emits a direct call with no runtime type check. The equivalent Python function must defensively check (or simply crash) if it's ever called with something that isn't a number, because nothing prevented `a` from becoming a `str` between its definition and the call.

> `[COMMON TRAP]` Seeing `var a = 10` with no colon-and-type after it, and concluding "Mojo must be dynamically typed here since there's no annotation," is the single easiest mistake to make when arriving from Python. The absence of a visible annotation says nothing about whether the type is fixed — it only means you're relying on the compiler to write the annotation for you. Two questions distinguish the cases in *any* language: (1) is the type decided before or during execution, and (2) can the same variable later hold a different type? Mojo answers "before, and no." Python answers "during, and yes."

## 1.3 Compile-Time Constants with `alias` `[FOUNDATIONAL]`

### Intuition

A page number printed on a physical book page is fixed the moment the book is printed — it can never change no matter what happens to the copy sitting on your shelf. A sticky note you've placed on that same page can be moved, rewritten, or removed at any time without touching the book itself. Both are "labels attached to a page," but one is baked in at production time and the other can change at any moment during use.

`alias` in Mojo is the printed page number: its value is substituted directly into the compiled program wherever it's used, during compilation, and cannot change once the program is running. `var` is the sticky note: its value lives in memory and can be read or overwritten while the program runs.

### Background

An `alias` declaration binds a name to a value that must be computable entirely at compile time, and every use of that name is replaced with the literal value before the program is ever run — there is no memory location for an `alias` the way there is for a `var`. This is closest to C++'s `constexpr` (a compile-time-evaluated constant) and stricter than C's `#define` (a raw text substitution the preprocessor performs with no type checking at all). Python has no real equivalent: a module-level `PI = 3.14159265359` is only a *naming convention* for "please don't reassign this" — Python enforces nothing, and `PI` is exactly as mutable as any other variable.

| | Mojo `alias` | Mojo `var` | C++ `constexpr` | Python module constant |
|---|---|---|---|---|
| Resolved | Compile time | Runtime | Compile time | Never — it's just a variable |
| Has a memory address at runtime | No | Yes | Not necessarily | Yes (it's a real object) |
| Can be reassigned | No — compile error | Yes | No | Yes — nothing stops it |

### Worked Example 1.3.1 — Tracing the substitution

```mojo
alias PI = 3.14159265359
alias MAX_SIZE = 1024

print("Constants - PI:", PI, "MAX_SIZE:", MAX_SIZE)
```

Mentally run the compiler's substitution pass on this snippet: every occurrence of the identifier `PI` in the source text is replaced with the literal `3.14159265359`, and every occurrence of `MAX_SIZE` is replaced with `1024`, *before* code generation happens. The line above compiles as though it had literally been written `print("Constants - PI:", 3.14159265359, "MAX_SIZE:", 1024)`. There is no variable named `PI` in the compiled binary at all — no stack slot, no memory read at runtime. Contrast this with `var x: Int = 42` from Section 1.1, which *does* reserve a stack slot that the running program reads from every time `x` is used.

### Worked Example 1.3.2 — Where the distinction changes behavior

```mojo
alias MAX_SIZE = 1024
var buffer = UnsafePointer[Int].alloc(MAX_SIZE)   # allocation size fixed at compile time
```

Because `MAX_SIZE` is an `alias`, the compiler knows the allocation size `1024` while compiling this line — this can enable compile-time bounds checking and even let the compiler choose a fixed-size stack buffer instead of a heap allocation in some contexts, optimizations that are only possible because the size is guaranteed never to change at runtime. Rewrite it with `var buffer_size = 1024` instead, and the same `.alloc(buffer_size)` call becomes a runtime-determined size as far as the compiler's optimizer is concerned, even though `1024` happens to be the only value ever assigned to it — the compiler cannot *prove* it stays `1024` the way it can for an `alias`, because nothing in the language stops a later line from writing `buffer_size = 2048`.

## 1.4 SIMD as a First-Class Type `[FOUNDATIONAL]`

### Intuition

Picture two ways of stapling 1,000 sheets of paper. The first way: one worker staples one sheet, then the next, then the next — 1,000 individual staple actions. The second way: four workers stand side by side, each staples one sheet at exactly the same moment, so every "round" processes four sheets for the cost of one — 250 rounds instead of 1,000 individual actions.

A normal `Float32` variable is the first worker — one value, one operation. `SIMD[DType.float32, 4]` is the four workers standing side by side: it is a single value that actually holds four `Float32` numbers in one CPU vector register, and a single `+` or `*` on it operates on all four simultaneously, in the same number of clock cycles a single scalar operation would take.

### Background

**SIMD** stands for Single Instruction, Multiple Data — one CPU instruction that operates on several data elements ("lanes") packed into one wide register, at the same time. In Mojo, `SIMD[DType, width]` is a genuine built-in type, not a library-provided abstraction over the hardware feature: arithmetic operators (`+`, `*`, `pow`, comparisons) are defined directly on it and compile straight down to the CPU's native vector instructions. In C or C++, achieving the same thing traditionally means either hoping the compiler's auto-vectorizer notices the opportunity, or calling explicit *intrinsic* functions (`_mm_add_ps`, `_mm256_mul_ps`) that are ugly, non-portable across CPU vendors, and easy to get wrong. Python has no language-level SIMD type at all — a NumPy array gets the *benefit* of vectorization, but only because NumPy's own C internals do the vectorizing outside the Python language; from Python's perspective an array is just a library object, not a numeric type the language itself understands.

| | Mojo `SIMD[dtype, width]` | C/C++ manual intrinsics | Python (NumPy array) |
|---|---|---|---|
| Vectorization is | A language-level type | Explicit intrinsic function calls | Delegated to a C library underneath |
| Portable across CPU vendors | Yes (compiler targets the right instruction) | No (SSE/AVX/NEON intrinsics differ) | Yes (NumPy hides it) |
| `a + b` on 4 values | One vector-add instruction | Must call `_mm_add_ps(a, b)` explicitly | One Python-level call into C, which then vectorizes |

### Worked Example 1.4.1 — Four lanes, traced by hand

```mojo
var simd_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var simd_squared = simd_vec * simd_vec
```

`simd_vec` packs four `Float32` values into lanes `0` through `3`: lane 0 = `1.0`, lane 1 = `2.0`, lane 2 = `3.0`, lane 3 = `4.0`. The multiply `simd_vec * simd_vec` is a single vector instruction that multiplies each lane by *itself*, independently and simultaneously:

```
lane 0: 1.0 * 1.0 =  1.0
lane 1: 2.0 * 2.0 =  4.0
lane 2: 3.0 * 3.0 =  9.0
lane 3: 4.0 * 4.0 = 16.0
```

`simd_squared` is therefore `[1.0, 4.0, 9.0, 16.0]` — four multiplications, one instruction. A scalar loop computing the same four values would need four separate multiply instructions; the SIMD version needs one.

### Worked Example 1.4.2 — A second operation, `pow`, confirming the pattern

```mojo
var simd_cubed = pow(simd_vec, 3.0)
```

`pow` applied to a `SIMD` value cubes every lane independently, exactly like the multiply above: `1.0³=1.0`, `2.0³=8.0`, `3.0³=27.0`, `4.0³=64.0`, giving `[1.0, 8.0, 27.0, 64.0]`. This is the same operation, applied lane-wise, regardless of which arithmetic function is used — the "four workers" pattern from the intuition holds for any operator or built-in function defined on `SIMD`.

### ASCII Diagram — one register, four lanes, one instruction

```
SIMD[DType.float32, 4] register:

 lane:     0        1        2        3
        +--------+--------+--------+--------+
 value: |  1.0   |  2.0   |  3.0   |  4.0   |
        +--------+--------+--------+--------+
              |        |        |        |
              v        v        v        v      <- ONE multiply instruction,
        +--------+--------+--------+--------+       all four lanes at once
 result:|  1.0   |  4.0   |  9.0   | 16.0   |
        +--------+--------+--------+--------+
```

Part 5 of this book (GPU acceleration) reuses this exact lane-parallel idea at a much larger scale — a GPU kernel launches thousands of threads that each play the role of one "lane" here, executing the same instruction on different data simultaneously.

## 1.5 Heap Memory and `UnsafePointer` `[FOUNDATIONAL]`

### Intuition

A hotel room you book for a trip is not yours forever — you're given a key, you use the room for as long as you need it, and then you *must* check out so the hotel can hand that same room to the next guest. If you lose the key and never check out, the hotel has no way to know the room is actually empty; from its bookkeeping, that room is occupied forever, unusable by anyone else, even though nobody is in it. That permanently-and-uselessly-occupied room is exactly what a **memory leak** is.

The stack variables from Section 1.1 (`x`, `y`, `z`) are more like a rucksack you carry yourself — when the function that declared them ends, the rucksack (the whole stack frame) is dropped all at once, automatically, with no separate action required for each item inside it. Heap memory obtained through `UnsafePointer.alloc()` is the hotel room: it must be explicitly "checked out" with `.free()`, and nothing happens automatically when the enclosing function ends.

### Background

Mojo's stack variables are automatically reclaimed when their scope ends — the same automatic behavior C, C++, and Mojo all share for local variables. Heap memory is different in all of these languages: it must be requested explicitly (`malloc` in C, `new` in C++, `UnsafePointer[T].alloc(n)` in Mojo) and released explicitly (`free` in C, `delete` in C++, `.free()` in Mojo) — nothing reclaims it automatically. Python sidesteps this entirely for ordinary code: every Python object, whether it looks like a "stack" variable or not, is secretly heap-allocated and reference-counted, and the interpreter calls the equivalent of `free()` for you the moment the reference count reaches zero. `UnsafePointer` in Mojo is deliberately the *manual*, C-like model, not the Python model — the "Unsafe" in the name is a direct warning that the compiler will not stop you from forgetting to free it, reading past the end of it, or freeing it twice.

| | Stack `var` (Mojo/C/C++) | Heap via `UnsafePointer`/`malloc`/`new` | Python object (any variable) |
|---|---|---|---|
| Reclaimed automatically? | Yes, at end of scope | No — must call `.free()`/`free()`/`delete` | Yes, via reference counting / GC |
| Forgetting to release it | Impossible — not a choice you make | Memory leak | Impossible — not a choice you make |
| Releasing it twice | Impossible | Undefined behavior (double free) | Impossible |

### Worked Example 1.5.1 — Allocating, filling, and freeing five integers

```mojo
var size = 5
var heap_ptr = UnsafePointer[Int].alloc(size)

for i in range(size):
    heap_ptr[i] = i * i

for i in range(size):
    print("  heap_ptr[", i, "] =", heap_ptr[i])

heap_ptr.free()
```

Trace the allocation first: `.alloc(5)` requests room for five `Int`s from the heap. Since Mojo's `Int` is 8 bytes (Section 1.1), this is a request for `5 × 8 = 40` bytes, and the heap allocator returns the starting address of that 40-byte block — call it address `A`. `heap_ptr[i]` is address arithmetic: `heap_ptr[i]` really means "read/write the 8 bytes starting at `A + i×8`." The fill loop computes `i × i` for `i = 0..4`:

```
i=0: 0*0 = 0    -> written at address A+0
i=1: 1*1 = 1    -> written at address A+8
i=2: 2*2 = 4    -> written at address A+16
i=3: 3*3 = 9    -> written at address A+24
i=4: 4*4 = 16   -> written at address A+32
```

The read loop then prints exactly those five values back: `0, 1, 4, 9, 16`. Finally, `heap_ptr.free()` returns the entire 40-byte block to the heap allocator — after this line, reading `heap_ptr[0]` again would be reading freed memory, which is undefined behavior, not a checked error, because `UnsafePointer` (true to its name) does not track whether it's still valid.

### ASCII Diagram — a 40-byte heap block, five `Int`s wide

```
Heap block returned by alloc(5), starting at address A:

        A+0      A+8      A+16     A+24     A+32
       +--------+--------+--------+--------+--------+
value: |   0    |   1    |   4    |   9    |   16   |
       +--------+--------+--------+--------+--------+
       heap_ptr[0]  [1]      [2]      [3]      [4]

After heap_ptr.free():  the allocator may hand this exact 40-byte
range to the very next .alloc() call anywhere in the program --
whatever "0, 1, 4, 9, 16" meant here is gone, overwritten by
whatever the next owner writes.
```

> `[COMMON TRAP]` The single most common `UnsafePointer` bug is calling `.free()` and then continuing to use the pointer — "use after free" — because nothing about the `heap_ptr` variable itself changes when you free it; it's still sitting there holding the same numeric address, now pointing at memory that's no longer yours. The second most common bug is the mirror image: allocating inside a loop and never freeing, which leaks a little more memory on every iteration until the program exhausts it. Chapter 2 (Memory Management) builds the reference-counted and arena-allocator patterns this book uses specifically to stop writing raw `alloc`/`free` pairs by hand for anything beyond a small, short-lived example like this one.

## Chapter Summary

A type is a compile-time promise about what a region of memory means, and Mojo, like C and C++, resolves every variable's type before the program runs — whether you write the type explicitly (`var x: Int = 42`) or let the compiler infer it from the initializer (`var a = 10`), the result is identical: a fixed type, checked once, at compile time, never rechecked while the program runs. This is the fundamental difference from Python, where a name has no type of its own and simply points at whichever object it was last assigned, with the type tag living on the object and being checked on every use. `alias` sits one level further toward the compiler than `var`: an `alias` value is substituted directly into the compiled code and has no runtime memory location at all, unlike a `var`, which always occupies a real, readable, writable slot. `SIMD[DType, width]` extends the same static-typing discipline to vector hardware, making "operate on four numbers at once" a property of the type itself rather than something bolted on through library calls or hand-written CPU intrinsics. And heap memory obtained via `UnsafePointer` restores full manual control — the same control (and the same risks: leaks, use-after-free, double-free) that C's `malloc`/`free` and C++'s `new`/`delete` have always required, deliberately not hidden behind the automatic reference counting that makes every Python object effortless to allocate and forget about.

## Self-Check Questions

1. `var a = 10` and `a = 10` (Python) both omit an explicit type annotation. Explain precisely why the Mojo line is still statically typed and the Python line is not — what question distinguishes the two cases?
2. Why does `alias MAX_SIZE = 1024` have no runtime memory address, while `var buffer_size: Int = 1024` does — even though both, in this example, hold the value `1024` for the entire life of the program?
3. A `SIMD[DType.float32, 4]` value holds `[2.0, 4.0, 6.0, 8.0]`. What is the result of squaring it, and how many actual CPU instructions does that squaring take compared to squaring the same four numbers stored as four separate `Float32` variables in a loop?
4. `UnsafePointer[Int].alloc(3)` is called, the three slots are filled with `10`, `20`, `30`, and then `.free()` is called. What specifically becomes undefined behavior immediately afterward, and why doesn't Mojo simply stop you from doing it (as Python would, by construction)?
5. Explain, in one or two sentences, why Mojo is compared against C/C++ for *memory and performance behavior* in this chapter, but against Python for *syntax and ergonomics* — what does each comparison actually establish?

## Where We Go Next

Every type introduced in this chapter — `Int`, `Float32`, `SIMD[DType, width]` — has been a *primitive*: a single, indivisible value with a fixed, built-in meaning the compiler already knows. Chapter 2 introduces `struct`, Mojo's tool for defining an entirely new type out of your own choice of fields — and does so by first tracing how C and C++ solve the exact same problem, since Mojo's `struct` is deliberately closer in spirit to C++'s than to anything Python offers natively.

## Worked Solutions

**1.** The distinguishing question is: *is the type fixed before the program runs, and can the same variable later hold a different type?* For `var a = 10` in Mojo, the compiler examines the literal `10` during compilation, assigns `a` the type `Int` in its symbol table, and rejects any later attempt in the same scope to assign a non-`Int` value to `a` — the type is fixed at compile time, permanently. For `a = 10` in Python, there is no symbol-table type-fixing step at all; `a` is just a name bound to a heap object, and `a = "ten"` on the very next line is completely legal, because the type lives on the object, not on the name.

**2.** `alias MAX_SIZE = 1024` is a compile-time substitution: every occurrence of `MAX_SIZE` in the source is textually/semantically replaced by the literal `1024` before code generation, so there is nothing left to *read* at runtime — no address, no memory access. `var buffer_size: Int = 1024` reserves an actual 8-byte stack slot that the running program reads from every time `buffer_size` is used, even though, in this particular example, nothing ever changes what's stored there — the compiler cannot assume that in general, since `var` permits reassignment.

**3.** Squaring `[2.0, 4.0, 6.0, 8.0]` lane-wise gives `[4.0, 16.0, 36.0, 64.0]` (`2²=4`, `4²=16`, `6²=36`, `8²=64`). The `SIMD` version takes one vector-multiply instruction for all four lanes simultaneously. Four separate scalar `Float32` variables squared in a loop take four separate scalar-multiply instructions — four times the instruction count for the identical arithmetic result.

**4.** After `.free()`, any further read or write through `heap_ptr` (e.g., `heap_ptr[0]`) is a **use-after-free**: the 40-... (here, 24-byte, for three `Int`s) block has been returned to the allocator and may already have been handed to a completely unrelated `.alloc()` call elsewhere in the program, so reading it reads garbage (or someone else's data) and writing it corrupts someone else's allocation. Mojo doesn't prevent this by construction because `UnsafePointer` is explicitly the manual, no-safety-net memory primitive — the same tradeoff C's raw pointers make — precisely so that Chapter 2's higher-level, *safe* abstractions (reference-counted buffers, arenas) can be built on top of it without paying for safety checks in the one place that needs to be as fast as C.

**5.** The C/C++ comparison establishes *mechanism*: what the compiler and hardware actually do with a given piece of code — stack layout, instruction counts, manual allocation — because Mojo's performance model is built to match theirs. The Python comparison establishes *ergonomics*: how the code reads and feels to write — inferred types, no manual memory management by default — because Mojo's syntax is designed to feel approachable to programmers coming from Python. Neither comparison alone describes Mojo; the language's whole premise is occupying both positions at once.
