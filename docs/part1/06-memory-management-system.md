# Chapter 11: Memory Management System — Shared Ownership, Bump Allocation, and Pooling

> "Chapter 6 gave every tensor exactly one owner, and one place — `__del__` — where that owner's memory gets freed. That rule is simple, correct, and too rigid for what's coming: a computational graph that needs a tensor's buffer visible from two places at once. This chapter is where 'exactly one owner' becomes 'as many owners as the graph needs, safely' — and where two of the three new tricks that make that possible turn out to have a real gap of their own."

**What you will understand by the end of this chapter:**

- `RefCountedBuffer`: genuine shared ownership, where *any* of several live copies can validly be the one whose destructor actually frees the memory — a real generalization of Chapter 6's single-owner `_owns_data` flag and Chapter 10's copy-never-owns `MemoryBlock`, not just a variation on either
- The bump-pointer `Arena`: why its 64-byte alignment arithmetic is correct as written, and why its one safety check — `debug_assert` — is a debug-build-only guard that silently disappears in the release build a real training run would actually use
- `GPUMemoryPool`'s size-classed free list, traced through three training steps to a `2/3` hit rate — and a destructor this struct never got, which leaks every device buffer it ever hands out for the lifetime of the pool
- Why this chapter's own closing claim — "there is no code path that can leak a resource" — is contradicted by a struct defined one section earlier in the very same chapter, and why the `UnsafePointer` its RAII example passes around doesn't actually get the compile-time safety the surrounding comment claims for it

**What you need to know first:**

- Chapter 6 (`Tensor`'s single-owner `_owns_data` flag, and RAII as `__init__`-allocates / `__del__`-frees)
- Chapter 7.2 (`TensorView`, the direct consumer of this chapter's `RefCountedBuffer` — Section 11.1 is the machinery that section's sharing depends on)
- Chapter 3 (row-major and cache-line reasoning — `Arena`'s 64-byte alignment choice is that reasoning applied to a bump allocator)
- Chapter 2.4 (constructors, destructors, and why every buffer-owning struct in this book pairs an `alloc` with a `.free()`)

<a id="21-reference-counting-implementation"></a>
## 11.1 Reference Counting: When More Than One Owner Needs the Same Buffer `[FOUNDATIONAL]`

### Intuition

Two roommates share a one-bedroom apartment. Neither of them individually decides when the lease ends — the landlord only reclaims the apartment once *both* have moved out, and it doesn't matter which one happens to be the one who hands back the last key. Chapter 10's `MemoryBlock` didn't work this way: it had exactly one legitimate owner at a time, and a copy explicitly gave up any claim to free the memory (`_owns_data = False`, set the moment a copy is made), so exactly one specific value was always "the" owner. `RefCountedBuffer` is the roommate arrangement instead — a small shared counter, allocated alongside the data, that every copy increments on arrival and decrements on departure, so that *whichever* copy happens to go out of scope last is correctly the one that frees the memory, without either copy needing to know in advance which one that will be.

### Background

```mojo
struct RefCountedBuffer[T: AnyType]:
    var _data: UnsafePointer[T]
    var _refcount: UnsafePointer[Int]
    var count: Int
```

| | Chapter 6 `Tensor` (`_owns_data`) | Chapter 10 `MemoryBlock` | `RefCountedBuffer` (this chapter) |
|---|---|---|---|
| How many owners at once | exactly one | exactly one (a copy explicitly opts out) | any number, tracked live |
| Who frees the memory | the one flagged owner | the one flagged owner | whichever copy's destructor sees the count hit zero |
| Cost per copy | none — copies aren't meant to alias | none — copies aren't meant to free | one shared-memory increment |
| Cost per destruction | one flag check | one flag check | one shared-memory decrement plus a zero check |

### Worked Example 11.1.1 — The count, traced through three lifetimes

`var t = Tensor.zeros([4])` allocates the buffer and, inside `RefCountedBuffer.__init__`, sets `self._refcount[0] = 1`. `var v = t.view()` triggers `__copyinit__`: `v` gets the *same* `_data` and `_refcount` pointers as `t` (not copies of the data itself), and `self._refcount[0] += 1` brings the shared count to `2`. If `v` goes out of scope first, its `__del__` runs: `self._refcount[0] -= 1` drops the count to `1`, and since `1 != 0`, the `if` guard is false — the buffer survives, because whatever `_refcount` now points at still reads `1`, and `t`'s own copy of that same `_refcount` pointer sees the identical value. Only when `t` *also* goes out of scope does its `__del__` decrement the count to `0`, at which point *that* call's `if self._refcount[0] == 0` is true, and it frees both `_data` and `_refcount`. Neither `t` nor `v` ever needed to check whether the other was still alive — the shared integer they both point at carried that information for both of them.

```
Chapter 7.2's TensorView is the direct consumer: `parent.refcount()` reports
`2` for exactly as long as any view built from `parent` is alive, and the
underlying allocation only frees once every view and the owning tensor have
each run their own `__del__`.
```

One honest scope limit worth naming: `__copyinit__` copies `existing.count` directly, so every share of a `RefCountedBuffer` reports the *same* element count as whatever it was copied from. This struct can express "two things looking at the exact same whole buffer" but has no field for "a view of just the first ten elements of a hundred-element buffer, sharing that same allocation." That finer-grained slicing — an independent length and offset layered on top of a shared buffer — is exactly what Chapter 7.2's `SliceSpec`/`TensorView` machinery adds; `RefCountedBuffer` on its own only ever hands out whole-buffer aliases.

<a id="22-arena-based-memory-allocation"></a>
## 11.2 Arena-Based Allocation: Trading Individual Frees for One Bulk Reset `[FOUNDATIONAL]`

### Intuition

A coat check that issues a numbered tag for every coat, and requires that exact tag back before releasing it, does real bookkeeping on every single coat, all night. A coat check that instead says "we close at midnight, and every coat gets returned to its owner in one pass, whatever order they show up in" does none of that per-coat bookkeeping at all — it just needs to know where the coats end and the empty space begins. `Arena` is the second kind of coat check applied to memory: instead of tracking each allocation's lifetime individually (as `RefCountedBuffer` does), it hands out increasing offsets into one big pre-allocated slab with nothing more than addition, and reclaims *everything* at once with a single `reset()` — appropriate exactly when every allocation in the batch is known to die together, the way a whole training step's intermediate activations do.

### Background

```mojo
struct Arena:
    var _base: UnsafePointer[UInt8]
    var _capacity: Int
    var _offset: Int
```

`alloc[T](count)` computes `bytes_needed = count * sizeof[T]()`, rounds `self._offset` up to the next 64-byte boundary with `(self._offset + 63) & ~63`, checks the result still fits inside `_capacity` via `debug_assert`, and advances `_offset` past the new allocation. `reset()` is one line: `self._offset = 0`. Nothing here does a per-allocation free; everything is reclaimed at once when the arena resets or is dropped.

### Worked Example 11.2.1 — The alignment arithmetic, bit by bit

`(offset + 63) & ~63` is the standard round-up-to-a-power-of-two-boundary idiom: adding `63` (one less than the 64-byte alignment) pushes any offset that isn't already a multiple of 64 up into the next block, and `& ~63` (clearing the low 6 bits) then rounds back down to that block's start — landing exactly on the next multiple of 64 at or above the original offset. Traced against the chapter's own numbers: first request, `_offset = 0`, wants `100` `Float32`s (`400` bytes) — `(0 + 63) & ~63`: `63` in binary has only its low 6 bits set, so clearing those 6 bits leaves `0`. Already aligned; the allocation runs from byte `0` to byte `400`, and `_offset` becomes `400`. Second request, `50` more `Float32`s (`200` bytes): `(400 + 63) & ~63 = 463 & ~63`. `463 = 7×64 + 15`, so clearing the low 6 bits drops the `15` and leaves `448` — the allocation runs from `448` to `648`, having quietly skipped `448 - 400 = 48` bytes of padding to land on a cache-line boundary.

### Worked Example 11.2.2 — Bounded waste, in exchange for zero bookkeeping

That `48`-byte gap is the price of alignment, and it's bounded: rounding up to a 64-byte boundary can never waste more than `63` bytes on any single allocation, regardless of how many allocations came before it or how large this one is. Compare that fixed, small, per-allocation ceiling to `RefCountedBuffer`'s cost model from Section 11.1 — a shared-memory increment and decrement on every copy and every destruction — and to Chapter 10's `MemoryAllocator`, which updates several running counters (`total_allocated_bytes`, `active_allocations`, `peak_allocations`) on every call. `Arena.alloc` touches none of that: one addition, one bitwise mask, one bounds check, and a pointer arithmetic step — the entire reason a training step with thousands of small per-layer allocations reaches for this structure instead of `RefCountedBuffer` or a general-purpose allocator.

```
[COMMON TRAP]  "Arena exhausted" is a debug-build-only promise

debug_assert(aligned_offset + bytes_needed <= self._capacity, "Arena
exhausted") is the *only* bounds check alloc() performs — and debug_assert
is compiled out entirely in a release build, the build configuration an
actual training run uses. In debug mode, requesting more memory than
capacity_bytes allows correctly halts the program with a clear message. In
release mode, that same over-capacity request silently proceeds: aligned_offset
+ bytes_needed exceeds self._capacity, the returned pointer lands past the
end of the buffer alloc() originally reserved, and every read or write
through it touches memory the Arena was never granted — a silent, undetected
overrun, in exactly the build configuration where nothing else in this
function would catch it. Nothing about Arena's own code or this chapter's
prose calls this out; it's only visible by treating debug_assert as what it
actually is (a development-time aid) rather than as production-grade
capacity enforcement.
```

<a id="23-gpu-memory-management"></a>
## 11.3 GPU Memory Pooling: Reuse Instead of Re-Asking the Driver `[FOUNDATIONAL]`

### Intuition

A construction site that rents a specific size of scaffolding for every job, returns it when the job's done, and rents an identical size again the very next week, is paying a rental company's overhead twice for equipment it could have simply kept on-site between jobs. `GPUMemoryPool` is the "keep it on-site" version for GPU buffers: instead of returning device memory to the driver the moment a training step finishes (an expensive round trip, one to three orders of magnitude slower than a CPU allocation), it holds released buffers in a free list, bucketed by exact element count, ready to hand the *same* buffer straight back out the next time something asks for that exact size — which, across training steps that repeat the same layer shapes every iteration, is most of the time.

### Background

```mojo
struct GPUMemoryPool:
    var ctx: DeviceContext
    var _free_lists: Dict[Int, List[UnsafePointer[Scalar[DType.float32]]]]
    var bytes_allocated: Int
    var bytes_reused: Int
```

`acquire(count)` checks whether `_free_lists[count]` has anything in it; if so, it pops and returns an existing buffer (`bytes_reused += count * 4`) instead of asking the driver for new memory. Otherwise it allocates fresh (`bytes_allocated += count * 4`). `release(ptr, count)` never frees anything — it appends `ptr` to the free list for that size, keeping it available for the next `acquire` of the same count.

### Worked Example 11.3.1 — Three steps, one buffer, climbing toward a hit rate of 1.0

Step 1 requests a `256`-element buffer: the free list for size `256` is empty, so the pool allocates fresh (`bytes_allocated = 256 × 4 = 1024`) and hands it out; at the end of the step it's released back, not freed, leaving one entry in that size's free list. Step 2 requests another `256`-element buffer: the free list isn't empty this time, so the pool pops that *same* buffer and reuses it (`bytes_reused = 1024`) instead of touching the driver. Step 3: identical story, `bytes_reused = 2048`. After three steps, `hit_rate() = bytes_reused / (bytes_allocated + bytes_reused) = 2048 / (1024 + 2048) = 0.667` — and every step after this one that requests the same `256`-element size reuses the same buffer yet again, pushing the ratio toward `1.0` as training continues, since real training loops request the same handful of activation shapes on every iteration.

```
[COMMON TRAP]  GPUMemoryPool has no __del__ — it leaks every buffer it
                ever allocates, for the pool's entire lifetime

Look for a destructor on GPUMemoryPool and there isn't one. Mojo synthesizes
a default, fieldwise destructor for a struct that defines none — meaning
when a GPUMemoryPool goes out of scope, self.ctx, self._free_lists,
self.bytes_allocated, and self.bytes_reused are each torn down using their
own types' destruction rules. Dict and List's own destructors correctly free
the *bookkeeping* structures — the hash table, the array of pointer values
sitting inside each free list — but UnsafePointer has no automatic freeing
behavior at all (exactly why every other allocating struct in this book,
Arena and RefCountedBuffer included, explicitly calls .free() in its own
__del__). Freeing a List[UnsafePointer[...]] frees the list's own backing
array — the slots that held the addresses — never the device memory each
address actually points to. Every buffer this pool ever created through
acquire() and never handed back out again before the pool itself is dropped
is gone, in the sense that nothing can reach it anymore, while the memory it
occupies is never returned to the driver: a textbook leak, and one that
directly contradicts the claim Section 11.4 is about to make one paragraph
later in this very chapter.
```

## 11.4 RAII Patterns and Automatic Cleanup `[FOUNDATIONAL]`

### Intuition

Every allocating struct so far in this book has followed one rule: acquire the resource in `__init__`, release it in `__del__`, and let the compiler's ownership tracking make sure that pairing actually happens. This section states that rule as the chapter's closing claim — and is the right place to check whether every struct in the chapter actually lives up to it.

### Background

```mojo
fn training_step(mut pool: GPUMemoryPool, batch: Tensor):
    var activations = pool.acquire(batch.shape.size())
    # ... forward + backward pass using `activations` ...
    pool.release(activations, batch.shape.size())
    # `activations` is not read again after this line -- Mojo's
    # lifetime checker rejects any attempt to do so.
```

The chapter's own text states: "every allocator in this chapter follows the same rule Chapter 6's `Tensor` established: acquisition happens in `__init__`, release happens in `__del__`, and there is no code path that can leak a resource."

```
[COMMON TRAP]  two claims in this section, checked against the chapter's
                own structs

First: "every allocator in this chapter follows the same rule... release
happens in __del__." Section 11.3's GPUMemoryPool doesn't — it has no
__del__ at all, as just traced. The rule holds for RefCountedBuffer and
Arena, both of which do define an explicit destructor; it does not hold for
the third struct this same chapter builds, one section earlier.

Second, narrower claim, in the code comment itself: "Mojo's lifetime checker
rejects any attempt to [read `activations` again after release]." acquire()
returns an UnsafePointer, and release() takes one as a plain (non-owned)
parameter — copying a raw pointer value, not consuming or moving out of
`activations` the way passing an `owned` value would. UnsafePointer is, by
design, a trivially copyable type exempt from Mojo's ownership and lifetime
tracking (that exemption is exactly what the "Unsafe" in its name signals) —
so nothing at the type-system level actually prevents `activations` from
being read again after this release() call, or prevents calling
pool.acquire(...) a second time and comparing pointers, or any other reuse a
programmer might attempt. The safety this comment describes is real only as
programmer discipline, not as something the compiler enforces for this
particular type — which matters precisely because it's asserted in the one
section of the chapter dedicated to explaining why resource leaks and misuse
supposedly can't happen here.
```

Taken together with `Arena`'s debug-only bounds check from Section 11.2, this chapter's three structures don't quite earn the blanket safety claim its own closing section makes for them: `RefCountedBuffer`'s shared-ownership discipline genuinely works as described, but `Arena`'s protection against overrun disappears in release builds, and `GPUMemoryPool` — the very struct whose reuse this section's example code demonstrates — has no cleanup path at all. Combined, Part 1 now has a real memory story: deterministic single ownership from Chapter 6, shared ownership where a graph needs it (Section 11.1), bump allocation where per-step churn dominates (Section 11.2), and a device-side pool where the *allocation itself*, not the memory, is the expensive resource (Section 11.3) — with two specific, traceable gaps in that story now on the record rather than taken on faith.

## 11.5 Reference Implementations

Unlike every other chapter in Part 1, this chapter's source was written as annotated reference snippets embedded directly in prose — hand-traced values in comments, no standalone `.mojo` test files, and no captured console output to check numeric claims against. The four snippets below are reproduced exactly as they appear in the original source, in reading order, for exactly that reason: there is no separate "Expected Output" to preserve here, because none was ever captured. Every numeric trace in this chapter's worked examples was verified by hand against the arithmetic in these snippets and the prose's own inline comments instead.

### `RefCountedBuffer` — Section 11.1

```mojo
from memory import UnsafePointer

struct RefCountedBuffer[T: AnyType]:
    """A manually reference-counted heap buffer.

    Layout: a single allocation holds an Int refcount header
    immediately followed by `count` elements of T, so incrementing
    or decrementing the count touches no extra cache line.
    """
    var _data: UnsafePointer[T]
    var _refcount: UnsafePointer[Int]
    var count: Int

    fn __init__(out self, count: Int):
        self.count = count
        self._refcount = UnsafePointer[Int].alloc(1)
        self._refcount[0] = 1                          # t = Tensor.zeros([4]): refcount starts at 1
        self._data = UnsafePointer[T].alloc(count)

    fn __copyinit__(out self, existing: Self):
        # Shallow copy: share the buffer, bump the count.
        self._data = existing._data
        self._refcount = existing._refcount
        self.count = existing.count
        self._refcount[0] += 1                          # v = t.view(): refcount becomes 2

    fn __del__(owned self):
        self._refcount[0] -= 1                          # v goes out of scope: refcount becomes 1
        if self._refcount[0] == 0:                       # t later goes out of scope: refcount becomes 0
            self._data.free()
            self._refcount.free()

    fn refcount(self) -> Int:
        return self._refcount[0]
```

### `Arena` — Section 11.2

```mojo
struct Arena:
    """Bump-pointer allocator over one large backing buffer.

    Individual allocations are never freed; the whole arena resets
    (or is dropped) once, at graph-teardown time. This turns O(n)
    tensor-destructor calls during a training step into O(1).
    """
    var _base: UnsafePointer[UInt8]
    var _capacity: Int
    var _offset: Int

    fn __init__(out self, capacity_bytes: Int):
        self._base = UnsafePointer[UInt8].alloc(capacity_bytes)
        self._capacity = capacity_bytes
        self._offset = 0

    fn __del__(owned self):
        self._base.free()

    fn alloc[T: AnyType](mut self, count: Int) -> UnsafePointer[T]:
        var bytes_needed = count * sizeof[T]()
        # 64-byte alignment keeps every allocation SIMD- and
        # cache-line-friendly (see Part 0.3 on AoS vs SoA layouts).
        var aligned_offset = (self._offset + 63) & ~63
        debug_assert(aligned_offset + bytes_needed <= self._capacity,
                     "Arena exhausted")
        var ptr = (self._base + aligned_offset).bitcast[T]()
        self._offset = aligned_offset + bytes_needed
        return ptr

    fn reset(mut self):
        """O(1) reclaim of every allocation made since the last reset."""
        self._offset = 0

    fn bytes_used(self) -> Int:
        return self._offset
```

### `GPUMemoryPool` — Section 11.3

```mojo
from gpu.host import DeviceContext
from memory import UnsafePointer
from collections import Dict, List

struct GPUMemoryPool:
    """Size-classed free list of device buffers, so repeated
    same-shape tensor allocations (very common across training
    steps) reuse memory instead of round-tripping the driver."""
    var ctx: DeviceContext
    var _free_lists: Dict[Int, List[UnsafePointer[Scalar[DType.float32]]]]
    var bytes_allocated: Int
    var bytes_reused: Int

    fn __init__(out self, owned ctx: DeviceContext):
        self.ctx = ctx^
        self._free_lists = Dict[Int, List[UnsafePointer[Scalar[DType.float32]]]]()
        self.bytes_allocated = 0
        self.bytes_reused = 0

    fn acquire(mut self, count: Int) -> UnsafePointer[Scalar[DType.float32]]:
        if count in self._free_lists and len(self._free_lists[count]) > 0:
            self.bytes_reused += count * 4
            return self._free_lists[count].pop()
        self.bytes_allocated += count * 4
        return UnsafePointer[Scalar[DType.float32]].alloc(count)

    fn release(mut self, ptr: UnsafePointer[Scalar[DType.float32]], count: Int):
        if count not in self._free_lists:
            self._free_lists[count] = List[UnsafePointer[Scalar[DType.float32]]]()
        self._free_lists[count].append(ptr)

    fn hit_rate(self) -> Float64:
        var total = self.bytes_allocated + self.bytes_reused
        return Float64(self.bytes_reused) / Float64(total) if total > 0 else 0.0
```

### RAII usage pattern — Section 11.4

```mojo
fn training_step(mut pool: GPUMemoryPool, batch: Tensor):
    var activations = pool.acquire(batch.shape.size())
    # ... forward + backward pass using `activations` ...
    pool.release(activations, batch.shape.size())
    # `activations` is not read again after this line -- Mojo's
    # lifetime checker rejects any attempt to do so.
```

## Chapter Summary

`RefCountedBuffer` generalizes Chapter 6's single-owner `_owns_data` flag and Chapter 10's copy-never-owns `MemoryBlock` into genuine shared ownership: a counter incremented on every copy and decremented on every destruction, so that whichever live copy happens to see the count reach zero is correctly the one that frees the buffer, traced end to end through a three-step lifetime (`1 → 2 → 1 → 0`). `Arena` trades that per-copy bookkeeping for a bump-pointer scheme — one addition and a 64-byte alignment mask per allocation, verified by hand against the chapter's own `0 → 400 → 448 → 648` offset sequence — with individual frees replaced entirely by one bulk `reset()`, at the cost of a bounds check (`debug_assert`) that silently disappears in a release build. `GPUMemoryPool` trades the *allocation itself* for reuse, bucketing released device buffers by exact element count and climbing toward a hit rate of `1.0` as the same shapes recur across training steps — traced to an exact `0.667` after three steps — but the struct itself has no destructor at all, leaking every device buffer it creates for as long as the pool lives, a gap that directly undercuts this chapter's own closing claim that "every allocator in this chapter" leaks nothing, and that sits right next to a second overclaim: the comment asserting Mojo's lifetime checker polices reuse of a released `UnsafePointer`, a type explicitly exempted from that tracking by design.

## Self-Check Questions

1. `var a = RefCountedBuffer[Float32](10)`, then `var b = a`, then `var c = b`. In what order would the refcount need to reach `0` for the underlying buffer to be freed, and does it matter which of `a`, `b`, or `c` goes out of scope last?
2. An `Arena` with `_offset = 200` receives a request for `20` `Float32`s (`80` bytes). Compute `(200 + 63) & ~63` by hand the way Worked Example 11.2.1 did, state the resulting aligned offset, and state how many bytes of padding were skipped.
3. A release-mode program calls `arena.alloc[Float32](count)` with a `count` large enough that `aligned_offset + bytes_needed` exceeds `self._capacity`. Given that `debug_assert` is compiled out in release builds, what does `alloc` actually do in this situation, and what does the pointer it returns actually point to?
4. A `GPUMemoryPool` is created, used for an entire training run (many `acquire`/`release` cycles across several distinct tensor sizes), and then goes out of scope. Trace what happens to (a) the `Dict` and `List` bookkeeping structures inside `_free_lists`, and (b) the actual GPU buffers those structures were holding pointers to. Are both reclaimed?
5. The comment in Section 11.4's `training_step` example claims Mojo's lifetime checker "rejects any attempt" to read `activations` again after `pool.release(activations, ...)`. Explain, in terms of what kind of type `UnsafePointer` is, why that specific claim doesn't actually hold — and what *would* have to change about `acquire`/`release`'s signatures for a claim like that to become true.

## Where We Go Next

Chapter 12 (`part2/01-element-wise-operations.md`) is the first chapter of Part 2, and builds the first real tensor operations directly on top of the memory story this chapter completed: single ownership from Chapter 6 for the common case, `RefCountedBuffer` from Section 11.1 wherever an operation's result needs to alias rather than copy, and the `Arena`/`GPUMemoryPool` machinery from Sections 11.2–11.3 wherever an operation's intermediate results are cheap to churn through and expensive to allocate one at a time.

## Worked Solutions

**1.** The refcount starts at `1` when `a` is constructed, becomes `2` when `b = a` runs `__copyinit__`, and becomes `3` when `c = b` runs it again. It needs to count down from `3` to `0` — through `2`, then `1`, then `0` — for the buffer to free, and each step happens whenever any one of the three currently-live copies has its `__del__` run, regardless of which variable that copy is bound to. It does *not* matter which of `a`, `b`, or `c` is the one that happens to go out of scope last — the shared counter, not the specific variable, is what determines when the buffer is actually freed; whichever copy's destructor sees the count reach `0` is correctly the one that calls `.free()`.

**2.** `(200 + 63) & ~63 = 263 & ~63`. `263 = 4×64 + 7`, so clearing the low 6 bits removes the `7` and leaves `256`. The aligned offset is `256`, and the padding skipped is `256 - 200 = 56` bytes — within the `0`-to-`63`-byte range Worked Example 11.2.2 established as the maximum possible waste per allocation.

**3.** `alloc` computes `aligned_offset` and `ptr` exactly as it would in debug mode — the `debug_assert` line is compiled out entirely, not replaced with any other check, so nothing stops execution. `ptr` is computed as `(self._base + aligned_offset).bitcast[T]()`, a pointer that lands at an offset past the actual end of the `capacity_bytes`-sized allocation `self._base` points to. `self._offset` is updated to a value past `self._capacity` as well. The returned pointer is not garbage in the sense of being uninitialized or null — it's a validly computed address, just one that lands outside memory this `Arena` was ever granted, meaning every subsequent read or write through it touches memory belonging to something else entirely, with nothing in this code path ever detecting it.

**4.** The `Dict` and `List` bookkeeping structures are reclaimed: Mojo's synthesized default destructor for `GPUMemoryPool` tears down `self._free_lists` using `Dict`'s own destructor, which in turn frees each `List`'s internal backing array — the storage that held the `UnsafePointer` *values* (addresses) themselves. The actual GPU buffers those addresses pointed to are not reclaimed: `UnsafePointer` has no destructor that calls `.free()` on itself, so tearing down the list of addresses simply discards the addresses — it never issues the device-memory-freeing call those addresses would need for the underlying allocations to actually be released back to the driver. Only (a) happens; (b) is the leak traced in Section 11.3's `[COMMON TRAP]`.

**5.** `UnsafePointer` is a trivially copyable, unmanaged pointer type, exempted by design from the ownership and lifetime tracking Mojo applies to types like `List` or a custom RAII struct — that exemption is what the "Unsafe" in its name is signaling. `release(mut self, ptr: UnsafePointer[...], count: Int)` takes `ptr` as an ordinary (non-`owned`) parameter, meaning the call passes a *copy* of the address, not a value the caller gives up — nothing about calling `pool.release(activations, ...)` invalidates or consumes `activations` at the type-system level, so a later line reading `activations` again compiles and runs exactly as it would before the release call, whether or not doing so is a good idea. For the comment's claim to become literally true, `acquire`/`release` would need to work with a type that Mojo's ownership tracker actually understands as being consumed on release — for instance, wrapping the raw pointer in an `owned`-parameter-taking struct (much like this chapter's own `RefCountedBuffer` or `Arena` wrap `UnsafePointer` in a type whose `__del__` enforces the safety directly) rather than passing a bare `UnsafePointer` around, which by itself carries no such enforcement.
