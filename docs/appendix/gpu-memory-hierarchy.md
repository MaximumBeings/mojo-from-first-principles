# Appendix C: GPU Memory Spaces and Ownership at the Hardware Boundary

> "Chapter 4 named three memory scopes and Chapter 18 built a real kernel in exactly one of them — shared memory, `stack_allocation[..., address_space = AddressSpace.SHARED]()` — because that was the one Chapter 18's convolution actually needed. This appendix goes back for the other two, and for the ownership question Chapter 18 quietly sidestepped every time it wrote `owned ctx: DeviceContext` without ever asking what happens if you don't."

**What you will understand by the end of this appendix:**

- Mojo's `AddressSpace` enum names four memory spaces, not one — `GLOBAL`, `SHARED`, `CONSTANT`, and `LOCAL` — and this book has so far only ever written code in `SHARED`; this appendix names what the other three are for and shows the syntax `SHARED` already established extended to `CONSTANT`
- Why this appendix, alone among this book series' three memory-architecture appendices, has the *least* direct evidence to offer for register pressure and local-memory spilling — and why that gap is itself worth stating plainly rather than papered over
- A full grid → block → warp → thread hierarchy diagram for Mojo's own launch syntax, with a new ceiling-division worked example that doesn't reuse any number Chapter 18 or its Self-Check already spent
- Why `DeviceContext` is neither Chapter 6's single-fixed-owner `Tensor` nor Chapter 11's many-owners `RefCountedBuffer`, but a third ownership shape this book has been using since Chapter 18 without ever naming it: exactly one owner, move-only, no `__copyinit__` at all

**What you need to know first:**

- Chapter 4 in full: the thread hierarchy, the three memory scopes named in the abstract, and memory coalescing.
- Chapter 18.1 and 18.3: the ceiling-division launch pattern, and the one Mojo GPU memory space this book has written actual code against — `stack_allocation[..., address_space = AddressSpace.SHARED]()`.
- Chapter 11 in full: `RefCountedBuffer`'s shared-ownership refcounting, `Arena`'s bump-pointer allocation and 64-byte alignment arithmetic, and `GPUMemoryPool`'s size-classed free list — this appendix's Section C.4 is a direct extension of the ownership question Chapter 11.1 opens with.
- Chapter 2.4 (constructors, destructors, `owned`, and the `^` transfer sigil) — Section C.4 is built entirely out of vocabulary from this section.

## C.1 The `AddressSpace` Hierarchy: What Chapter 18 Showed, and What It Left Out `[FOUNDATIONAL]`

### Intuition

A library has exactly four kinds of shelving, even though most patrons only ever interact with two of them. The open stacks (anyone can walk up and take any book) are what every kernel argument in this book has used without ever saying so — Chapter 4 and Chapter 18's `output`, `input`, `rate`, `spread` parameters are all ordinary `UnsafePointer` values, and every one of them is, implicitly, a pointer into the GPU's large, off-chip global memory: the open stacks. The reserve shelf behind the front desk, reachable only by the one reading table nearest it (Chapter 18.3's `stack_allocation[..., address_space = AddressSpace.SHARED]()`), is the second kind this book has actually shown code for. The other two — a locked reference case holding one fixed, read-only set of volumes everyone at every table consults but nobody checks out (`CONSTANT`), and the stack of books a single reader has open on their own desk at that exact moment (`LOCAL`) — exist in the same library and are named in Mojo's own `AddressSpace` enum, but this book has never written a line of code that asks for either of them by name.

### Background

Mojo's `memory` module defines `AddressSpace` as an enum with (at least) four members this book's own established syntax already implies the existence of, even though only one has appeared in actual code before this appendix:

| `AddressSpace` member | What it names | Established in this book? |
|---|---|---|
| `GLOBAL` | The GPU's large, off-chip DRAM — every ordinary kernel `UnsafePointer` argument in Chapters 4 and 18 lives here, implicitly, without ever spelling `address_space = AddressSpace.GLOBAL` | Used implicitly since Chapter 4; never spelled out explicitly until this appendix |
| `SHARED` | Fast, on-chip memory scoped to one block, allocated with `stack_allocation[SIZE, DType, address_space = AddressSpace.SHARED]()` | Chapter 18.3, the convolution tiling kernel — the only `AddressSpace` variant this book has used in real code before now |
| `CONSTANT` | Small, read-only-from-the-device memory backed by a dedicated broadcast-friendly cache — this book's CUDA-book sibling's Appendix C.4 covers the identical hardware concept for `__constant__` arrays | Not used anywhere before this appendix |
| `LOCAL` | Per-thread memory the compiler falls back to when a value that *should* live in a register doesn't fit — Section C.2 below | Not used anywhere before this appendix |

The syntax for allocating `CONSTANT` memory follows the exact same `stack_allocation` call Chapter 18.3 already established for `SHARED`, with only the `address_space` parameter changed:

```mojo
from memory import UnsafePointer, AddressSpace, stack_allocation

alias NUM_COEFFS = 8

fn constant_broadcast_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    var coeffs = stack_allocation[NUM_COEFFS, DType.float32, address_space = AddressSpace.CONSTANT]()
    # In a real launch, `coeffs` would be populated once, before the kernel
    # runs, from a value known at launch time -- the Mojo-level analogue of
    # cudaMemcpyToSymbol into a __constant__ array. Every thread below reads
    # coeffs[0], the broadcast case this same book's CUDA sibling's
    # Appendix C.4 covers on the same underlying hardware.
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * coeffs[0]
```

**[COMMON TRAP]** This snippet is written in this book's own established idiom, exactly extending Chapter 18.3's syntax — but, exactly like Chapter 18.3 and 18.4's own code, it has not been compiled. Unlike those two sections, which reused syntax and worked-example values the chapter's own text (or its author's independently-checked reference) had already exercised, `AddressSpace.CONSTANT` is new territory this book has not written before. This appendix treats it with the same honesty Chapter 18.5 uses for its own material: presented as source, not as a captured session, and flagged as such rather than left to look verified by proximity to code that was.

### The full hierarchy, restated as one diagram

```
HOST (CPU)
  |
  |  DeviceContext.enqueue_function[kernel](..., grid_dim=(...), block_dim=(...))
  v
DEVICE
  |
  |-- GLOBAL   (AddressSpace.GLOBAL)  -- large, off-chip, every ordinary
  |             UnsafePointer kernel argument since Chapter 4 lives here
  |
  |-- CONSTANT (AddressSpace.CONSTANT) -- small, read-only, broadcast-cached;
  |             new to this appendix, same silicon this book's CUDA
  |             sibling's Appendix C.4 names directly
  |
  |-- one BLOCK  (a group of threads sharing one on-chip allocation)
  |     |-- SHARED (AddressSpace.SHARED) -- Chapter 18.3's stack_allocation;
  |     |           visible to every thread in this block, gone once the
  |     |           block finishes
  |     |-- one THREAD
  |           |-- registers  (no Mojo keyword at all -- Section C.2)
  |           '-- LOCAL (AddressSpace.LOCAL) -- per-thread fallback when a
  |                       value doesn't fit in registers -- Section C.2
  '-- ... (more blocks, each with its own SHARED allocation)
```

Nothing above changes what any earlier chapter's code does. `GLOBAL` was always the address space Chapter 4's `output[idx] = input[idx] * 2.0` was reading and writing — the only thing new here is finally naming it.

## C.2 Local Memory and the Register Pressure Question This Book Can't Answer From Here `[FOUNDATIONAL]`

### Intuition

Three chess players are each asked to play a game entirely in their head, no board. One has a small enough position that every piece fits in memory with room to spare — nothing written down. A second starts running out of working memory partway through, and begins jotting a few piece positions on a scrap of paper next to them — slower to consult than pure memory, but still fast, still at their elbow. A third has no memory at all for this purpose and has to write down the *entire* position, move by move, on a notepad across the room. All three "have registers" in the sense that this book's other two sibling appendices use that word (CUDA's real per-thread register file; Triton's compiler-implicit tile allocation), but only the first book can actually watch a real compiler decide, for a real kernel, which of these three players it's building.

### Background

This is the one place in this entire three-book series where the honest answer is a gap this book cannot close, and it is worth being direct about exactly why, rather than presenting a hand-traced guess as though it carried the same weight as measured evidence:

| Book | Register pressure evidence available |
|---|---|
| CUDA (this series' Appendix C, Section C.2) | `nvcc -Xptxas -v` genuinely compiled the same kernel twice, once unconstrained and once with `-maxrregcount`, and reported real, measured spill byte counts from the actual compiler backend |
| Triton (this series' Appendix D, Section D.1/D.4) | No spill counter exists, but every kernel genuinely executes under `TRITON_INTERPRET=1` — real numeric output, even without register-level visibility |
| Mojo (this appendix) | No Mojo compiler exists in this environment at all — not "a compiler exists but reports nothing," the way Triton's interpreter has no vectorizing backend, but no compiler present, period, matching this entire book's own established methodology since Chapter 11.5 and 18.5 |

Mojo, like Triton, exposes no `register` keyword and no `AddressSpace.REGISTER` — ordinary local variables inside a kernel are the only spelling for "a per-thread value," and whether any given one lives in a real hardware register or spills to `AddressSpace.LOCAL` memory is a decision Mojo's compiler backend makes, invisibly, the same way Triton's PTX-generating backend does for a `tl.constexpr`-sized block. This book cannot observe that decision the way the CUDA appendix's `-Xptxas -v` genuinely can, because there is no Mojo compiler in this sandbox to ask.

What this section *can* do honestly is the arithmetic a compiler's decision would be operating on, hand-traced exactly the way the rest of this book verifies its claims:

### Worked Example C.2.1 — Sizing a per-thread array against Arena's own alignment unit

Chapter 11.2's `Arena` rounds every allocation up to a 64-byte boundary: `(offset + 63) & ~63`. A per-thread local array of `64` `Float32` values — deliberately the same element count as this book series' CUDA sibling's own register-pressure worked example, so the two appendices' numbers can be compared directly — is `64 × 4 = 256` bytes. `256` is itself an exact multiple of `64`, so this particular size needs no alignment padding at all: a hypothetical `AddressSpace.LOCAL` allocation of this exact size would sit at a clean 4-cache-line boundary with zero waste, the same "no padding needed" case Chapter 11.2's own Self-Check already established for its `200 → 256`-byte example rounding up, not down. A smaller array of `40` `Float32` values is `160` bytes; rounding up to the next 64-byte multiple gives `192` bytes, `32` bytes of padding — comfortably inside the "at most 63 bytes" bound Chapter 11.2's own text states for `Arena`'s worst case, restated here on a genuinely different size to confirm the bound holds beyond the one example the chapter itself worked through.

None of this tells you whether Mojo's real compiler would actually spill either array to `AddressSpace.LOCAL` memory or keep it entirely in registers — that depends on register pressure from the rest of the kernel, occupancy targets, and compiler heuristics this book has no way to observe. What it does tell you is the *cost* of spilling, if it happens: a spilled 64-element array costs the same 256 bytes of traffic per thread regardless of which book's compiler is doing the spilling, because that number comes from `sizeof[Float32]() × 64`, not from anything specific to CUDA, Triton, or Mojo.

**[COMMON TRAP]** It's tempting to read "no compiler available" as "therefore assume Mojo doesn't spill." That does not follow, and this book makes no such claim. GPU compilers for every one of these three languages target the same category of underlying hardware, with the same finite register file per thread; a large enough per-thread array will spill somewhere regardless of source language. The honest position, and the one this section takes, is narrower: this book cannot show you a *measured* number the way its CUDA sibling can, so it doesn't claim one.

## C.3 The Execution Hierarchy, Ceiling Division, and a Full Grid → Block → Warp → Thread Diagram `[FOUNDATIONAL]`

### Intuition

Chapter 18.1's moving-crew analogy — fixed-size trucks, a job that doesn't divide evenly, an idle mover on the last truck rather than a dropped box — is restated here for one reason: to place it inside the *complete* chain this book's CUDA sibling names explicitly with the word "CTA" (Cooperative Thread Array) and this book has always called, correctly but only ever partially, a "block." A block is itself made of warps — fixed groups of `32` threads the hardware schedules and executes in lockstep — and Chapter 18.4's `WARP_SIZE = 32` constant and its `shuffle_down`-based reduction are, mechanically, warp-scoped operations this book has already relied on without ever drawing the full picture connecting "block" down to "warp" down to "thread."

### Background

### Worked Example C.3.1 — A ceiling-division launch this book hasn't computed before

Chapter 18.1 traces `size = 1,000,000, THREADS_PER_BLOCK = 256`; its Self-Check traces `size = 5,000, THREADS_PER_BLOCK = 512`. Neither has touched `size = 50,000, THREADS_PER_BLOCK = 256`, computed fresh here with the identical formula:

`num_blocks = (50,000 + 255) // 256 = 50,255 // 256 = 196`. Total threads launched: `196 × 256 = 50,176`. Wasted: `50,176 − 50,000 = 176`. The last block is block `195`, covering global indices `195 × 256 = 49,920` through `50,175`; valid indices are `49,920` through `49,999` — `80` active threads — leaving `256 − 80 = 176` idle, matching the wasted count computed from the totals, exactly the cross-check Chapter 18's own Worked Solutions perform.

### Worked Example C.3.2 — The same launch, one warp lower

Every block above is itself `256 / 32 = 8` warps. The block containing the boundary between valid and wasted work — block `195` — has `80` active threads. `80 / 32 = 2.5`, so those `80` active threads occupy `3` full warps (`96` thread-slots), of which the third warp has only `80 − 64 = 16` threads doing real work and `16` idle — the same wasted-lane phenomenon this book series' CUDA sibling's Appendix C.6 computes directly (`warps_per_block()`), now shown to be a *second*, finer-grained instance of the exact same rounding problem Chapter 18.1's block-level ceiling division already solves at a coarser grain. A block can be "correctly sized" at the block level (block `195` is a completely ordinary block, no different from block `0`) and still contain warps that are only partially active, because block-level padding (Chapter 18.1's bounds check) and warp-level occupancy are two independent rounding problems stacked on top of each other, not the same problem solved twice.

```
GRID  (ctx.enqueue_function[kernel](..., grid_dim=(196, 1, 1), block_dim=(256, 1, 1)))
  |
  |-- BLOCK 0            (256 threads, 8 warps, all fully active)
  |     |-- WARP 0  (threads 0-31)
  |     |-- WARP 1  (threads 32-63)
  |     '-- ... (6 more warps)
  |
  |-- ... (blocks 1 through 194, identical to block 0)
  |
  '-- BLOCK 195          (256 threads launched; only 80 pass `if idx < size`)
        |-- WARP 0  (threads 0-31,  global idx 49,920-49,951  -- fully active)
        |-- WARP 1  (threads 32-63, global idx 49,952-49,983  -- fully active)
        '-- WARP 2  (threads 64-95, global idx 49,984-50,015  -- only 16 of 32
                      active; the other 16 fail `idx < size` and idle)
```

Warps 3 through 7 of block 195 are, in this trace, *entirely* idle — every one of their 32 threads fails the bounds check, since the last valid global index (`49,999`) falls inside warp 2's range. This is worth naming because it means Chapter 18.4's `warp_reduce_sum` — which, per that chapter's own Self-Check Question 5 and Worked Solution 5, requires every lane in a warp to still be executing when `shuffle_down` runs — would be entirely safe to call inside warps 0 and 1 of block 195 without modification, but would need the identity-value substitution Chapter 18's own Worked Solution 5 describes if called inside warp 2, and would simply have nothing to reduce in warps 3 through 7. The bounds-check-versus-shuffle interaction Chapter 18 flags as a hazard in the abstract has a concrete, fully worked address range attached to it here.

## C.4 Ownership at the Host/Device Boundary: `owned`, `^`, and Why a `DeviceContext` Can't Be Copied `[FOUNDATIONAL]`

### Intuition

Chapter 11.1 opens with two roommates who can both hold a key to the same apartment, because the landlord only cares that all keys are eventually returned — that's `RefCountedBuffer`, any number of owners, whichever one finishes last does the freeing. Chapter 6 and Chapter 10's `Tensor`/`MemoryBlock` are a stricter version of the same idea: exactly one legitimate owner, established once and never renegotiated. `DeviceContext`, which every GPU-launching function since Chapter 18.1 has taken as `owned ctx: DeviceContext` and stored via `self.ctx = ctx^`, is neither of those. It's closer to a single house key that can be *handed off* — from one hand to another, so that at every instant exactly one person holds it — but never *duplicated*, because the thing it represents (a live, open connection to one physical GPU) genuinely cannot be split into two independently valid copies without one of them being a lie about what it actually controls.

### Background

Chapter 18's launch functions establish the pattern without ever stopping to name the category it belongs to:

```mojo
struct GPUKernelLauncher:
    var ctx: DeviceContext

    fn __init__(out self, owned ctx: DeviceContext):
        self.ctx = ctx^   # moved in, not copied -- the caller's `ctx` is
                           # no longer valid to use after this call

    fn launch_elementwise(self, output: UnsafePointer[Scalar[DType.float32]],
                           input: UnsafePointer[Scalar[DType.float32]], size: Int) raises:
        alias THREADS_PER_BLOCK = 256
        var num_blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self.ctx.enqueue_function[generic_elementwise_kernel](
            output, input, size,
            grid_dim=(num_blocks, 1, 1),
            block_dim=(THREADS_PER_BLOCK, 1, 1),
        )
        self.ctx.synchronize()
```

`GPUKernelLauncher` defines no `__copyinit__` at all — not Chapter 6's `_owns_data`-flag kind, not Chapter 11.1's refcount-incrementing kind, none. That absence is the entire design: without a `__copyinit__`, Mojo simply does not allow `var launcher2 = launcher1` to compile — there is no copy constructor for it to call. The *only* way a `DeviceContext` (or anything wrapping one) moves from one place to another is the `^` transfer sigil, which Chapter 2.4 establishes and Chapter 18's own launch functions already use every time they write `ctx^`. This is the pattern's whole point: a `DeviceContext` genuinely represents one live handle onto one physical GPU's driver-level context, and letting two struct instances both believe they independently own that same handle would leave two code paths capable of calling `ctx.synchronize()` or issuing kernel launches against a resource only one of them actually still controls — not a bookkeeping inconvenience the way an extra refcount increment is, but a real double-ownership bug against a resource that has no safe way to be shared the `RefCountedBuffer` way.

### Worked Example C.4.1 — Why adding `__copyinit__` here would be actively wrong, traced against Chapter 11.1's own reasoning

Chapter 11.1 justifies `RefCountedBuffer`'s design with a specific claim: sharing is safe there because "whichever copy's destructor sees the count hit zero" is the one that frees the memory, and every copy up to that point is an equally valid way to read or write the shared buffer. Suppose `GPUKernelLauncher` were given a `__copyinit__` that shallow-copied `self.ctx` the naive way (`self.ctx = existing.ctx`, if such a copy were even permitted for `DeviceContext`, which it is not, precisely to prevent this): both the original and the copy would now believe they hold a valid, independent `DeviceContext`. Nothing analogous to `RefCountedBuffer`'s shared `_refcount` integer exists to coordinate between them — there is no shared counter, so neither instance has any way to know the other is still using the same physical GPU connection. If the original's `__del__` ran first and freed or closed the underlying context (the GPU-driver equivalent of `Arena`'s `reset()`, but irreversible rather than reusable), the copy would be left holding a handle to a context that no longer exists — a use-after-free at the driver level, not the buffer level, and one `RefCountedBuffer`'s entire mechanism was specifically built to prevent for ordinary heap memory. The absence of `__copyinit__` here isn't a missing feature; it's the one design Chapter 11.1's own reasoning, applied honestly to a resource with no shared-counter equivalent, rules back in as the only safe option.

**[COMMON TRAP]** It's tempting to read "no `__copyinit__`" as "this struct forgot Chapter 11's lesson about resource safety." Chapter 11.4 closes by admitting its own `GPUMemoryPool` has exactly this kind of gap — a destructor it never got, leaking every buffer it hands out. `GPUKernelLauncher`'s missing `__copyinit__` is the opposite situation: not an oversight, but a deliberate absence, because for this specific resource, the correct number of legitimate "owners at once" the way Chapter 11.1's comparison table frames it is not "exactly one" (Chapter 6), not "any number, tracked live" (`RefCountedBuffer`), but "exactly one, and never renegotiated except by an explicit, visible `^` move that invalidates the source" — a fourth ownership shape this book's own Chapter 11.1 table didn't yet have a column for.

## C.5 Reference Implementations

Consistent with this book's own established methodology since Chapter 11.5 and Chapter 18.5, none of this appendix's Mojo has been compiled or run — it is presented as source, hand-traced against the arithmetic and reasoning in the prose above, not as a captured session. `AddressSpace.CONSTANT` (Section C.1) and the `GPUKernelLauncher` ownership pattern (Section C.4) extend past what Chapters 11 and 18 individually verified; Section C.1's own `[COMMON TRAP]` and Section C.4's own prose say so explicitly rather than letting proximity to earlier, more established code imply a verification this appendix cannot actually provide. What follows is every snippet this appendix derived, assembled in one place:

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu.warp import shuffle_down
from memory import UnsafePointer, AddressSpace, stack_allocation

# ---- C.1: constant memory, extending Chapter 18.3's stack_allocation syntax ----

alias NUM_COEFFS = 8

fn constant_broadcast_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    var coeffs = stack_allocation[NUM_COEFFS, DType.float32, address_space = AddressSpace.CONSTANT]()
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * coeffs[0]


# ---- C.3: ceiling-division launch, size=50,000 worked example ----

fn generic_elementwise_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        output[idx] = input[idx] * Float32(2.0)


# ---- C.4: DeviceContext ownership, move-only, no __copyinit__ ----

struct GPUKernelLauncher:
    var ctx: DeviceContext

    fn __init__(out self, owned ctx: DeviceContext):
        self.ctx = ctx^

    fn launch_elementwise(self, output: UnsafePointer[Scalar[DType.float32]],
                           input: UnsafePointer[Scalar[DType.float32]], size: Int) raises:
        alias THREADS_PER_BLOCK = 256
        var num_blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self.ctx.enqueue_function[generic_elementwise_kernel](
            output, input, size,
            grid_dim=(num_blocks, 1, 1),
            block_dim=(THREADS_PER_BLOCK, 1, 1),
        )
        self.ctx.synchronize()
```

### Expected Output

There is no captured run to reproduce here, for the same reason Chapter 11.5 and Chapter 18.5 have none: this appendix's Worked Examples (C.2.1, C.3.1, C.3.2, C.4.1) are the source of truth for every number above, each hand-traced against the arithmetic in the prose, not against a compiled session.

## Chapter Summary

Mojo's `AddressSpace` enum names four memory spaces — `GLOBAL`, `SHARED`, `CONSTANT`, `LOCAL` — of which this book had, before this appendix, only ever written real code against one (`SHARED`, Chapter 18.3); `GLOBAL` was always the implicit space behind every ordinary kernel `UnsafePointer` since Chapter 4, and `CONSTANT`'s `stack_allocation` syntax extends Chapter 18.3's own pattern directly, flagged explicitly as new, unverified-by-compilation territory rather than presented as equally established. Register pressure and local-memory spilling (`AddressSpace.LOCAL`) are the one area where this book's own no-compiler constraint is a genuine, stated limit rather than a stylistic choice — unlike its CUDA sibling's measured `-Xptxas -v` spill counts or its Triton sibling's genuinely interpreter-executed kernels, this book can only hand-trace the *cost* of a hypothetical spill (`64` `Float32` = `256` bytes, cleanly `Arena`-aligned; `40` `Float32` = `160` bytes, padded to `192`), not observe whether any real compiler backend would actually produce one. The grid → block → warp → thread hierarchy is one chain, not two independent rounding problems solved once each: a `50,000`-element, `256`-thread-per-block launch needs `196` blocks and wastes `176` threads at the block level exactly as Chapter 18.1's formula predicts, and that same launch's boundary block additionally splits unevenly at the warp level, leaving one 32-thread warp only half active and five entirely idle — a second, finer rounding effect this book's CUDA sibling names directly as CTA/warp granularity and this chapter shows arising from the exact same launch Chapter 18.1 already knew how to size. Finally, `DeviceContext`'s complete absence of a `__copyinit__` is neither an oversight nor a stricter version of Chapter 6's single-owner `Tensor` — applying Chapter 11.1's own "how many owners at once" framework honestly to a live GPU driver handle with no shared-counter equivalent to `RefCountedBuffer`'s shows that move-only, `^`-only ownership is the *only* safe design for this specific resource, a fourth ownership shape alongside the three Chapter 11.1's own comparison table already named.

## Self-Check Questions

1. Using `stack_allocation[..., address_space = AddressSpace.CONSTANT]()` from Section C.1, write the one-line change to `constant_broadcast_kernel` that would make every thread read a *different* `coeffs` entry instead of broadcasting `coeffs[0]` — matching the "varying" contrast this book series' CUDA sibling's Appendix C.4 draws between its own two constant-memory kernels.
2. A per-thread local array holds `20` `Float64` values (`8` bytes each, not `Float32`'s `4`). Using Section C.2's method, compute its size in bytes and how many bytes of `Arena`-style 64-byte alignment padding it would need.
3. For `size = 50,000, THREADS_PER_BLOCK = 256` (Worked Example C.3.1), a second, independent kernel launch uses `THREADS_PER_BLOCK = 128` instead. Compute `num_blocks`, total threads launched, and wasted threads for this second configuration.
4. Section C.4 argues `GPUKernelLauncher` should never get a `__copyinit__`. Chapter 11.1's `RefCountedBuffer` argues the opposite for its own buffer. Name the one structural difference between the two structs that makes both arguments correct simultaneously, rather than contradictory.
5. Worked Example C.3.2 found that block 195's warp 2 has `16` active threads and `16` idle ones. Using the same `size = 50,000, THREADS_PER_BLOCK = 256` launch, which warp index (0 through 7) within block 195 is the *last* warp to contain at least one active thread, and how many of its 32 threads are active?

## Where We Go Next

This appendix closes this book's coverage of memory the way its CUDA and Triton siblings' own final appendices close theirs: not with new operations this book's chapters will build on going forward, but with the vocabulary — `AddressSpace`'s full four-member enum, the grid/warp hierarchy underneath every launch this book has ever written, and `DeviceContext`'s move-only ownership shape — needed to read GPU-facing Mojo code this book didn't happen to need for any of its own worked examples, and to recognize which of this appendix's own claims rest on hand-traced arithmetic this book can fully verify versus which rest on Mojo API surface this environment has no compiler to check.

## Worked Solutions

**1.** Replace `coeffs[0]` with `coeffs[idx % NUM_COEFFS]`: `output[idx] = input[idx] * coeffs[idx % NUM_COEFFS]`. Every thread now reads a different entry (wrapping every `8` threads, since `NUM_COEFFS = 8`), the same broadcast-versus-varying contrast the CUDA appendix draws between `constant_broadcast_kernel` and `constant_varying_kernel`.

**2.** `20 × 8 = 160` bytes. Rounding up to the next 64-byte multiple: `ceil(160/64) × 64 = 3 × 64 = 192` bytes, so `192 − 160 = 32` bytes of padding — coincidentally the identical byte count Section C.2's own `40`-`Float32` example produces (`160` bytes there too, by a different route: `40 × 4 = 160`), confirming the padding amount depends only on the total byte size crossing a 64-byte boundary, not on the element type or count that produced it.

**3.** `num_blocks = (50,000 + 127) // 128 = 50,127 // 128 = 391`. Total threads launched: `391 × 128 = 50,048`. Wasted: `50,048 − 50,000 = 48`. (Cross-check: the last block, block `390`, starts at global index `390 × 128 = 49,920`; valid indices run through `49,999`, so `80` threads are active and `128 − 80 = 48` are idle, matching the wasted count computed from the totals.)

**4.** `RefCountedBuffer` carries its own shared coordination mechanism — the `_refcount` integer every copy points at and can safely increment or decrement — so multiple simultaneous owners can always agree, through that shared counter, on when the last one has gone out of scope. `DeviceContext` (and anything wrapping it, like `GPUKernelLauncher`) has no equivalent shared counter, and the resource it represents — a live driver-level connection to one physical GPU — cannot be safely split the way heap memory can, because there is no way to "partially" hold a GPU context the way two `RefCountedBuffer` copies can both validly read the same buffer. The structural difference is exactly that shared-counter mechanism's presence or absence: `RefCountedBuffer` has a real, working answer to "how do multiple owners safely coordinate," and `GPUKernelLauncher` has none, which is precisely why allowing multiple owners is safe for one and unsafe for the other.

**5.** Block 195's valid global indices are `49,920` through `49,999` — `80` values. Warp boundaries within the block fall at local thread indices `0–31` (warp 0), `32–63` (warp 1), `64–95` (warp 2), `96–127` (warp 3), and so on. Local index `79` (the 80th active thread, `idx = 49,999`) falls in the range `64–95`, which is warp 2 — so warp 2 is the last warp with any active thread, matching Worked Example C.3.2's own finding, and it has `80 − 64 = 16` active threads out of its `32`. Warps 3 through 7 (local indices `96` and above) contain no valid indices at all and are entirely idle.
