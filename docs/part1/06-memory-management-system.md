# Chapter 2: Memory Management System

Chapter 1 gave every tensor its own `UnsafePointer` allocation, freed deterministically in `__del__` — correct for a single owner, but too rigid for what's coming. Part 3's computational graph needs the *same* underlying buffer visible from two places at once: a tensor and the graph node that holds a reference to it for the backward pass. This chapter builds the reference-counting, pooled-allocation, and RAII machinery that makes shared ownership safe without giving up the manual-memory performance Chapter 1 established.

## 2.1 Reference Counting Implementation

Picture three lines of code: `var t = Tensor.zeros([4])` creates a buffer with one owner. `var v = t.view()` (Section 1.2.2) is meant to look at the *same* memory, not a copy of it — so now there are two things pointing at one buffer. If `t` goes out of scope first, do you free the memory out from under `v`? If neither ever explicitly says "I'm done," does the memory leak forever? Reference counting answers both questions with one small integer: keep a shared counter alongside the buffer, increment it every time something starts sharing the buffer, decrement it every time something stops, and only free the memory when the counter reaches zero.

Trace it with actual counts. `t = Tensor.zeros([4])` allocates the buffer and sets the counter to `1`. `v = t.view()` copies the *pointer* (not the data) and bumps the counter to `2`. Suppose `v` then goes out of scope: the counter drops to `1`, but the buffer survives, because `t` still needs it. Only when `t` *also* goes out of scope does the counter reach `0`, and the buffer is actually freed. At no point in that sequence does either `t` or `v` need to know whether the other one is still alive — the counter carries that information for both of them.

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

The tensor view infrastructure from [1.2.2](02-memory-layout-design.md#part-122-view-and-slicing-infrastructure) is the direct consumer: a `TensorView` copies the parent's `RefCountedBuffer`, not its data, so `parent.refcount()` correctly reports `2` while any view is alive, exactly as traced above, and the underlying allocation only frees once every view *and* the owning tensor have gone out of scope.

## 2.2 Arena-based Memory Allocation

Reference counting adds a decrement-and-check on every single tensor destructor — cheap in isolation, but a training step in [Chapter 11](../part6/01-neural-network-layers.md) allocates *thousands* of small intermediate tensors (one `Z` and one `A` per layer, per batch), and that overhead adds up. An arena sidesteps it with a much simpler idea: allocate one large slab once, and hand out offsets into it with nothing more than addition — no bookkeeping per allocation, and no individual frees at all, just one bulk reset at the end.

Work through actual byte offsets. Say the arena is freshly reset (`_offset = 0`), and the first request is for `100` `Float32`s — `100 × 4 = 400` bytes. Before handing that memory out, the offset is rounded up to the nearest 64-byte boundary (`(0 + 63) & ~63 = 0`, already aligned), the caller gets a pointer at offset `0`, and `_offset` advances to `400`. The next request, for `50` more `Float32`s (`200` bytes), first re-aligns: `(400 + 63) & ~63 = 448` — so `48` bytes are quietly skipped to keep the next allocation on a 64-byte (cache-line and SIMD-friendly) boundary — and the new allocation runs from byte `448` to byte `648`, leaving `_offset = 648`. No allocation in this sequence was ever individually freed; when the training step finishes, one call to `reset()` sets `_offset` back to `0` and every one of those allocations is invalidated at once.

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

Used per training step: `arena.reset()` at the top of `forward()`, allocate every intermediate activation from `arena.alloc[Float32](...)` (exactly the `100`-then-`50`-element sequence traced above), and never call an individual free until the arena itself is dropped at the end of training. This is the same allocation strategy production frameworks like PyTorch's caching allocator and JAX's donation-based buffer reuse converge on for the same reason — allocator churn, not compute, dominates small-tensor workloads.

## 2.3 GPU Memory Management

GPU allocations are one to three orders of magnitude more expensive to issue than a CPU `malloc`, because each one round-trips through the device driver. A framework that naively allocates fresh device memory on every forward pass will spend more wall-clock time talking to the driver than running kernels — so instead, keep a pool of already-allocated buffers, grouped by size, and hand the *same* buffer back out the next time something asks for that exact size.

Trace three training steps by hand. Step 1 requests a `256`-element buffer: the pool's free list for size `256` is empty, so it allocates fresh (`bytes_allocated += 1024`) and hands it out. At the end of the step, the buffer is released back to the pool (not freed) — the free list for size `256` now holds one entry. Step 2 requests another `256`-element buffer: this time the free list isn't empty, so the pool pops that exact buffer and reuses it (`bytes_reused += 1024`) instead of asking the driver for new memory. Step 3, same story: another reuse. After three steps, `bytes_allocated = 1024` and `bytes_reused = 2048`, so `hit_rate() = 2048 / (1024+2048) = 0.667` — climbing toward `1.0` as training continues, since the same activation shapes recur every iteration.

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

## 2.4 RAII Patterns and Automatic Cleanup

Every allocator in this chapter follows the same rule Chapter 1's `Tensor` established: acquisition happens in `__init__`, release happens in `__del__`, and there is no code path that can leak a resource because Mojo's ownership tracker will not compile a use of a value after it has been consumed. Concretely, for the pool traced in Section 2.3:

```mojo
fn training_step(mut pool: GPUMemoryPool, batch: Tensor):
    var activations = pool.acquire(batch.shape.size())
    # ... forward + backward pass using `activations` ...
    pool.release(activations, batch.shape.size())
    # `activations` is not read again after this line -- Mojo's
    # lifetime checker rejects any attempt to do so.
```

Combined with `Arena` for the per-step scratch space and `RefCountedBuffer` for values that outlive a single step (the graph's saved tensors for backward), Part 1 now has a complete memory story: deterministic single ownership from Chapter 1, shared ownership where the graph needs it (traced numerically in 2.1), bump allocation where per-step churn dominates (traced numerically in 2.2), and a device-side pool where the allocation itself, not the memory, is the expensive resource (traced numerically in 2.3). Part 2 builds the tensor operations on top of this foundation.
