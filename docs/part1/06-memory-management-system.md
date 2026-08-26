# Chapter 2: Memory Management System

Chapter 1 gave every tensor its own `UnsafePointer` allocation, freed deterministically in `__del__`. That's correct for a single owner, but the autograd engine we start building in Part 3 needs the *same* underlying buffer visible from multiple places at once — a tensor and the view a slicing operation returned from it, or a tensor and the node in the computational graph that holds a reference to it for the backward pass. This chapter builds the reference-counting, pooled-allocation, and RAII machinery that makes shared ownership safe without giving up the manual-memory performance from Chapter 1.

## 2.1 Reference Counting Implementation

A naive shared pointer needs an atomic counter, a pointer to the payload, and `__copyinit__`/`__del__` hooks that increment and decrement it. Mojo's ownership model lets us express this as a small wrapper struct rather than reaching for a garbage collector:

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
        self._refcount[0] = 1
        self._data = UnsafePointer[T].alloc(count)

    fn __copyinit__(out self, existing: Self):
        # Shallow copy: share the buffer, bump the count.
        self._data = existing._data
        self._refcount = existing._refcount
        self.count = existing.count
        self._refcount[0] += 1

    fn __del__(owned self):
        self._refcount[0] -= 1
        if self._refcount[0] == 0:
            self._data.free()
            self._refcount.free()

    fn refcount(self) -> Int:
        return self._refcount[0]
```

The tensor view infrastructure from [1.2.2](02-memory-layout-design.md) is the direct consumer: a `TensorView` copies the parent's `RefCountedBuffer`, not its data, so `parent.refcount()` correctly reports 2 while any view is alive and the underlying allocation only frees once every view *and* the owning tensor have gone out of scope.

## 2.2 Arena-based Memory Allocation

Reference counting adds an atomic-like decrement on every tensor destructor, which is measurable overhead in a tight training loop that allocates thousands of small intermediate tensors per step. An arena sidesteps that: allocate one large slab up front, hand out bump-pointer offsets into it, and free the *entire* slab in one call at the end of a forward/backward pass.

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

Used per training step: `arena.reset()` at the top of `forward()`, allocate every intermediate activation from `arena.alloc[Float32](...)`, and never call an individual free until the arena itself is dropped at the end of training. This is the same allocation strategy production frameworks like PyTorch's caching allocator and JAX's donation-based buffer reuse converge on for the same reason — allocator churn, not compute, dominates small-tensor workloads.

## 2.3 GPU Memory Management

GPU allocations are one to three orders of magnitude more expensive to issue than a `malloc`, and a naive framework that calls `ctx.enqueue_function`-adjacent device allocation on every forward pass will spend more time in the driver than in kernels. The fix is a device-side free list keyed by size class, sitting on top of the `DeviceContext` abstraction introduced in [Part 0.4](../part0/04-gpu-programming-introduction.md):

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

In steady-state training, `hit_rate()` should climb toward 1.0 within the first few steps, since the same activation shapes recur every iteration.

## 2.4 RAII Patterns and Automatic Cleanup

Every allocator in this chapter follows the same rule Chapter 1's `Tensor` established: acquisition happens in `__init__`, release happens in `__del__`, and there is no code path that can leak a resource because Mojo's ownership tracker will not compile a use of a value after it has been consumed. Concretely, for a scoped resource like the GPU memory pool above, the idiomatic pattern is:

```mojo
fn training_step(mut pool: GPUMemoryPool, batch: Tensor):
    var activations = pool.acquire(batch.shape.size())
    # ... forward + backward pass using `activations` ...
    pool.release(activations, batch.shape.size())
    # `activations` is not read again after this line — Mojo's
    # lifetime checker rejects any attempt to do so.
```

Combined with `Arena` for the per-step scratch space and `RefCountedBuffer` for values that outlive a single step (the graph's saved tensors for backward), Part 1 now has a complete memory story: deterministic single ownership from Chapter 1, shared ownership where the graph needs it, bump allocation where per-step churn dominates, and a device-side pool where the allocation itself (not the memory) is the expensive resource. Part 2 builds the tensor operations on top of this foundation.
