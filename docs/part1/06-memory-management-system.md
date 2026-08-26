# Chapter 2: Memory Management System

Memory management is an ownership design problem before it is an allocator problem. Mojo 1.0 provides safe owning pointers, reference-counted pointers, layout-aware allocations, and runtime-owned device buffers. Choose the highest-level owner that fits the lifetime, and expose raw pointers only inside a narrow implementation boundary.

## 2.1 Shared views use `ArcPointer`

Tensor views need shared lifetime, not duplicated storage. `ArcPointer` owns one heap value, increments its atomic count when copied, and destroys the value when the last owner disappears.

```mojo
from std.memory import ArcPointer

struct SharedStorage(ImplicitlyCopyable):
    var data: ArcPointer[List[Float32]]

    def __init__(out self, var values: List[Float32]):
        self.data = ArcPointer(values^)

    def __init__(out self, *, copy: Self):
        self.data = copy.data
```

**Manual worked example.** Create storage for `[10,20,30]`, then copy the `SharedStorage` into two views. There is still one three-element list, now reached by three `ArcPointer` owners. Destroying one view decrements the count but preserves the list; destroying the final owner releases it.

## 2.2 Arenas serve short, batch lifetimes

An arena is valuable when many graph nodes die together. It allocates from chunks and resets a cursor in constant time, but values with destructors still need correct destruction. Use it for trivially destructible metadata or track destructor work explicitly.

```mojo
@fieldwise_init
struct ArenaCursor(Copyable, Movable):
    var capacity: Int
    var used: Int

    def reserve(mut self, count: Int) raises -> Int:
        if count < 0 or self.used + count > self.capacity:
            raise Error("arena exhausted")
        var start = self.used
        self.used += count
        return start

    def reset(mut self):
        self.used = 0
```

**Manual worked example.** With capacity 16, reservations of 5 and 3 return offsets 0 and 5 and leave `used=8`. Reset changes only the cursor to 0, so the next reservation reuses offset 0. It must happen only after every consumer of the old eight slots has finished.

## 2.3 GPU buffers belong to their context

Use `DeviceContext.enqueue_create_buffer` rather than allocating host memory and relabeling its pointer. The context knows the accelerator, address space, stream, and destruction path.

```mojo
from std.gpu.host import DeviceContext

def allocate_grad_buffer(ctx: DeviceContext, elements: Int):
    return ctx.enqueue_create_buffer[DType.float32](elements)
```

**Manual worked example.** A gradient tensor with 1,024 Float32 elements requests 4,096 payload bytes on the selected device. The returned owner is queued on the context; a host read requires a device-to-host copy and synchronization, not direct pointer indexing.

## 2.4 Raw allocation is the narrow fallback

When a custom data structure truly needs uninitialized contiguous storage, Mojo 1.0's layout-aware allocator keeps size and alignment attached to the allocation. Initialize each element before reading and deallocate the owning handle exactly once.

```mojo
from std.memory.alloc import alloc, dealloc, Layout

def raw_four_ints():
    var allocation = alloc(Layout[Int32](count=4))
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        (ptr + i).init_pointee_move(Int32(i * 10))
    print(ptr[0], ptr[1], ptr[2], ptr[3])
    dealloc(allocation^)
```

**Manual worked example.** The four initialized values are 0, 10, 20, and 30, so the print is deterministic. The allocation owns 16 payload bytes plus alignment metadata. Transferring it with `^` into `dealloc` prevents a second deallocation through the same owner.

The resulting policy is simple: `List` for ordinary owned tensor data, `ArcPointer` for shared views, arenas for graph-wide batch lifetimes, `DeviceBuffer` for accelerator memory, and raw layout allocations only for measured low-level needs.
