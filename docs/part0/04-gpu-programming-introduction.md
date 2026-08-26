# 0.4 GPU Programming Introduction

A GPU program has a host side and a device side. The host owns the context, buffers, copies, launch geometry, and synchronization; the kernel maps each device thread to work. Keeping those responsibilities separate makes failures diagnosable.

## 0.4.1 Global indices identify work

`global_idx.x` already combines block and local thread coordinates. Convert it to `Int` for indexing and guard the logical length.

```mojo
from std.gpu import global_idx

def increment_kernel(data: UnsafePointer[Float32], size: Int32):
    var i = Int(global_idx.x)
    if i < Int(size):
        data[i] += 1
```

**Manual worked example.** With five values and eight launched threads, indices 0–4 increment their elements. Indices 5–7 do nothing. Input `[0,1,2,3,4]` becomes `[1,2,3,4,5]` without an out-of-bounds access.

## 0.4.2 Device buffers are not host lists

The context allocates storage in the accelerator's global memory. Host initialization reaches it through an explicit queued copy.

```mojo
from std.gpu.host import DeviceContext

def make_device_buffer(ctx: DeviceContext, count: Int):
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    var device = ctx.enqueue_create_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_buf=device, src_buf=host)
    return device
```

**Manual worked example.** For count 4, the host and device each hold space for four Float32 values. Writing `[1,2,3,4]` to the host does not change the device until the copy is enqueued and executed.

## 0.4.3 Ceiling division chooses grid size

Launch enough blocks to cover the logical length, accepting a small guarded tail.

```mojo
def blocks_for(size: Int, block_size: Int) -> Int:
    return (size + block_size - 1) // block_size
```

**Manual worked example.** For 1,000 elements and 256 threads per block, `(1000+255)//256=4`. Four blocks launch 1,024 threads; 24 take the bounds exit. Three blocks would miss 232 elements.

## 0.4.4 Synchronization is an observation boundary

Enqueue operations in dependency order and synchronize only when the host must observe their results.

```mojo
def run_increment(ctx: DeviceContext, device, size: Int32) raises:
    comptime block_size = 256
    ctx.enqueue_function[increment_kernel](
        device, size,
        grid_dim=blocks_for(Int(size), block_size),
        block_dim=block_size,
    )
    ctx.synchronize()
```

**Manual worked example.** For size 5, one 256-thread block is launched and only threads 0–4 mutate data. `synchronize()` waits until those writes complete; a host copy read before that boundary could observe incomplete work.

Chapter 9 returns to coalescing, shared memory, and reductions after the tensor and autograd invariants are established.
