# 1.4 Device Abstraction Layer

A tensor operation should state what it computes while the device layer states where storage lives and when queued work becomes visible. Mojo 1.0 already provides owning `HostBuffer` and `DeviceBuffer` values through `DeviceContext`; wrapping those types is safer and more current than inventing a nullable raw-pointer allocator.

## 1.4.1 Device selection is an explicit policy

Automatic selection is convenient at an application boundary, but a tensor should record the chosen device concretely. That makes transfers, equality checks, and error messages deterministic.

```mojo
@fieldwise_init
struct Device(Copyable, Movable, Equatable):
    var kind: UInt8
    var ordinal: Int32

    comptime CPU = UInt8(0)
    comptime GPU = UInt8(1)
```

**Manual worked example.** `Device(Device.GPU,0)` identifies the first GPU; `Device(Device.CPU,0)` identifies host execution. They differ in `kind`, so an element-wise operation can reject their buffers or insert an explicit transfer rather than reading device memory as if it were host memory.

## 1.4.2 The context owns asynchronous work

`DeviceContext` creates buffers, enqueues copies and kernels, and synchronizes its stream. Buffers remain owned values on the host; kernel parameters receive device-accessible views or pointers derived by the runtime.

```mojo
from std.gpu.host import DeviceContext

def allocate_pair(ctx: DeviceContext, count: Int):
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    var device = ctx.enqueue_create_buffer[DType.float32](count)
    return (host, device)
```

**Manual worked example.** For `count=4`, both buffers reserve four `Float32` elements, or 16 payload bytes each, in different memory domains. Filling the host buffer does not mutate the device buffer; an enqueued host-to-device copy is required before a kernel reads those four values.

## 1.4.3 Transfers state direction and synchronization

Copies should name source and destination. Synchronize only when the host must observe queued results; unnecessary synchronization destroys overlap, while missing synchronization lets the host read before the device finishes.

```mojo
def round_trip(ctx: DeviceContext, host_in, device, host_out) raises:
    ctx.enqueue_copy(dst_buf=device, src_buf=host_in)
    # enqueue a kernel that mutates `device` here
    ctx.enqueue_copy(dst_buf=host_out, src_buf=device)
    ctx.synchronize()
```

**Manual worked example.** Start with host input `[1,2,3,4]`. The first copy makes those values available to the GPU. If a queued kernel doubles them, the second copy schedules `[2,4,6,8]` back to `host_out`; only after `synchronize()` may the CPU reliably check those four results.

## 1.4.4 Operation dispatch validates placement

The simplest correct policy rejects mixed-device operands. A later optimizer may insert transfers, but making them implicit too early hides large costs and complicates ownership.

```mojo
def require_same_device(left: Device, right: Device) raises:
    if left != right:
        raise Error("tensor operands must be on the same device")
```

**Manual worked example.** CPU:0 plus CPU:0 passes. GPU:0 plus GPU:0 passes. CPU:0 plus GPU:0 raises before launching work, preventing an invalid pointer access. GPU:0 plus GPU:1 also raises because peer-to-peer transfer has not been declared.

This layer now follows Mojo's runtime ownership model: contexts queue work, buffers own allocations, fixed device identifiers describe placement, and transfers remain visible in the program.
