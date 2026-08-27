# Chapter 10: Device Abstraction Layer — Discovery and Memory Management, Minimally

> "Chapter 6.5 built a device abstraction and was honest about exactly how much of it was real: a label, a transfer function, and two allocation branches that both quietly call the same CPU allocator. This chapter builds a second, even smaller one — not because the first was wrong, but because sometimes the fastest way to find out whether a bug lives in the hard part of a system is to rebuild the easy part from scratch and see if the bug follows it."

**What you will understand by the end of this chapter:**

- `SimpleDeviceManager`: a second, deliberately stripped-down device-discovery implementation — no `DeviceInfo` struct, no capability queries, two hardcoded devices instead of a list — and exactly what got removed relative to Chapter 6.5's `DeviceManager`
- Why `get_cpu_device()`/`get_gpu_device()` can hand back a device confidently reporting itself "Available: Yes" even when `discover_devices()` was never called at all
- `MemoryBlock`/`MemoryAllocator`/`MemoryManager`: a device-aware allocation layer built on exactly the RAII discipline Chapter 7 established, including the same "a copy doesn't own the data" rule — and a new, sharper version of the danger that rule creates
- The routing condition inside `MemoryManager.allocate` that decides which of two independent allocators handles a request — and the one combination of arguments where it silently sends a request to the *wrong* one
- Why `allocate_aligned_memory` — a function whose entire purpose is to apply a custom alignment — never actually uses the alignment value it's given
- How to read cumulative allocator statistics (`allocation_counter`, `total_allocated_bytes`) honestly, by hand-tracing a benchmark that leaves both numbers looking far larger than the memory actually in use at the end

**What you need to know first:**

- Chapter 6.5 (`DeviceType`, `DeviceManager`, and the established precedent that this book's "device discovery" is an honestly-labeled simulation, not a real hardware query)
- Chapter 7 (`_owns_data` and RAII — every allocating struct in this chapter follows the same allocate-in-constructor, free-in-destructor discipline)
- Chapter 2.4 (constructors, destructors, and `__copyinit__` as the mechanism behind every "does this copy own the memory" decision in this book)

## 10.1 Device Discovery: An Even Smaller `DeviceManager` `[FOUNDATIONAL]`

### Intuition

The file this section comes from opens with its own confession, in a comment: `# Ultra-minimal device discovery to test basic functionality`. That's a familiar move for anyone who has debugged a complicated system: when something in a large component might be broken, rebuild the smallest possible version of the same idea from scratch, with nothing left in it that could plausibly be the culprit, and see whether the problem follows. Chapter 6.5's `DeviceManager` tracked a `DeviceInfo` struct per device (type, ID, memory capacity, availability, a name string) and answered capability queries. `SimpleDeviceManager` throws all of that away and keeps exactly two hardcoded devices — a CPU and a GPU, referred to by name rather than stored in a list — with nothing configurable and nothing to query beyond "is it there" and "how much memory does it claim to have."

### Background

```mojo
struct SimpleDevice:
    var device_id: Int
    var device_type: DeviceType
    var memory_mb: Int
    var is_available: Bool

struct SimpleDeviceManager:
    var cpu_id: Int
    var cpu_memory: Int
    var cpu_available: Bool
    var gpu_id: Int
    var gpu_memory: Int
    var gpu_available: Bool
    var device_count: Int
```

| | Chapter 6.5 `DeviceManager` | `SimpleDeviceManager` (this chapter) |
|---|---|---|
| Devices tracked | a `DeviceInfo` per device (type, ID, capacity, name, availability) | two hardcoded flat field groups (`cpu_*`, `gpu_*`) — no list, no struct-per-device |
| Memory reported | queried per `DeviceInfo` | hardcoded constants (`4096` MB CPU, `8192` MB GPU) set once, in the constructor |
| "Discovery" | sets `has_gpu = True`, an honestly-labeled simulation | sets `cpu_available = True` and `gpu_available = True`, an equally honest simulation |
| Default device selection (`AUTO`) | yes, resolved by `DeviceManager.get_default_device()` | no such concept — callers ask for CPU or GPU directly |

### Worked Example 10.1.1 — Discovery that can't discover anything

`discover_devices()` doesn't inspect the machine it's running on at all — it unconditionally sets `self.cpu_available = True`, increments `device_count`, then unconditionally sets `self.gpu_available = True` and increments `device_count` again:

```mojo
fn discover_devices(mut self):
    # Simulate CPU discovery
    self.cpu_available = True
    self.device_count += 1
    # Simulate GPU discovery
    self.gpu_available = True
    self.device_count += 1
```

This reports "Found 2 devices" every single time, on every machine, with or without a GPU present — matching the recorded output exactly. This is the same honest-simulation pattern Chapter 6.5's `[COMMON TRAP]` already flagged for `DeviceManager` (which likewise hardcodes `has_gpu = True`), just with the pretense of a discovery step removed even further: there isn't even a boolean field being *read* here to decide availability, only two literals being *written*.

### Worked Example 10.1.2 — Manual construction bypasses the manager entirely

`test_basic_operations` never touches `SimpleDeviceManager` at all — it builds `SimpleDevice(0, CPU_DEVICE, 2048)` and `SimpleDevice(1, GPU_DEVICE, 4096)` directly. `SimpleDevice.__init__` sets `is_available = True` unconditionally, with no reference to whether any manager ever "discovered" these devices — so the recorded output shows both manually built devices reporting `Available: Yes`, for exactly the reason `get_type_string()` reports "CPU" or "GPU" correctly for them: every field on a `SimpleDevice` is supplied directly by whoever constructs one, and nothing about the struct itself enforces that construction only happens through `SimpleDeviceManager`'s discovery flow.

```
[COMMON TRAP]  get_cpu_device() doesn't check cpu_available before
                building a device that claims to be available

get_cpu_device() is `return SimpleDevice(self.cpu_id, CPU_DEVICE,
self.cpu_memory)` — it never reads self.cpu_available at all. Combined with
Worked Example 10.1.2's observation that SimpleDevice.__init__ always sets
is_available = True, this means calling manager.get_cpu_device() *before*
ever calling manager.discover_devices() still returns a SimpleDevice that
confidently reports "Available: Yes" when printed — even though the
manager's own cpu_available field is still False at that moment, and would
say (if you asked print_all_devices(), which does check the flag) that no
CPU has been found yet. Every test in this section calls discover_devices()
first, so this gap never surfaces in the recorded output — but nothing in
get_cpu_device()'s own code prevents a caller from skipping that step, and
nothing about the returned SimpleDevice's is_available field would tell you
they did.
```

## 10.2 Memory Management & Allocation: A Device-Aware Allocator on Chapter 7's RAII `[FOUNDATIONAL]`

### Intuition

Picture two teller windows at the same bank branch — one handling host-currency deposits, one handling a separate foreign currency — each keeping its own independent ledger of deposits, withdrawals, and a running balance. A customer doesn't pick a window directly; a branch manager looks at what currency they're depositing and sends them to the right teller. `MemoryManager` is that branch manager, and `self.host_allocator` / `self.device_allocator` are the two tellers — two separate `MemoryAllocator` instances, each tracking its own `total_allocated_bytes`, `active_allocations`, and `peak_allocations`, completely independently of the other.

### Background

```mojo
struct MemoryBlock:
    var ptr: UnsafePointer[UInt8]
    var size_bytes: Int
    var alignment: Int
    var memory_type: MemoryType
    var device_id: Int
    var allocation_id: Int
    var status: AllocationStatus
    var _owns_data: Bool

struct MemoryAllocator:
    var total_allocated_bytes: Int
    var total_deallocated_bytes: Int
    var active_allocations: Int
    var peak_allocations: Int
    var allocation_counter: Int
    var device_id: Int
    ...

struct MemoryManager:
    var host_allocator: MemoryAllocator   # constructed with device_id = 0
    var device_allocator: MemoryAllocator # constructed with device_id = 1
```

`MemoryBlock` follows the exact ownership rule Chapter 7 established for every buffer-owning struct in this book: `allocate()` sets `_owns_data = True` after a successful `UnsafePointer[UInt8].alloc`, `__del__` frees only `if self._owns_data`, and — the detail worth pausing on — `__copyinit__` explicitly sets the copy's `_owns_data = False`, with the source comment saying so directly: `# Copy doesn't own the data`. A copy of a `MemoryBlock` gets the *same* raw `ptr` as the original (pointers copy by value, same address both places) but is forbidden from being the one to free it — precisely Chapter 7's single-owner discipline, now applied to raw device-agnostic byte buffers instead of tensors.

`MemoryManager.allocate` decides which of its two allocators handles a request with one condition:

```mojo
fn allocate(mut self, size_bytes: Int, memory_type: MemoryType = HOST_MEMORY, device_id: Int = 0) -> MemoryBlock:
    if device_id == 0 or memory_type == HOST_MEMORY:
        return self.host_allocator.allocate_memory(size_bytes, memory_type)
    else:
        return self.device_allocator.allocate_memory(size_bytes, memory_type)
```

### Worked Example 10.2.1 — The routing condition, checked against all four combinations

`device_id == 0 OR memory_type == HOST_MEMORY` routes to `host_allocator` whenever *either* half is true. Every call in this chapter's own test suite happens to fall into one of the three combinations where that's clearly the intended behavior:

| `device_id` | `memory_type` | Condition | Routes to | Tested in this chapter? |
|---|---|---|---|---|
| 0 | `HOST_MEMORY` | true or true = true | `host_allocator` | yes |
| 1 | `DEVICE_MEMORY` | false or false = false | `device_allocator` | yes |
| 1 | `HOST_MEMORY` | false or true = true | `host_allocator` | not directly, but consistent |
| 0 | `DEVICE_MEMORY` | **true** or false = **true** | `host_allocator` | **no** |

That fourth row is the one worth tracing by hand, since nothing in this chapter's test suite exercises it: `manager.allocate(1024, DEVICE_MEMORY, 0)` — an explicit request for *device* memory — still satisfies `device_id == 0`, so it's routed to `host_allocator.allocate_memory(1024, DEVICE_MEMORY)`. The resulting block would carry `memory_type = DEVICE_MEMORY` (whatever was passed in) but `device_id = 0` (`host_allocator`'s own configured ID, from `MemoryBlock(size_bytes, alignment, memory_type, self.device_id)` inside `allocate_memory`) — a block whose `print_info()` would read `Type: Device` next to `Device: 0`, counted entirely within the *host* allocator's statistics. The caller asked for device memory at device 0 and silently got host memory instead, with only the `memory_type` label surviving to say otherwise.

### Worked Example 10.2.2 — The alignment parameter that never arrives

`allocate_aligned_memory`'s entire reason to exist is applying a caller-chosen alignment:

```mojo
fn allocate_aligned_memory(mut manager: MemoryManager, size_bytes: Int, alignment: Int, device_id: Int = 0) -> MemoryBlock:
    """Allocate aligned memory block."""
    var memory_type = HOST_MEMORY if device_id == 0 else DEVICE_MEMORY
    return manager.allocate(size_bytes, memory_type, device_id)
```

Trace the call `allocate_aligned_memory(manager, 4096, 256, 0)` from this chapter's own test: `alignment = 256` is bound as a parameter — and never referenced again in the function body. `manager.allocate(size_bytes, memory_type, device_id)` is called with exactly three arguments; `MemoryManager.allocate`'s signature has no `alignment` parameter at all for it to receive. That call reaches `host_allocator.allocate_memory(size_bytes, memory_type)` — two arguments — leaving `allocate_memory`'s own `alignment` parameter at its default, `DEFAULT_ALIGNMENT = 64`. The recorded output confirms the value was dropped rather than honored: `allocate_aligned_memory(manager, 4096, 256, 0)` prints back `Alignment: 64 bytes`, not `256`.

```
[COMMON TRAP]  a function's own docstring ("Allocate aligned memory block")
                is not proof that its body does what the docstring says

The 256 passed to allocate_aligned_memory is a value with nowhere to go: it
is read by nothing, forwarded by nothing, and stored on nothing. This is a
different flavor of bug than Chapter 9's packed-index aliasing — there, a
formula computed the wrong answer; here, a whole parameter is simply inert,
present in the signature and absent from every line that follows it. The
only way to catch this class of bug by reading code is to check, for every
parameter a function accepts, that it is actually used somewhere in that
function's body or passed on to something that uses it — a docstring and a
parameter name are a promise, not a guarantee.
```

### Worked Example 10.2.3 — Reading a benchmark's ledger honestly

`test_memory_performance` runs three allocate-then-immediately-deallocate benchmark passes (1000 iterations at 64 bytes, 500 at 4096 bytes, 100 at 1,048,576 bytes), then a mixed pattern: 10 more allocations where every even-indexed one (`i = 0, 2, 4, 6, 8`) is immediately deallocated and the 5 odd-indexed ones are left active, followed by 5 more allocations that are all left active. Every one of these calls increments `allocation_counter`, whether or not the block is ever deallocated:

```
benchmark passes:  1000 + 500 + 100                = 1,600 allocate() calls
mixed pattern:     10 + 5                           =    15 allocate() calls
                                                     -----
                                                     1,615 total
```

— matching the recorded `Allocation Counter: 1615` exactly. `total_allocated_bytes` is a *lifetime sum*, not a current-usage figure, and it keeps growing even though most of these blocks are freed within microseconds of being created: `1000×64 + 500×4096 + 100×1048576 = 106,969,600` from the benchmarks, plus `1024+2048+...+10240 = 56,320` and `512+1024+...+2560 = 7,680` from the two mixed-pattern loops (`56,320 + 7,680 = 64,000`), for a grand total of `107,033,600` bytes — exactly the recorded `Total Allocated: 107033600 bytes`. Meanwhile `Current Usage` — `total_allocated_bytes - total_deallocated_bytes` — is only `38,400` bytes: the five odd-indexed blocks from the first mixed loop (`2048+4096+6144+8192+10240 = 30,720`) plus all five blocks from the second (`512+1024+1536+2048+2560 = 7,680`), matching the recorded figure exactly. A monotonically increasing counter and a lifetime byte total both look alarming next to a comparatively tiny "current usage" number — but that gap is exactly the point of tracking all three separately, the same lesson Chapter 9's `SparseTensorCOO` taught from the opposite direction (there, a *reservation* number dwarfed actual *usage*; here, a *lifetime total* dwarfs actual *current* usage).

One more figure in this same output is easy to misread the opposite way: before a single byte has ever been allocated, `print_all_stats()` reports `Efficiency: 100.0%` for both allocators. `get_allocation_efficiency()` is `active_allocations / peak_allocations` — except when `peak_allocations == 0`, where it explicitly returns `1.0` instead of dividing zero by zero. Reading "100% efficient" as "fully utilized" would be backwards here: it means nothing has happened yet, not that everything available is in use.

```
[COMMON TRAP]  a copy's status field is a photograph, not a live view

MemoryBlock.__copyinit__ copies every field, including status and ptr,
exactly as they stood at the moment of the copy — and, as the Background
section noted, sets the copy's _owns_data to False so it won't double-free.
Nothing links a copy back to whatever it was copied from afterward. If code
anywhere holds two live values from the same original block — `var b2 = b1`
followed later by `b1.deallocate()` — b2.status still reads ALLOCATION_ACTIVE
and b2.is_valid() still returns True, because deallocate() only mutates the
receiver it's called on. b2.ptr is, at that point, a pointer to memory that
has already been freed through b1 — a stale reference with every outward
sign of still being good. None of this chapter's own tests create a second
independent copy this way (every test passes a MemoryBlock by `mut`
reference rather than assigning it to a second variable), so it doesn't
surface in the recorded output — but the struct's own copy semantics make it
possible, the same way Chapter 7's TensorView made a shared-buffer mutation
possible through a route its own tests never happened to exercise either.
```

## 10.3 Complete Runnable Code

The two sections above are drawn from two independent, runnable Mojo files. Each is reproduced here in full, exactly as written, together with its own recorded output.

### File: `46_device_abstraction_part_a.mojo` — Section 10.1

**Run:** `pixi run mojo 46_device_abstraction_part_a.mojo`

```mojo


# Ultra-minimal device discovery to test basic functionality


@register_passable("trivial")
struct DeviceType:
    var value: Int
    
    fn __init__(out self, value: Int):
        self.value = value
    
    fn __eq__(self, other: DeviceType) -> Bool:
        return self.value == other.value

alias CPU_DEVICE = DeviceType(0)
alias GPU_DEVICE = DeviceType(1)


struct SimpleDevice:
    var device_id: Int
    var device_type: DeviceType
    var memory_mb: Int
    var is_available: Bool
    
    fn __init__(out self, device_id: Int, device_type: DeviceType, memory_mb: Int):
        self.device_id = device_id
        self.device_type = device_type
        self.memory_mb = memory_mb
        self.is_available = True
    
    fn __copyinit__(out self, existing: Self):
        self.device_id = existing.device_id
        self.device_type = existing.device_type
        self.memory_mb = existing.memory_mb
        self.is_available = existing.is_available
    
    fn get_type_string(self) -> String:
        if self.device_type == CPU_DEVICE:
            return "CPU"
        elif self.device_type == GPU_DEVICE:
            return "GPU"
        else:
            return "Unknown"
    
    fn print_info(self):
        print("Device " + String(self.device_id) + ":")
        print("  Type: " + self.get_type_string())
        print("  Memory: " + String(self.memory_mb) + " MB")
        print("  Available: " + ("Yes" if self.is_available else "No"))


struct SimpleDeviceManager:
    var cpu_id: Int
    var cpu_memory: Int
    var cpu_available: Bool
    var gpu_id: Int
    var gpu_memory: Int
    var gpu_available: Bool
    var device_count: Int
    
    fn __init__(out self):
        self.cpu_id = 0
        self.cpu_memory = 4096
        self.cpu_available = False
        self.gpu_id = 1
        self.gpu_memory = 8192
        self.gpu_available = False
        self.device_count = 0
    
    fn __copyinit__(out self, existing: Self):
        self.cpu_id = existing.cpu_id
        self.cpu_memory = existing.cpu_memory
        self.cpu_available = existing.cpu_available
        self.gpu_id = existing.gpu_id
        self.gpu_memory = existing.gpu_memory
        self.gpu_available = existing.gpu_available
        self.device_count = existing.device_count
    
    fn discover_devices(mut self):
        print("Starting ultra-minimal device discovery...")
        
        # Simulate CPU discovery
        self.cpu_available = True
        self.device_count += 1
        
        # Simulate GPU discovery
        self.gpu_available = True
        self.device_count += 1
        
        print("Found " + String(self.device_count) + " devices")
    
    fn get_device_count(self) -> Int:
        return self.device_count
    
    fn get_cpu_device(self) -> SimpleDevice:
        return SimpleDevice(self.cpu_id, CPU_DEVICE, self.cpu_memory)
    
    fn get_gpu_device(self) -> SimpleDevice:
        return SimpleDevice(self.gpu_id, GPU_DEVICE, self.gpu_memory)
    
    fn print_all_devices(self):
        print("All Devices:")
        if self.cpu_available:
            var cpu = self.get_cpu_device()
            cpu.print_info()
            print("")
        if self.gpu_available:
            var gpu = self.get_gpu_device()
            gpu.print_info()
            print("")


fn test_ultra_simple_discovery():
    print("=== Testing Ultra-Simple Device Discovery ===")
    
    var manager = SimpleDeviceManager()
    manager.discover_devices()
    
    print("Device count: " + String(manager.get_device_count()))
    manager.print_all_devices()
    
    print("Ultra-simple discovery test completed")

fn test_device_creation():
    print("\n=== Testing Device Creation ===")
    
    var manager = SimpleDeviceManager()
    manager.discover_devices()
    
    var cpu = manager.get_cpu_device()
    var gpu = manager.get_gpu_device()
    
    print("CPU Device:")
    cpu.print_info()
    
    print("\nGPU Device:")
    gpu.print_info()
    
    print("Device creation test completed")

fn test_basic_operations():
    print("\n=== Testing Basic Operations ===")
    
    var device1 = SimpleDevice(0, CPU_DEVICE, 2048)
    var device2 = SimpleDevice(1, GPU_DEVICE, 4096)
    
    print("Manual device creation:")
    device1.print_info()
    print("")
    device2.print_info()
    
    print("Basic operations test completed")

fn run_all_ultra_simple_tests():
    print("=====================================")
    print("=== ULTRA-SIMPLE DEVICE TESTS ===")
    print("=====================================")
    
    test_ultra_simple_discovery()
    test_device_creation()
    test_basic_operations()
    
    print("\n=====================================")
    print("=== ULTRA-SIMPLE TESTS COMPLETE ===")
    print("=====================================")


fn main():
    print("=== Device Abstraction Layer - Part 1.4A (Ultra-Minimal) ===")
    print("Ultra-Simple Device Discovery & Management")
    print("")
    
    run_all_ultra_simple_tests()
    
    print("\n=== Ultra-Simple Device Discovery Summary ===")
    print("+ Basic device enumeration without system calls")
    print("+ Device type identification (CPU/GPU)")
    print("+ Simple device information storage")
    print("+ Device creation and access")
    print("+ No complex memory management or system interactions")
    print("")
    print("Ultra-Simple Device Discovery (Part 1.4A Minimal) Complete")
    print("This version should run without any system-level issues")
```

### Expected Output for `46_device_abstraction_part_a.mojo`

```
=== Device Abstraction Layer - Part 1.4A (Ultra-Minimal) ===
Ultra-Simple Device Discovery & Management

=====================================
=== ULTRA-SIMPLE DEVICE TESTS ===
=====================================
=== Testing Ultra-Simple Device Discovery ===
Starting ultra-minimal device discovery...
Found 2 devices
Device count: 2
All Devices:
Device 0:
  Type: CPU
  Memory: 4096 MB
  Available: Yes

Device 1:
  Type: GPU
  Memory: 8192 MB
  Available: Yes

Ultra-simple discovery test completed

=== Testing Device Creation ===
Starting ultra-minimal device discovery...
Found 2 devices
CPU Device:
Device 0:
  Type: CPU
  Memory: 4096 MB
  Available: Yes

GPU Device:
Device 1:
  Type: GPU
  Memory: 8192 MB
  Available: Yes
Device creation test completed

=== Testing Basic Operations ===
Manual device creation:
Device 0:
  Type: CPU
  Memory: 2048 MB
  Available: Yes

Device 1:
  Type: GPU
  Memory: 4096 MB
  Available: Yes
Basic operations test completed

=====================================
=== ULTRA-SIMPLE TESTS COMPLETE ===
=====================================

=== Ultra-Simple Device Discovery Summary ===
+ Basic device enumeration without system calls
+ Device type identification (CPU/GPU)
+ Simple device information storage
+ Device creation and access
+ No complex memory management or system interactions

Ultra-Simple Device Discovery (Part 1.4A Minimal) Complete
This version should run without any system-level issues
```

### File: `46_device_abstraction_part_b.mojo` — Section 10.2

**Run:** `pixi run mojo 46_device_abstraction_part_b.mojo`

```mojo

from memory import UnsafePointer
from sys import sizeof


# Device Abstraction Layer - Part 1.4B: Memory Management & Allocation
#
# This section implements unified memory allocation and management across
# different device types. Provides device-agnostic memory operations with
# automatic cleanup and efficient resource management.
#
# Key Design Principles:
# - Unified memory allocation API across all device types
# - Automatic memory cleanup and leak prevention
# - Device-specific optimization strategies
# - Memory usage tracking and monitoring
# - Simple and robust error handling

alias MAX_ALLOCATIONS = 1000
alias DEFAULT_ALIGNMENT = 64
alias MIN_ALLOCATION_SIZE = 64


@register_passable("trivial")
struct MemoryType:
    var value: Int
    
    fn __init__(out self, value: Int):
        self.value = value
    
    fn __eq__(self, other: MemoryType) -> Bool:
        return self.value == other.value

alias HOST_MEMORY = MemoryType(0)
alias DEVICE_MEMORY = MemoryType(1)
alias SHARED_MEMORY = MemoryType(2)

@register_passable("trivial")
struct AllocationStatus:
    var value: Int
    
    fn __init__(out self, value: Int):
        self.value = value
    
    fn __eq__(self, other: AllocationStatus) -> Bool:
        return self.value == other.value

alias ALLOCATION_ACTIVE = AllocationStatus(0)
alias ALLOCATION_FREED = AllocationStatus(1)
alias ALLOCATION_ERROR = AllocationStatus(2)


struct MemoryBlock:
    var ptr: UnsafePointer[UInt8]
    var size_bytes: Int
    var alignment: Int
    var memory_type: MemoryType
    var device_id: Int
    var allocation_id: Int
    var status: AllocationStatus
    var _owns_data: Bool
    
    fn __init__(out self, size_bytes: Int, alignment: Int, memory_type: MemoryType, device_id: Int):
        self.size_bytes = size_bytes
        self.alignment = alignment
        self.memory_type = memory_type
        self.device_id = device_id
        self.allocation_id = 0
        self.status = ALLOCATION_ERROR
        self._owns_data = False
        self.ptr = UnsafePointer[UInt8]()
    
    fn __copyinit__(out self, existing: Self):
        self.ptr = existing.ptr
        self.size_bytes = existing.size_bytes
        self.alignment = existing.alignment
        self.memory_type = existing.memory_type
        self.device_id = existing.device_id
        self.allocation_id = existing.allocation_id
        self.status = existing.status
        self._owns_data = False  # Copy doesn't own the data
    
    fn __del__(owned self):
        if self._owns_data and self.ptr:
            self.ptr.free()
    
    fn allocate(mut self) -> Bool:
        """Allocate memory block."""
        if self.size_bytes < MIN_ALLOCATION_SIZE:
            return False
        
        self.ptr = UnsafePointer[UInt8].alloc(self.size_bytes)
        self._owns_data = True
        self.status = ALLOCATION_ACTIVE
        
        # Initialize to zero
        for i in range(self.size_bytes):
            self.ptr[i] = 0
        
        return True
    
    fn deallocate(mut self):
        """Deallocate memory block."""
        if self._owns_data and self.ptr:
            self.ptr.free()
            self._owns_data = False
            self.status = ALLOCATION_FREED
    
    fn is_valid(self) -> Bool:
        """Check if memory block is valid."""
        return self.status == ALLOCATION_ACTIVE and self.ptr
    
    fn get_memory_type_string(self) -> String:
        """Get memory type as string."""
        if self.memory_type == HOST_MEMORY:
            return "Host"
        elif self.memory_type == DEVICE_MEMORY:
            return "Device"
        elif self.memory_type == SHARED_MEMORY:
            return "Shared"
        else:
            return "Unknown"
    
    fn get_status_string(self) -> String:
        """Get allocation status as string."""
        if self.status == ALLOCATION_ACTIVE:
            return "Active"
        elif self.status == ALLOCATION_FREED:
            return "Freed"
        elif self.status == ALLOCATION_ERROR:
            return "Error"
        else:
            return "Unknown"
    
    fn print_info(self):
        """Print memory block information."""
        print("Memory Block:")
        print("  ID: " + String(self.allocation_id))
        print("  Size: " + String(self.size_bytes) + " bytes")
        print("  Alignment: " + String(self.alignment) + " bytes")
        print("  Type: " + self.get_memory_type_string())
        print("  Device: " + String(self.device_id))
        print("  Status: " + self.get_status_string())
        print("  Valid: " + ("Yes" if self.is_valid() else "No"))


struct MemoryAllocator:
    var total_allocated_bytes: Int
    var total_deallocated_bytes: Int
    var active_allocations: Int
    var peak_allocations: Int
    var allocation_counter: Int
    var device_id: Int
    var supports_host_memory: Bool
    var supports_device_memory: Bool
    var supports_shared_memory: Bool
    
    fn __init__(out self, device_id: Int):
        self.total_allocated_bytes = 0
        self.total_deallocated_bytes = 0
        self.active_allocations = 0
        self.peak_allocations = 0
        self.allocation_counter = 0
        self.device_id = device_id
        self.supports_host_memory = True
        self.supports_device_memory = True
        self.supports_shared_memory = False
    
    fn __copyinit__(out self, existing: Self):
        self.total_allocated_bytes = existing.total_allocated_bytes
        self.total_deallocated_bytes = existing.total_deallocated_bytes
        self.active_allocations = existing.active_allocations
        self.peak_allocations = existing.peak_allocations
        self.allocation_counter = existing.allocation_counter
        self.device_id = existing.device_id
        self.supports_host_memory = existing.supports_host_memory
        self.supports_device_memory = existing.supports_device_memory
        self.supports_shared_memory = existing.supports_shared_memory
    
    fn allocate_memory(mut self, size_bytes: Int, memory_type: MemoryType, alignment: Int = DEFAULT_ALIGNMENT) -> MemoryBlock:
        """Allocate memory block of specified size and type."""
        # Validate inputs
        if size_bytes <= 0:
            return MemoryBlock(0, alignment, memory_type, self.device_id)
        
        # Check if memory type is supported
        if memory_type == HOST_MEMORY and not self.supports_host_memory:
            return MemoryBlock(0, alignment, memory_type, self.device_id)
        elif memory_type == DEVICE_MEMORY and not self.supports_device_memory:
            return MemoryBlock(0, alignment, memory_type, self.device_id)
        elif memory_type == SHARED_MEMORY and not self.supports_shared_memory:
            return MemoryBlock(0, alignment, memory_type, self.device_id)
        
        # Create memory block
        var block = MemoryBlock(size_bytes, alignment, memory_type, self.device_id)
        block.allocation_id = self.allocation_counter
        self.allocation_counter += 1
        
        # Attempt allocation
        if block.allocate():
            self.total_allocated_bytes += size_bytes
            self.active_allocations += 1
            if self.active_allocations > self.peak_allocations:
                self.peak_allocations = self.active_allocations
        
        return block
    
    fn deallocate_memory(mut self, mut block: MemoryBlock):
        """Deallocate memory block."""
        if block.is_valid():
            self.total_deallocated_bytes += block.size_bytes
            self.active_allocations -= 1
            block.deallocate()
    
    fn get_total_allocated(self) -> Int:
        """Get total bytes allocated."""
        return self.total_allocated_bytes
    
    fn get_total_deallocated(self) -> Int:
        """Get total bytes deallocated."""
        return self.total_deallocated_bytes
    
    fn get_active_allocations(self) -> Int:
        """Get number of active allocations."""
        return self.active_allocations
    
    fn get_peak_allocations(self) -> Int:
        """Get peak number of allocations."""
        return self.peak_allocations
    
    fn get_memory_usage(self) -> Int:
        """Get current memory usage in bytes."""
        return self.total_allocated_bytes - self.total_deallocated_bytes
    
    fn get_allocation_efficiency(self) -> Float64:
        """Calculate allocation efficiency (allocated/peak ratio)."""
        if self.peak_allocations == 0:
            return 1.0
        return Float64(self.active_allocations) / Float64(self.peak_allocations)
    
    fn print_stats(self):
        """Print allocator statistics."""
        print("Memory Allocator Statistics:")
        print("  Device ID: " + String(self.device_id))
        print("  Total Allocated: " + String(self.total_allocated_bytes) + " bytes")
        print("  Total Deallocated: " + String(self.total_deallocated_bytes) + " bytes")
        print("  Current Usage: " + String(self.get_memory_usage()) + " bytes")
        print("  Active Allocations: " + String(self.active_allocations))
        print("  Peak Allocations: " + String(self.peak_allocations))
        print("  Allocation Counter: " + String(self.allocation_counter))
        print("  Efficiency: " + String(self.get_allocation_efficiency() * 100.0) + "%")
        print("  Host Memory: " + ("Supported" if self.supports_host_memory else "Not supported"))
        print("  Device Memory: " + ("Supported" if self.supports_device_memory else "Not supported"))
        print("  Shared Memory: " + ("Supported" if self.supports_shared_memory else "Not supported"))


struct MemoryManager:
    var host_allocator: MemoryAllocator
    var device_allocator: MemoryAllocator
    var total_allocators: Int
    var default_memory_type: MemoryType
    
    fn __init__(out self):
        self.host_allocator = MemoryAllocator(0)  # CPU device
        self.device_allocator = MemoryAllocator(1)  # GPU device
        self.total_allocators = 2
        self.default_memory_type = HOST_MEMORY
    
    fn __copyinit__(out self, existing: Self):
        self.host_allocator = existing.host_allocator
        self.device_allocator = existing.device_allocator
        self.total_allocators = existing.total_allocators
        self.default_memory_type = existing.default_memory_type
    
    fn allocate(mut self, size_bytes: Int, memory_type: MemoryType = HOST_MEMORY, device_id: Int = 0) -> MemoryBlock:
        """Allocate memory on specified device."""
        if device_id == 0 or memory_type == HOST_MEMORY:
            return self.host_allocator.allocate_memory(size_bytes, memory_type)
        else:
            return self.device_allocator.allocate_memory(size_bytes, memory_type)
    
    fn deallocate(mut self, mut block: MemoryBlock):
        """Deallocate memory block."""
        if block.device_id == 0:
            self.host_allocator.deallocate_memory(block)
        else:
            self.device_allocator.deallocate_memory(block)
    
    fn get_total_memory_usage(self) -> Int:
        """Get total memory usage across all devices."""
        return self.host_allocator.get_memory_usage() + self.device_allocator.get_memory_usage()
    
    fn get_total_allocations(self) -> Int:
        """Get total active allocations across all devices."""
        return self.host_allocator.get_active_allocations() + self.device_allocator.get_active_allocations()
    
    fn print_all_stats(self):
        """Print statistics for all allocators."""
        print("Memory Manager - All Allocators:")
        print("Total allocators: " + String(self.total_allocators))
        print("Total memory usage: " + String(self.get_total_memory_usage()) + " bytes")
        print("Total allocations: " + String(self.get_total_allocations()))
        print("")
        
        print("--- Host Allocator ---")
        self.host_allocator.print_stats()
        print("")
        
        print("--- Device Allocator ---")
        self.device_allocator.print_stats()
        print("")


fn create_memory_manager() -> MemoryManager:
    """Create and initialize memory manager."""
    return MemoryManager()

fn allocate_tensor_memory(mut manager: MemoryManager, element_count: Int, element_size: Int, device_id: Int = 0) -> MemoryBlock:
    """Allocate memory for tensor data."""
    var total_size = element_count * element_size
    var memory_type = HOST_MEMORY if device_id == 0 else DEVICE_MEMORY
    return manager.allocate(total_size, memory_type, device_id)

fn allocate_aligned_memory(mut manager: MemoryManager, size_bytes: Int, alignment: Int, device_id: Int = 0) -> MemoryBlock:
    """Allocate aligned memory block."""
    var memory_type = HOST_MEMORY if device_id == 0 else DEVICE_MEMORY
    return manager.allocate(size_bytes, memory_type, device_id)


struct PerformanceTimer:
    var name: String
    
    fn __init__(out self, name: String):
        self.name = name
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
    
    fn start(self):
        print("Starting: " + self.name)
    
    fn end(self, operations: Int = 1):
        var ops_str: String = String(operations)
        print("Completed: " + self.name + " (" + ops_str + " operations)")

fn benchmark_memory_allocation(mut manager: MemoryManager, size_bytes: Int, iterations: Int) -> Float64:
    """Benchmark memory allocation performance."""
    var timer = PerformanceTimer("Memory Allocation Benchmark")
    timer.start()
    
    for _ in range(iterations):
        var block = manager.allocate(size_bytes, HOST_MEMORY, 0)
        if block.is_valid():
            manager.deallocate(block)
    
    timer.end(iterations)
    return Float64(iterations) / 1000.0


fn test_basic_memory_allocation():
    print("=== Testing Basic Memory Allocation ===")
    
    print("\n1. Memory Manager Creation:")
    var manager = create_memory_manager()
    manager.print_all_stats()
    
    print("\n2. Host Memory Allocation:")
    var host_block = manager.allocate(1024, HOST_MEMORY, 0)
    host_block.print_info()
    
    print("\n3. Device Memory Allocation:")
    var device_block = manager.allocate(2048, DEVICE_MEMORY, 1)
    device_block.print_info()
    
    print("\n4. Memory Statistics After Allocation:")
    manager.print_all_stats()
    
    print("\n5. Memory Deallocation:")
    manager.deallocate(host_block)
    manager.deallocate(device_block)
    
    print("\n6. Memory Statistics After Deallocation:")
    manager.print_all_stats()
    
    print("Basic memory allocation testing completed successfully")

fn test_memory_block_operations():
    print("\n=== Testing Memory Block Operations ===")
    
    print("\n1. Memory Block Creation:")
    var block = MemoryBlock(512, 64, HOST_MEMORY, 0)
    block.allocation_id = 100
    print("Before allocation:")
    block.print_info()
    
    print("\n2. Memory Block Allocation:")
    var success = block.allocate()
    print("Allocation success: " + ("Yes" if success else "No"))
    print("After allocation:")
    block.print_info()
    
    print("\n3. Memory Block Validation:")
    print("Is valid: " + ("Yes" if block.is_valid() else "No"))
    
    print("\n4. Memory Block Deallocation:")
    block.deallocate()
    print("After deallocation:")
    block.print_info()
    
    print("Memory block operations testing completed successfully")

fn test_tensor_memory_allocation():
    print("\n=== Testing Tensor Memory Allocation ===")
    
    var manager = create_memory_manager()
    
    print("\n1. Float32 Tensor Memory:")
    var float32_elements = 1000
    var float32_size = sizeof[Float32]()
    var float32_block = allocate_tensor_memory(manager, float32_elements, float32_size, 0)
    
    print("Float32 tensor memory:")
    float32_block.print_info()
    
    print("\n2. Float64 Tensor Memory:")
    var float64_elements = 500
    var float64_size = sizeof[Float64]()
    var float64_block = allocate_tensor_memory(manager, float64_elements, float64_size, 1)
    
    print("Float64 tensor memory:")
    float64_block.print_info()
    
    print("\n3. Aligned Memory Allocation:")
    var aligned_block = allocate_aligned_memory(manager, 4096, 256, 0)
    
    print("Aligned memory:")
    aligned_block.print_info()
    
    print("\n4. Memory Manager Statistics:")
    manager.print_all_stats()
    
    print("\n5. Cleanup:")
    manager.deallocate(float32_block)
    manager.deallocate(float64_block)
    manager.deallocate(aligned_block)
    
    print("Final statistics:")
    manager.print_all_stats()
    
    print("Tensor memory allocation testing completed successfully")

fn test_memory_performance():
    print("\n=== Testing Memory Performance ===")
    
    var manager = create_memory_manager()
    
    print("\n1. Small Allocation Performance:")
    var small_result = benchmark_memory_allocation(manager, 64, 1000)
    print("Small allocation performance: " + String(small_result) + " k-ops")
    
    print("\n2. Medium Allocation Performance:")
    var medium_result = benchmark_memory_allocation(manager, 4096, 500)
    print("Medium allocation performance: " + String(medium_result) + " k-ops")
    
    print("\n3. Large Allocation Performance:")
    var large_result = benchmark_memory_allocation(manager, 1048576, 100)  # 1MB
    print("Large allocation performance: " + String(large_result) + " k-ops")
    
    print("\n4. Allocation Pattern Test:")
    var timer = PerformanceTimer("Mixed Allocation Pattern")
    timer.start()
    
    # Allocate and deallocate blocks without storing in list
    for i in range(10):
        var size = (i + 1) * 1024
        var block = manager.allocate(size, HOST_MEMORY, 0)
        if i % 2 == 0:  # Deallocate every other block
            manager.deallocate(block)
    
    # Allocate more blocks
    for i in range(5):
        var size = (i + 1) * 512
        var _ = manager.allocate(size, HOST_MEMORY, 0)
        # Let these blocks remain allocated
    
    timer.end(15)
    
    print("Final memory statistics:")
    manager.print_all_stats()
    
    print("Memory performance testing completed successfully")

fn test_memory_error_handling():
    print("\n=== Testing Memory Error Handling ===")
    
    var test_count = 0
    var passed_count = 0
    
    print("\n1. Invalid Size Allocation:")
    test_count += 1
    var manager = create_memory_manager()
    var invalid_block = manager.allocate(0, HOST_MEMORY, 0)
    if not invalid_block.is_valid():
        print("Correctly rejected zero-size allocation")
        passed_count += 1
    else:
        print("Should have rejected zero-size allocation")
    
    test_count += 1
    var negative_block = manager.allocate(-100, HOST_MEMORY, 0)
    if not negative_block.is_valid():
        print("Correctly rejected negative-size allocation")
        passed_count += 1
    else:
        print("Should have rejected negative-size allocation")
    
    print("\n2. Double Deallocation:")
    test_count += 1
    var block = manager.allocate(1024, HOST_MEMORY, 0)
    if block.is_valid():
        manager.deallocate(block)
        # Second deallocation should be safe
        manager.deallocate(block)
        print("Double deallocation handled safely")
        passed_count += 1
    else:
        print("Failed to create block for double deallocation test")
    
    var passed_str: String = String(passed_count)
    var total_str: String = String(test_count)
    print("\nError Handling Summary: " + passed_str + "/" + total_str + " tests passed")
    
    if passed_count == test_count:
        print("All error handling tests passed")
    else:
        print("Some error handling tests failed")

fn test_memory_efficiency():
    print("\n=== Testing Memory Efficiency ===")
    
    var manager = create_memory_manager()
    
    print("\n1. Memory Fragmentation Test:")
    # Allocate many small blocks
    for _ in range(100):
        var _ = manager.allocate(64, HOST_MEMORY, 0)
    
    print("After 100 small allocations:")
    manager.print_all_stats()
    
    print("\n2. Memory Pool Efficiency:")
    # Test allocation efficiency
    var efficiency = manager.host_allocator.get_allocation_efficiency()
    print("Host allocator efficiency: " + String(efficiency * 100.0) + "%")
    
    var device_efficiency = manager.device_allocator.get_allocation_efficiency()
    print("Device allocator efficiency: " + String(device_efficiency * 100.0) + "%")
    
    print("\n3. Memory Usage Analysis:")
    var total_usage = manager.get_total_memory_usage()
    var total_allocations = manager.get_total_allocations()
    
    print("Total memory usage: " + String(total_usage) + " bytes")
    print("Total allocations: " + String(total_allocations))
    
    if total_allocations > 0:
        var avg_allocation_size = total_usage / total_allocations
        print("Average allocation size: " + String(avg_allocation_size) + " bytes")
    
    print("Memory efficiency testing completed successfully")

fn run_all_memory_tests():
    print("=====================================")
    print("=== MEMORY MANAGEMENT TEST SUITE ===")
    print("=====================================")
    
    test_basic_memory_allocation()
    test_memory_block_operations()
    test_tensor_memory_allocation()
    test_memory_performance()
    test_memory_error_handling()
    test_memory_efficiency()
    
    print("\n=====================================")
    print("=== MEMORY MANAGEMENT TESTS COMPLETE ===")
    print("=====================================")


fn main():
    print("=== Device Abstraction Layer - Part 1.4B ===")
    print("Memory Management & Allocation - Complete Standalone Module")
    print("")
    
    run_all_memory_tests()
    
    print("\n=== Memory Management & Allocation Summary ===")
    print("+ Unified memory allocation API across device types")
    print("+ Device-specific memory allocators (CPU/GPU)")
    print("+ Automatic memory cleanup and leak prevention")
    print("+ Memory usage tracking and statistics")
    print("+ Performance benchmarking and efficiency analysis")
    print("+ Comprehensive error handling and validation")
    print("+ Support for different memory types (Host/Device/Shared)")
    print("+ Aligned memory allocation with custom alignment")
    print("+ Tensor-specific memory allocation utilities")
    print("+ Integration-ready for data transfer operations")
    print("")
    print("Performance Characteristics:")
    print("- Memory allocation: O(1) - direct system allocation")
    print("- Memory deallocation: O(1) - direct system deallocation")
    print("- Memory tracking: O(1) - simple counter updates")
    print("- Statistics computation: O(1) - cached values")
    print("")
    print("Memory Types Supported:")
    print("- Host Memory: CPU-accessible system memory")
    print("- Device Memory: GPU-accessible device memory")
    print("- Shared Memory: Accessible by both CPU and GPU")
    print("- Aligned Memory: Custom alignment for SIMD operations")
    print("")
    print("Allocation Features:")
    print("- Size and alignment specification")
    print("- Device-specific optimization")
    print("- Memory pool management")
    print("- Usage statistics and monitoring")
    print("- Automatic cleanup on destruction")
    print("")
    print("Memory Management & Allocation (Part 1.4B) Complete")
    print("Ready for integration with data transfer operations")
```

### Expected Output for `46_device_abstraction_part_b.mojo`

```
=== Device Abstraction Layer - Part 1.4B ===
Memory Management & Allocation - Complete Standalone Module

=====================================
=== MEMORY MANAGEMENT TEST SUITE ===
=====================================
=== Testing Basic Memory Allocation ===

1. Memory Manager Creation:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 0 bytes
Total allocations: 0

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 0 bytes
  Total Deallocated: 0 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 0
  Allocation Counter: 0
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 0 bytes
  Total Deallocated: 0 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 0
  Allocation Counter: 0
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported


2. Host Memory Allocation:
Memory Block:
  ID: 0
  Size: 1024 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Active
  Valid: Yes

3. Device Memory Allocation:
Memory Block:
  ID: 0
  Size: 2048 bytes
  Alignment: 64 bytes
  Type: Device
  Device: 1
  Status: Active
  Valid: Yes

4. Memory Statistics After Allocation:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 3072 bytes
Total allocations: 2

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 1024 bytes
  Total Deallocated: 0 bytes
  Current Usage: 1024 bytes
  Active Allocations: 1
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 2048 bytes
  Total Deallocated: 0 bytes
  Current Usage: 2048 bytes
  Active Allocations: 1
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported


5. Memory Deallocation:

6. Memory Statistics After Deallocation:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 0 bytes
Total allocations: 0

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 1024 bytes
  Total Deallocated: 1024 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 0.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 2048 bytes
  Total Deallocated: 2048 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 0.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

Basic memory allocation testing completed successfully

=== Testing Memory Block Operations ===

1. Memory Block Creation:
Before allocation:
Memory Block:
  ID: 100
  Size: 512 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Error
  Valid: No

2. Memory Block Allocation:
Allocation success: Yes
After allocation:
Memory Block:
  ID: 100
  Size: 512 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Active
  Valid: Yes

3. Memory Block Validation:
Is valid: Yes

4. Memory Block Deallocation:
After deallocation:
Memory Block:
  ID: 100
  Size: 512 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Freed
  Valid: No
Memory block operations testing completed successfully

=== Testing Tensor Memory Allocation ===

1. Float32 Tensor Memory:
Float32 tensor memory:
Memory Block:
  ID: 0
  Size: 4000 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Active
  Valid: Yes

2. Float64 Tensor Memory:
Float64 tensor memory:
Memory Block:
  ID: 0
  Size: 4000 bytes
  Alignment: 64 bytes
  Type: Device
  Device: 1
  Status: Active
  Valid: Yes

3. Aligned Memory Allocation:
Aligned memory:
Memory Block:
  ID: 1
  Size: 4096 bytes
  Alignment: 64 bytes
  Type: Host
  Device: 0
  Status: Active
  Valid: Yes

4. Memory Manager Statistics:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 12096 bytes
Total allocations: 3

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 8096 bytes
  Total Deallocated: 0 bytes
  Current Usage: 8096 bytes
  Active Allocations: 2
  Peak Allocations: 2
  Allocation Counter: 2
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 4000 bytes
  Total Deallocated: 0 bytes
  Current Usage: 4000 bytes
  Active Allocations: 1
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported


5. Cleanup:
Final statistics:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 0 bytes
Total allocations: 0

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 8096 bytes
  Total Deallocated: 8096 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 2
  Allocation Counter: 2
  Efficiency: 0.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 4000 bytes
  Total Deallocated: 4000 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 1
  Allocation Counter: 1
  Efficiency: 0.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

Tensor memory allocation testing completed successfully

=== Testing Memory Performance ===

1. Small Allocation Performance:
Starting: Memory Allocation Benchmark
Completed: Memory Allocation Benchmark (1000 operations)
Small allocation performance: 1.0 k-ops

2. Medium Allocation Performance:
Starting: Memory Allocation Benchmark
Completed: Memory Allocation Benchmark (500 operations)
Medium allocation performance: 0.5 k-ops

3. Large Allocation Performance:
Starting: Memory Allocation Benchmark
Completed: Memory Allocation Benchmark (100 operations)
Large allocation performance: 0.1 k-ops

4. Allocation Pattern Test:
Starting: Mixed Allocation Pattern
Completed: Mixed Allocation Pattern (15 operations)
Final memory statistics:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 38400 bytes
Total allocations: 10

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 107033600 bytes
  Total Deallocated: 106995200 bytes
  Current Usage: 38400 bytes
  Active Allocations: 10
  Peak Allocations: 10
  Allocation Counter: 1615
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 0 bytes
  Total Deallocated: 0 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 0
  Allocation Counter: 0
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

Memory performance testing completed successfully

=== Testing Memory Error Handling ===

1. Invalid Size Allocation:
Correctly rejected zero-size allocation
Correctly rejected negative-size allocation

2. Double Deallocation:
Double deallocation handled safely

Error Handling Summary: 3/3 tests passed
All error handling tests passed

=== Testing Memory Efficiency ===

1. Memory Fragmentation Test:
After 100 small allocations:
Memory Manager - All Allocators:
Total allocators: 2
Total memory usage: 6400 bytes
Total allocations: 100

--- Host Allocator ---
Memory Allocator Statistics:
  Device ID: 0
  Total Allocated: 6400 bytes
  Total Deallocated: 0 bytes
  Current Usage: 6400 bytes
  Active Allocations: 100
  Peak Allocations: 100
  Allocation Counter: 100
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported

--- Device Allocator ---
Memory Allocator Statistics:
  Device ID: 1
  Total Allocated: 0 bytes
  Total Deallocated: 0 bytes
  Current Usage: 0 bytes
  Active Allocations: 0
  Peak Allocations: 0
  Allocation Counter: 0
  Efficiency: 100.0%
  Host Memory: Supported
  Device Memory: Supported
  Shared Memory: Not supported


2. Memory Pool Efficiency:
Host allocator efficiency: 100.0%
Device allocator efficiency: 100.0%

3. Memory Usage Analysis:
Total memory usage: 6400 bytes
Total allocations: 100
Average allocation size: 64.0 bytes
Memory efficiency testing completed successfully

=====================================
=== MEMORY MANAGEMENT TESTS COMPLETE ===
=====================================

=== Memory Management & Allocation Summary ===
+ Unified memory allocation API across device types
+ Device-specific memory allocators (CPU/GPU)
+ Automatic memory cleanup and leak prevention
+ Memory usage tracking and statistics
+ Performance benchmarking and efficiency analysis
+ Comprehensive error handling and validation
+ Support for different memory types (Host/Device/Shared)
+ Aligned memory allocation with custom alignment
+ Tensor-specific memory allocation utilities
+ Integration-ready for data transfer operations

Performance Characteristics:
- Memory allocation: O(1) - direct system allocation
- Memory deallocation: O(1) - direct system deallocation
- Memory tracking: O(1) - simple counter updates
- Statistics computation: O(1) - cached values

Memory Types Supported:
- Host Memory: CPU-accessible system memory
- Device Memory: GPU-accessible device memory
- Shared Memory: Accessible by both CPU and GPU
- Aligned Memory: Custom alignment for SIMD operations

Allocation Features:
- Size and alignment specification
- Device-specific optimization
- Memory pool management
- Usage statistics and monitoring
- Automatic cleanup on destruction

Memory Management & Allocation (Part 1.4B) Complete
Ready for integration with data transfer operations
```

## Chapter Summary

`SimpleDeviceManager` is a second, deliberately smaller device-discovery implementation than Chapter 6.5's `DeviceManager`: two hardcoded devices instead of a queryable list, constant memory figures instead of anything read from the machine, and the same honestly-labeled simulation ("Simulate CPU discovery," in the source's own comment) rather than a real hardware query — with one gap Chapter 6.5's version didn't share: `get_cpu_device()`/`get_gpu_device()` never check whether discovery has actually run, and `SimpleDevice`'s own constructor always marks a device `Available: Yes` regardless. `MemoryBlock`, `MemoryAllocator`, and `MemoryManager` build a device-aware allocation layer directly on Chapter 7's RAII discipline — same allocate-in-one-place, free-in-`__del__` pattern, same "a copy doesn't own the data" rule — and this chapter traced two ways that discipline can go wrong in practice: `MemoryManager.allocate`'s routing condition silently sends a `DEVICE_MEMORY` request at `device_id=0` to the *host* allocator instead, and `allocate_aligned_memory` accepts an `alignment` argument it never once uses, leaving every "aligned" allocation at the hardcoded default. The chapter closed by tracing a benchmark's cumulative statistics by hand — a `1615`-call `allocation_counter` and a `107,033,600`-byte lifetime total sitting next to a mere `38,400` bytes of blocks still actually active — the same "reservation versus real usage" gap Chapter 9's `SparseTensorCOO` demonstrated from the opposite direction.

## Self-Check Questions

1. `SimpleDeviceManager.get_gpu_device()` is called on a freshly constructed manager, before `discover_devices()` has ever run. What does the returned `SimpleDevice.is_available` field report, and why does nothing in `get_gpu_device()`'s own code prevent this?
2. Using the routing condition `device_id == 0 or memory_type == HOST_MEMORY`, which allocator (`host_allocator` or `device_allocator`) handles a call to `manager.allocate(512, DEVICE_MEMORY, 0)`? What `device_id` field would the returned `MemoryBlock` actually carry, and why might that surprise a caller who explicitly asked for device 0's device memory?
3. Trace `allocate_aligned_memory(manager, 8192, 128, 0)` one function call at a time, the way Worked Example 10.2.2 traced the `256`-byte case, and state what `alignment` value the resulting `MemoryBlock` actually ends up with.
4. A benchmark loop calls `manager.allocate(size, HOST_MEMORY, 0)` 200 times, deallocating immediately after each call. What does `allocation_counter` equal at the end? What does `active_allocations` equal? Explain why those two numbers diverge even though every single allocation was immediately freed.
5. `var b2 = b1` copies a `MemoryBlock`, and then `b1.deallocate()` is called. What does `b2.is_valid()` return immediately afterward, and why — given that the same underlying memory `b2.ptr` points to has, in reality, already been freed?

## Where We Go Next

Chapter 11 (`part1/06-memory-management-system.md`) is where the "a copy doesn't own the data, only one owner ever frees it" rule this chapter and Chapter 7 both leaned on gets replaced with something more flexible: a real reference count, shared explicitly between every struct that points at the same buffer, so that *more than one* live owner can exist safely at once — the direct fix for the stale-copy danger this chapter's closing `[COMMON TRAP]` traced but never resolved.

## Worked Solutions

**1.** It reports `True`. `get_gpu_device()` is `return SimpleDevice(self.gpu_id, GPU_DEVICE, self.gpu_memory)` — it never reads `self.gpu_available` at all, so whether discovery has run is irrelevant to what this function returns. `SimpleDevice.__init__` then unconditionally sets `self.is_available = True` on the new object it constructs, with no parameter controlling that field at all. The two facts compound: even though the *manager* would honestly report `gpu_available = False` if asked directly, the *device object* it hands back has no way to inherit that answer, because nothing wires the manager's flag into the device's own field.

**2.** `device_id == 0` is `True`, so the condition is `True` regardless of the `memory_type` half — `host_allocator` handles the call, via `self.host_allocator.allocate_memory(512, DEVICE_MEMORY)`. Inside `allocate_memory`, the returned block is built as `MemoryBlock(size_bytes, alignment, memory_type, self.device_id)`, where `self.device_id` is `host_allocator`'s own configured ID — `0` — not the `0` the caller happened to also pass to `MemoryManager.allocate`. The surprise: the block's `device_id` field (`0`) and `memory_type` field (`DEVICE_MEMORY`) together describe something that never actually happened this way — "device memory at device 0" — and the allocation is counted entirely in `host_allocator`'s statistics, not `device_allocator`'s, even though the caller's whole intent was to get device memory.

**3.** `allocate_aligned_memory(manager, 8192, 128, 0)`: `alignment = 128` is bound as a parameter and never read again. `memory_type = HOST_MEMORY` (since `device_id == 0`). The call becomes `manager.allocate(8192, HOST_MEMORY, 0)` — three arguments, no room for `128` to travel through. `MemoryManager.allocate` calls `self.host_allocator.allocate_memory(8192, HOST_MEMORY)` — two arguments — so `allocate_memory`'s own `alignment` parameter falls back to its default, `DEFAULT_ALIGNMENT = 64`. The resulting `MemoryBlock.alignment` is `64`, not the requested `128` — the same drop traced for `256` in Worked Example 10.2.2, with a different number attached.

**4.** `allocation_counter` equals `200` — it increments once per call to `allocate_memory`, inside the block-construction path, regardless of whether the resulting block is ever deallocated. `active_allocations` equals `0`, because each of the 200 blocks was deallocated immediately after being created, and `deallocate_memory` decrements `active_allocations` by exactly one for each successful deallocation. The two numbers diverge because `allocation_counter` is a monotonically increasing lifetime count of "how many times was `allocate` called," while `active_allocations` is a live count of "how many of those are still outstanding right now" — exactly the distinction Worked Example 10.2.3 traced between `Allocation Counter: 1615` and `Current Usage: 38400 bytes` for a different benchmark.

**5.** `b2.is_valid()` still returns `True`. `is_valid()` is `self.status == ALLOCATION_ACTIVE and self.ptr` — and `b2.status` was copied from `b1` at the moment `var b2 = b1` ran, back when `b1` really was active. `b1.deallocate()` only mutates `b1`: it sets `b1.status = ALLOCATION_FREED` and frees `b1.ptr`, but has no reference to `b2` at all and so cannot update `b2.status` to match. `b2.ptr` holds the same address `b1.ptr` did (pointers copy by value), which by this point has already been passed to `.free()` through `b1` — so `b2` reports itself valid while pointing at memory that is, in reality, no longer owned by anything and may already have been reused.
