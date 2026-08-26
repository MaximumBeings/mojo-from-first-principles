# 0.4 GPU Programming Introduction

# Mojo GPU Programming Introduction - Part 0.4

This section demonstrates GPU programming fundamentals in Mojo, essential for building high-performance automatic differentiation frameworks. GPU programming enables massive parallelization of tensor operations and gradient computations.

---

## 0.4 GPU Programming Introduction

### File: `15_gpu_basics.mojo`

**Execution:** `pixi run mojo 15_gpu_basics.mojo`

```mojo
from memory import UnsafePointer

fn simple_vector_add_cpu(a: UnsafePointer[Float32], b: UnsafePointer[Float32], 
                        result: UnsafePointer[Float32], size: Int):
    """CPU version of vector addition for comparison."""
    for i in range(size):
        result[i] = a[i] + b[i]

fn gpu_basics_demo():
    """Demonstrate basic GPU concepts without GPU execution."""
    print("=== GPU Programming Basics ===")
    
    print("GPU Programming Concepts:")
    print("1. Thread Hierarchy:")
    print("   - Grid: Collection of thread blocks")
    print("   - Block: Collection of threads (up to 1024 threads)")
    print("   - Thread: Individual execution unit")
    
    print("\n2. Memory Types:")
    print("   - Global Memory: Large, slow, accessible by all threads")
    print("   - Shared Memory: Fast, shared within a block")
    print("   - Local Memory: Per-thread private memory")
    
    print("\n3. Execution Model:")
    print("   - Kernel: Function that runs on GPU")
    print("   - Host: CPU code that launches kernels")
    print("   - Device: GPU that executes kernels")
    
    # Demonstrate CPU vector addition for comparison
    var size = 1000
    var a = UnsafePointer[Float32].alloc(size)
    var b = UnsafePointer[Float32].alloc(size)
    var result = UnsafePointer[Float32].alloc(size)
    
    # Initialize test data
    for i in range(size):
        a[i] = Float32(i)
        b[i] = Float32(i * 2)
    
    # CPU computation
    simple_vector_add_cpu(a, b, result, size)
    
    print("\nCPU Vector Addition Results:")
    print("a[0:5] =", a[0], a[1], a[2], a[3], a[4])
    print("b[0:5] =", b[0], b[1], b[2], b[3], b[4])
    print("result[0:5] =", result[0], result[1], result[2], result[3], result[4])
    
    a.free()
    b.free()
    result.free()

fn main():
    """Main function for GPU basics demonstration."""
    gpu_basics_demo()
```

### File: `16_thread_indexing.mojo`

**Execution:** `pixi run mojo 16_thread_indexing.mojo`

```mojo
from memory import UnsafePointer

fn simulate_gpu_thread_indexing():
    """Simulate GPU thread indexing patterns."""
    print("=== GPU Thread Indexing Simulation ===")
    
    # Simulate GPU grid configuration
    alias THREADS_PER_BLOCK = 8
    alias BLOCKS_PER_GRID = 4
    alias TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_PER_GRID
    
    print("Grid Configuration:")
    print("  Threads per block:", THREADS_PER_BLOCK)
    print("  Blocks per grid:", BLOCKS_PER_GRID)
    print("  Total threads:", TOTAL_THREADS)
    
    print("\nThread Index Calculation:")
    print("  thread_idx = block_idx * block_dim + thread_idx_within_block")
    print()
    
    # Simulate what each thread would compute
    for block_id in range(BLOCKS_PER_GRID):
        print("Block", block_id, ":")
        for thread_in_block in range(THREADS_PER_BLOCK):
            var global_thread_id = block_id * THREADS_PER_BLOCK + thread_in_block
            print("  Thread", thread_in_block, "-> Global ID:", global_thread_id)
    
    print("\nData Parallel Processing:")
    var data_size = 32
    var data = UnsafePointer[Float32].alloc(data_size)
    var result = UnsafePointer[Float32].alloc(data_size)
    
    # Initialize data
    for i in range(data_size):
        data[i] = Float32(i)
    
    # Simulate parallel processing (each "thread" processes one element)
    print("Simulating GPU kernel: result[i] = data[i] * 2 + 1")
    for block_id in range(BLOCKS_PER_GRID):
        for thread_in_block in range(THREADS_PER_BLOCK):
            var global_id = block_id * THREADS_PER_BLOCK + thread_in_block
            if global_id < data_size:
                result[global_id] = data[global_id] * 2 + 1
    
    print("Results (first 16 elements):")
    for i in range(16):
        print("  data[" + String(i) + "] =", data[i], "-> result[" + String(i) + "] =", result[i])
    
    data.free()
    result.free()

fn main():
    """Main function for thread indexing demonstration."""
    simulate_gpu_thread_indexing()
```

### File: `17_memory_patterns.mojo`

**Execution:** `pixi run mojo 17_memory_patterns.mojo`

```mojo
from memory import UnsafePointer

fn demonstrate_memory_coalescing():
    """Demonstrate memory access patterns for GPU optimization."""
    print("=== GPU Memory Access Patterns ===")
    
    var size = 16
    var threads_per_block = 4
    var data = UnsafePointer[Float32].alloc(size)
    
    # Initialize data
    for i in range(size):
        data[i] = Float32(i)
    
    print("Data array:", end=" ")
    for i in range(size):
        print(data[i], end=" ")
    print()
    
    print("\n1. Coalesced Access Pattern (GOOD):")
    print("   Adjacent threads access adjacent memory locations")
    print("   Thread 0 -> data[0], Thread 1 -> data[1], etc.")
    
    # Simulate coalesced access
    for block in range(size // threads_per_block):
        print("   Block", block, "threads access:", end=" ")
        for thread in range(threads_per_block):
            var index = block * threads_per_block + thread
            print("data[" + String(index) + "]", end=" ")
        print()
    
    print("\n2. Strided Access Pattern (BAD):")
    print("   Threads access memory with large strides")
    print("   Thread 0 -> data[0], Thread 1 -> data[4], etc.")
    
    # Simulate strided access
    var stride = 4
    for thread in range(threads_per_block):
        var index = thread * stride
        if index < size:
            print("   Thread", thread, "accesses data[" + String(index) + "]")
    
    print("\n3. Random Access Pattern (WORST):")
    print("   Threads access memory randomly - no pattern")
    var random_indices = List[Int]()
    random_indices.append(7)
    random_indices.append(2)
    random_indices.append(11)
    random_indices.append(5)
    
    for i in range(len(random_indices)):
        print("   Thread", i, "accesses data[" + String(random_indices[i]) + "]")
    
    print("\nMemory Coalescing Rules:")
    print("  + Adjacent threads should access adjacent memory")
    print("  + 32-thread warps should access 128-byte aligned blocks")
    print("  + Avoid bank conflicts in shared memory")
    print("  + Use SoA layout for better coalescing")
    
    data.free()

fn main():
    """Main function for memory patterns demonstration."""
    demonstrate_memory_coalescing()
```

### File: `18_broadcast_kernel_sim.mojo`

**Execution:** `pixi run mojo 18_broadcast_kernel_sim.mojo`

```mojo
from memory import UnsafePointer

struct Matrix2D:
    """Simple 2D matrix structure for GPU kernel simulation."""
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int
    
    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Float32].alloc(rows * cols)
    
    fn __del__(owned self):
        self.data.free()
    
    fn get(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]
    
    fn set(self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value
    
    fn print_matrix(self, name: String):
        print(name + ":")
        for i in range(self.rows):
            print("  [", end="")
            for j in range(self.cols):
                print(self.get(i, j), end="")
                if j < self.cols - 1:
                    print(", ", end="")
            print("]")

fn simulate_broadcast_add_kernel(output: Matrix2D, a: Matrix2D, b: Matrix2D):
    """Simulate GPU broadcasting kernel like the example provided."""
    print("=== Broadcasting Addition Kernel Simulation ===")
    
    print("Kernel Logic:")
    print("  output[row][col] = a[0][col] + b[row][0]")
    print("  Broadcasting (1,N) + (N,1) -> (N,N)")
    
    # Simulate GPU threads - each thread handles one output element
    for row in range(output.rows):
        for col in range(output.cols):
            # This simulates what each GPU thread would do:
            # thread_idx.y = row, thread_idx.x = col
            var a_val = a.get(0, col)      # a is (1, cols) - broadcast row
            var b_val = b.get(row, 0)      # b is (rows, 1) - broadcast column
            var result = a_val + b_val
            output.set(row, col, result)
            
            print("  Thread(" + String(row) + "," + String(col) + "): " + 
                  String(a_val) + " + " + String(b_val) + " = " + String(result))

fn demonstrate_gpu_broadcasting():
    """Demonstrate GPU broadcasting patterns."""
    print("=== GPU Broadcasting Demonstration ===")
    
    alias SIZE = 3
    
    # Create matrices for broadcasting
    var output = Matrix2D(SIZE, SIZE)     # (3, 3) output
    var a = Matrix2D(1, SIZE)            # (1, 3) - row vector
    var b = Matrix2D(SIZE, 1)            # (3, 1) - column vector
    
    # Initialize input matrices
    print("Initializing input matrices:")
    for i in range(SIZE):
        a.set(0, i, Float32(i))          # a = [0, 1, 2]
        b.set(i, 0, Float32(i))          # b = [0; 1; 2]
    
    a.print_matrix("Matrix A (1x3)")
    b.print_matrix("Matrix B (3x1)")
    
    # Simulate GPU kernel execution
    simulate_broadcast_add_kernel(output, a, b)
    
    print("\nResult:")
    output.print_matrix("Output (3x3)")
    
    print("\nGPU Execution Model:")
    print("  Grid dim: (1, 1, 1)")
    print("  Block dim: (3, 3, 1)")
    print("  Total threads: 9")
    print("  Each thread computes one output element")

fn main():
    """Main function for broadcasting demonstration."""
    demonstrate_gpu_broadcasting()
```

### File: `19_tensor_operations_gpu.mojo`

**Execution:** `pixi run mojo 19_tensor_operations_gpu.mojo`

```mojo
from memory import UnsafePointer

struct GPUTensorSim[dtype: DType]:
    """Simulated GPU tensor for automatic differentiation."""
    var data: UnsafePointer[Scalar[dtype]]
    var gradients: UnsafePointer[Scalar[dtype]]
    var shape: List[Int]
    var size: Int
    var requires_grad: Bool
    
    fn __init__(out self, shape: List[Int], requires_grad: Bool = False):
        self.shape = shape
        self.requires_grad = requires_grad
        
        # Calculate total size - fixed List iteration
        self.size = 1
        for i in range(len(shape)):
            self.size *= shape[i]
        
        # Allocate memory
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        if requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            # Initialize gradients to zero
            for i in range(self.size):
                self.gradients[i] = Scalar[dtype](0)
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor copying."""
        self.requires_grad = existing.requires_grad
        self.size = existing.size
        
        # Copy shape
        self.shape = List[Int]()
        for i in range(len(existing.shape)):
            self.shape.append(existing.shape[i])
        
        # Allocate and copy data
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]
        
        # Allocate and copy gradients if needed
        if self.requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            for i in range(self.size):
                self.gradients[i] = existing.gradients[i]
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()
    
    fn __del__(owned self):
        self.data.free()
        if self.requires_grad:
            self.gradients.free()
    
    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with constant value."""
        for i in range(self.size):
            self.data[i] = value
    
    fn simulate_gpu_elementwise_add(self, other: GPUTensorSim[dtype]) -> GPUTensorSim[dtype]:
        """Simulate GPU element-wise addition kernel."""
        print("Simulating GPU kernel: elementwise_add")
        
        var result = GPUTensorSim[dtype](self.shape, self.requires_grad or other.requires_grad)
        
        # Simulate GPU execution with thread blocks
        alias THREADS_PER_BLOCK = 256
        var num_blocks = (self.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        
        print("  Grid config: " + String(num_blocks) + " blocks, " + String(THREADS_PER_BLOCK) + " threads per block")
        print("  Total threads: " + String(num_blocks * THREADS_PER_BLOCK))
        print("  Elements to process: " + String(self.size))
        
        # Simulate what each GPU thread would do
        for block_id in range(num_blocks):
            for thread_id in range(THREADS_PER_BLOCK):
                var global_id = block_id * THREADS_PER_BLOCK + thread_id
                
                # Bounds check (important in GPU kernels)
                if global_id < self.size:
                    result.data[global_id] = self.data[global_id] + other.data[global_id]
        
        return result
    
    fn simulate_gpu_matrix_multiply(self, other: GPUTensorSim[dtype]) -> GPUTensorSim[dtype]:
        """Simulate GPU matrix multiplication kernel."""
        print("Simulating GPU kernel: matrix_multiply")
        
        # Assume 2D matrices for simplicity
        var m = self.shape[0]
        var k = self.shape[1]
        var n = other.shape[1]
        
        var result_shape = List[Int]()
        result_shape.append(m)
        result_shape.append(n)
        var result = GPUTensorSim[dtype](result_shape, self.requires_grad or other.requires_grad)
        
        print("  Matrix A: " + String(m) + "x" + String(k))
        print("  Matrix B: " + String(k) + "x" + String(n))
        print("  Result C: " + String(m) + "x" + String(n))
        
        # Simulate GPU thread grid (one thread per output element)
        print("  Thread grid: (" + String(m) + ", " + String(n) + ")")
        
        for row in range(m):
            for col in range(n):
                var sum: Scalar[dtype] = 0
                
                # Each thread computes one dot product
                for i in range(k):
                    var a_val = self.data[row * k + i]
                    var b_val = other.data[i * n + col]
                    sum += a_val * b_val
                
                result.data[row * n + col] = sum
        
        return result
    
    fn print_tensor(self, name: String):
        """Print tensor values."""
        print(name + " (shape: [", end="")
        for i in range(len(self.shape)):
            print(self.shape[i], end="")
            if i < len(self.shape) - 1:
                print(", ", end="")
        print("]):")
        
        if len(self.shape) == 1:
            # Vector
            print("  [", end="")
            for i in range(self.size):
                print(self.data[i], end="")
                if i < self.size - 1:
                    print(", ", end="")
            print("]")
        elif len(self.shape) == 2:
            # Matrix
            var rows = self.shape[0]
            var cols = self.shape[1]
            for i in range(rows):
                print("  [", end="")
                for j in range(cols):
                    print(self.data[i * cols + j], end="")
                    if j < cols - 1:
                        print(", ", end="")
                print("]")

fn demonstrate_gpu_tensor_operations():
    """Demonstrate GPU tensor operations for automatic differentiation."""
    print("=== GPU Tensor Operations for Automatic Differentiation ===")
    
    # Create test tensors
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    
    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(2)
    
    var tensor_a = GPUTensorSim[DType.float32](shape1, True)
    var tensor_b = GPUTensorSim[DType.float32](shape2, True)
    
    # Initialize with test data
    for i in range(tensor_a.size):
        tensor_a.data[i] = Float32(i + 1)
    
    for i in range(tensor_b.size):
        tensor_b.data[i] = Float32(i + 1) * 0.1
    
    tensor_a.print_tensor("Tensor A")
    tensor_b.print_tensor("Tensor B")
    
    # Simulate GPU matrix multiplication
    print("\nGPU Matrix Multiplication:")
    var result = tensor_a.simulate_gpu_matrix_multiply(tensor_b)
    result.print_tensor("Result C = A @ B")
    
    print("\nGPU Programming Benefits for AD:")
    print("  + Massive parallelization of tensor operations")
    print("  + Efficient gradient computation across thousands of parameters")
    print("  + Memory bandwidth optimization for large tensors")
    print("  + Concurrent forward and backward pass execution")
    print("  + Optimal for deep learning workloads")

fn main():
    """Main function for GPU tensor operations demonstration."""
    demonstrate_gpu_tensor_operations()
```

### File: `20_gpu_programming_complete.mojo`

**Execution:** `pixi run mojo 20_gpu_programming_complete.mojo`

```mojo
from memory import UnsafePointer

# Simulate the GPU programming concepts from the provided example
fn simulate_broadcast_add_complete():
    """Complete simulation of GPU broadcasting addition kernel."""
    print("=== Complete GPU Programming Demonstration ===")
    
    # Configuration matching the provided example
    alias SIZE = 2
    alias BLOCKS_PER_GRID = 1
    alias THREADS_PER_BLOCK_X = 3
    alias THREADS_PER_BLOCK_Y = 3
    
    print("GPU Configuration:")
    print("  Grid dimensions: (" + String(BLOCKS_PER_GRID) + ", 1, 1)")
    print("  Block dimensions: (" + String(THREADS_PER_BLOCK_X) + ", " + String(THREADS_PER_BLOCK_Y) + ", 1)")
    print("  Total threads: " + String(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y))
    
    # Create tensors
    var output = UnsafePointer[Float32].alloc(SIZE * SIZE)
    var a = UnsafePointer[Float32].alloc(SIZE)        # (1, SIZE) broadcasted
    var b = UnsafePointer[Float32].alloc(SIZE)        # (SIZE, 1) broadcasted
    
    # Initialize input tensors (matching the example)
    for i in range(SIZE):
        a[i] = Float32(i)      # a = [0, 1]
        b[i] = Float32(i)      # b = [0, 1]
    
    print("\nInput tensors:")
    print("  a (1x" + String(SIZE) + " broadcasted): [", end="")
    for i in range(SIZE):
        print(a[i], end="")
        if i < SIZE - 1:
            print(", ", end="")
    print("]")
    
    print("  b (" + String(SIZE) + "x1 broadcasted): [", end="")
    for i in range(SIZE):
        print(b[i], end="")
        if i < SIZE - 1:
            print(", ", end="")
    print("]")
    
    print("\nSimulating GPU kernel execution:")
    print("  Kernel: broadcast_add(output, a, b)")
    print("  Operation: output[row][col] = a[col] + b[row]")
    
    # Simulate GPU threads executing the kernel
    for thread_y in range(SIZE):  # Simulates thread_idx.y
        for thread_x in range(SIZE):  # Simulates thread_idx.x
            var row = thread_y
            var col = thread_x
            
            # This is what each GPU thread computes
            var a_val = a[col]        # a[j] where j = col (broadcasting)
            var b_val = b[row]        # b[i] where i = row (broadcasting)
            var result = a_val + b_val
            
            output[row * SIZE + col] = result
            
            print("    Thread(" + String(thread_y) + "," + String(thread_x) + 
                  "): a[" + String(col) + "] + b[" + String(row) + "] = " + 
                  String(a_val) + " + " + String(b_val) + " = " + String(result))
    
    print("\nOutput tensor (" + String(SIZE) + "x" + String(SIZE) + "):")
    for i in range(SIZE):
        print("  [", end="")
        for j in range(SIZE):
            print(output[i * SIZE + j], end="")
            if j < SIZE - 1:
                print(", ", end="")
        print("]")
    
    print("\nExpected output:")
    print("  [0, 1]")
    print("  [1, 2]")
    
    # Verify results
    var expected = UnsafePointer[Float32].alloc(SIZE * SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            expected[i * SIZE + j] = Float32(i + j)
    
    var correct = True
    for i in range(SIZE * SIZE):
        if abs(output[i] - expected[i]) > 0.001:
            correct = False
            break
    
    print("\nVerification:", "PASSED" if correct else "FAILED")
    
    # GPU Programming Concepts Summary
    print("\nGPU Programming Key Concepts:")
    print("1. Thread Indexing:")
    print("   - thread_idx.x, thread_idx.y: Thread position within block")
    print("   - block_idx.x, block_idx.y: Block position within grid")
    print("   - block_dim.x, block_dim.y: Block dimensions")
    
    print("\n2. Memory Access:")
    print("   - Coalesced access: Adjacent threads access adjacent memory")
    print("   - Broadcasting: Efficient reuse of data across threads")
    print("   - Global memory: Accessible by all threads")
    
    print("\n3. Execution Model:")
    print("   - SIMT: Single Instruction, Multiple Threads")
    print("   - Warps: Groups of 32 threads execute together")
    print("   - Synchronization: __syncthreads() for block-level sync")
    
    print("\n4. Automatic Differentiation Applications:")
    print("   - Element-wise operations: Perfect for GPU parallelization")
    print("   - Matrix operations: Utilize shared memory and tiling")
    print("   - Gradient computation: Parallel backward pass")
    print("   - Parameter updates: Vectorized optimizer steps")
    
    # Cleanup
    output.free()
    a.free()
    b.free()
    expected.free()

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Complete GPU programming demonstration."""
    simulate_broadcast_add_complete()
```

### Expected Output for `20_gpu_programming_complete.mojo`
```
=== Complete GPU Programming Demonstration ===
GPU Configuration:
  Grid dimensions: (1, 1, 1)
  Block dimensions: (3, 3, 1)
  Total threads: 9

Input tensors:
  a (1x2 broadcasted): [0, 1]
  b (2x1 broadcasted): [0, 1]

Simulating GPU kernel execution:
  Kernel: broadcast_add(output, a, b)
  Operation: output[row][col] = a[col] + b[row]
    Thread(0,0): a[0] + b[0] = 0 + 0 = 0
    Thread(0,1): a[1] + b[0] = 1 + 0 = 1
    Thread(1,0): a[0] + b[1] = 0 + 1 = 1
    Thread(1,1): a[1] + b[1] = 1 + 1 = 2

Output tensor (2x2):
  [0, 1]
  [1, 2]

Expected output:
  [0, 1]
  [1, 2]

Verification: PASSED

GPU Programming Key Concepts:
1. Thread Indexing:
   - thread_idx.x, thread_idx.y: Thread position within block
   - block_idx.x, block_idx.y: Block position within grid
   - block_dim.x, block_dim.y: Block dimensions

2. Memory Access:
   - Coalesced access: Adjacent threads access adjacent memory
   - Broadcasting: Efficient reuse of data across threads
   - Global memory: Accessible by all threads

3. Execution Model:
   - SIMT: Single Instruction, Multiple Threads
   - Warps: Groups of 32 threads execute together
   - Synchronization: __syncthreads() for block-level sync

4. Automatic Differentiation Applications:
   - Element-wise operations: Perfect for GPU parallelization
   - Matrix operations: Utilize shared memory and tiling
   - Gradient computation: Parallel backward pass
   - Parameter updates: Vectorized optimizer steps
```

<!-- START: Part 0.5 - SIMD and Vectorization Basics -->
