# 0.3 Memory Layout Strategies (AoS vs SoA)

# Mojo Memory Layout Strategies - Part 0.3

This section demonstrates memory layout patterns crucial for high-performance tensor operations and automatic differentiation. Understanding Array of Structs (AoS) vs Struct of Arrays (SoA) patterns is essential for GPU optimization and SIMD vectorization.

---

## 0.3 Memory Layout Strategies (AoS vs SoA)

### File: `11_aos_pattern.mojo`

**Execution:** `pixi run mojo 11_aos_pattern.mojo`

```mojo
from memory import UnsafePointer

struct Particle:
    """Array of Structs (AoS) pattern - traditional object-oriented approach."""
    var x: Float32
    var y: Float32
    var z: Float32
    var vx: Float32
    var vy: Float32
    var vz: Float32
    var mass: Float32
    
    fn __init__(out self, x: Float32, y: Float32, z: Float32, 
               vx: Float32, vy: Float32, vz: Float32, mass: Float32):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor required for struct copying."""
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z
        self.vx = existing.vx
        self.vy = existing.vy
        self.vz = existing.vz
        self.mass = existing.mass
    
    fn update_position(mut self, dt: Float32):
        """Update particle position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
    
    fn kinetic_energy(self) -> Float32:
        """Calculate kinetic energy of particle."""
        var v_squared = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        return 0.5 * self.mass * v_squared

struct ParticleSystemAoS:
    """Particle system using Array of Structs layout."""
    var particles: UnsafePointer[Particle]
    var count: Int
    
    fn __init__(out self, count: Int):
        self.count = count
        self.particles = UnsafePointer[Particle].alloc(count)
        
        # Initialize particles with sample data
        for i in range(count):
            var x = Float32(i) * 0.1
            var y = Float32(i) * 0.2
            var z = Float32(i) * 0.3
            var vx = Float32(1.0)
            var vy = Float32(2.0)
            var vz = Float32(3.0)
            var mass = Float32(1.0)
            self.particles[i] = Particle(x, y, z, vx, vy, vz, mass)
    
    fn __del__(owned self):
        self.particles.free()
    
    fn update_all_positions(self, dt: Float32):
        """Update all particle positions - demonstrates scattered memory access."""
        print("AoS: Updating", self.count, "particle positions")
        for i in range(self.count):
            self.particles[i].update_position(dt)
    
    fn total_kinetic_energy(self) -> Float32:
        """Calculate total kinetic energy - demonstrates mixed memory access."""
        var total: Float32 = 0.0
        for i in range(self.count):
            total += self.particles[i].kinetic_energy()
        return total
    
    fn print_sample(self, num_samples: Int):
        """Print sample of particles."""
        var samples = min(num_samples, self.count)
        print("AoS Sample particles:")
        for i in range(samples):
            # Access fields directly to avoid copying
            print("  Particle", i, ": pos(", self.particles[i].x, ",", self.particles[i].y, ",", self.particles[i].z, 
                  ") vel(", self.particles[i].vx, ",", self.particles[i].vy, ",", self.particles[i].vz, ") mass:", self.particles[i].mass)

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b

fn aos_demo():
    """Demonstrate Array of Structs memory layout."""
    print("=== Array of Structs (AoS) Pattern ===")
    
    var system = ParticleSystemAoS(1000)
    
    # Memory layout visualization
    print("Memory Layout: [x,y,z,vx,vy,vz,mass][x,y,z,vx,vy,vz,mass][...]")
    print("Characteristics:")
    print("  - Good: Object-oriented, cache-friendly for single particle operations")
    print("  - Bad: Poor vectorization, scattered access for bulk operations")
    
    system.print_sample(3)
    
    # Simulate operations
    var dt: Float32 = 0.1
    system.update_all_positions(dt)
    
    var total_energy = system.total_kinetic_energy()
    print("Total kinetic energy:", total_energy)

fn main():
    """Main function for AoS demonstration."""
    aos_demo()
```

### File: `12_soa_pattern.mojo`

**Execution:** `pixi run mojo 12_soa_pattern.mojo`

```mojo
from memory import UnsafePointer

struct ParticleSystemSoA:
    """Particle system using Struct of Arrays layout for optimal SIMD performance."""
    # Separate arrays for each component (SoA pattern)
    var x: UnsafePointer[Float32]
    var y: UnsafePointer[Float32]  
    var z: UnsafePointer[Float32]
    var vx: UnsafePointer[Float32]
    var vy: UnsafePointer[Float32]
    var vz: UnsafePointer[Float32]
    var mass: UnsafePointer[Float32]
    var count: Int
    
    fn __init__(out self, count: Int):
        self.count = count
        
        # Allocate separate contiguous arrays for each component
        self.x = UnsafePointer[Float32].alloc(count)
        self.y = UnsafePointer[Float32].alloc(count)
        self.z = UnsafePointer[Float32].alloc(count)
        self.vx = UnsafePointer[Float32].alloc(count)
        self.vy = UnsafePointer[Float32].alloc(count)
        self.vz = UnsafePointer[Float32].alloc(count)
        self.mass = UnsafePointer[Float32].alloc(count)
        
        # Initialize with sample data
        for i in range(count):
            self.x[i] = Float32(i) * 0.1
            self.y[i] = Float32(i) * 0.2
            self.z[i] = Float32(i) * 0.3
            self.vx[i] = Float32(1.0)
            self.vy[i] = Float32(2.0)
            self.vz[i] = Float32(3.0)
            self.mass[i] = Float32(1.0)
    
    fn __del__(owned self):
        self.x.free()
        self.y.free()
        self.z.free()
        self.vx.free()
        self.vy.free()
        self.vz.free()
        self.mass.free()
    
    fn update_all_positions(self, dt: Float32):
        """Update all positions - demonstrates optimal vectorized memory access."""
        print("SoA: Updating", self.count, "particle positions with SIMD potential")
        
        # This loop is highly vectorizable because we access contiguous memory
        for i in range(self.count):
            self.x[i] += self.vx[i] * dt
            self.y[i] += self.vy[i] * dt
            self.z[i] += self.vz[i] * dt
    
    fn update_positions_simd(self, dt: Float32):
        """Vectorized position update using SIMD operations."""
        print("SoA: SIMD-optimized position update")
        
        # Process in chunks of 4 for SIMD operations
        var simd_count = (self.count // 4) * 4
        var dt_vec = SIMD[DType.float32, 4](dt)
        
        for i in range(0, simd_count, 4):
            # Manual load from memory using loop
            var x_vals = SIMD[DType.float32, 4](0)
            var vx_vals = SIMD[DType.float32, 4](0)
            for j in range(4):
                x_vals[j] = self.x[i + j]
                vx_vals[j] = self.vx[i + j]
            
            # Vectorized computation: x += vx * dt
            var new_x = x_vals + vx_vals * dt_vec
            
            # Manual store back to memory
            for j in range(4):
                self.x[i + j] = new_x[j]
            
            # Repeat for y and z
            var y_vals = SIMD[DType.float32, 4](0)
            var vy_vals = SIMD[DType.float32, 4](0)
            for j in range(4):
                y_vals[j] = self.y[i + j]
                vy_vals[j] = self.vy[i + j]
            
            var new_y = y_vals + vy_vals * dt_vec
            for j in range(4):
                self.y[i + j] = new_y[j]
            
            var z_vals = SIMD[DType.float32, 4](0)
            var vz_vals = SIMD[DType.float32, 4](0)
            for j in range(4):
                z_vals[j] = self.z[i + j]
                vz_vals[j] = self.vz[i + j]
            
            var new_z = z_vals + vz_vals * dt_vec
            for j in range(4):
                self.z[i + j] = new_z[j]
        
        # Handle remaining elements
        for i in range(simd_count, self.count):
            self.x[i] += self.vx[i] * dt
            self.y[i] += self.vy[i] * dt
            self.z[i] += self.vz[i] * dt
    
    fn total_kinetic_energy_simd(self) -> Float32:
        """Calculate total kinetic energy using SIMD operations."""
        var total: Float32 = 0.0
        var simd_count = (self.count // 4) * 4
        
        # SIMD processing
        var energy_sum = SIMD[DType.float32, 4](0.0)
        var half_vec = SIMD[DType.float32, 4](0.5)
        
        for i in range(0, simd_count, 4):
            # Manual load
            var vx_vals = SIMD[DType.float32, 4](0)
            var vy_vals = SIMD[DType.float32, 4](0)
            var vz_vals = SIMD[DType.float32, 4](0)
            var mass_vals = SIMD[DType.float32, 4](0)
            
            for j in range(4):
                vx_vals[j] = self.vx[i + j]
                vy_vals[j] = self.vy[i + j]
                vz_vals[j] = self.vz[i + j]
                mass_vals[j] = self.mass[i + j]
            
            var v_squared = vx_vals * vx_vals + vy_vals * vy_vals + vz_vals * vz_vals
            var kinetic = half_vec * mass_vals * v_squared
            energy_sum += kinetic
        
        # Sum the SIMD vector
        for i in range(4):
            total += energy_sum[i]
        
        # Handle remaining elements
        for i in range(simd_count, self.count):
            var v_sq = self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i] + self.vz[i] * self.vz[i]
            total += 0.5 * self.mass[i] * v_sq
        
        return total
    
    fn print_sample(self, num_samples: Int):
        """Print sample of particles."""
        var samples = min(num_samples, self.count)
        print("SoA Sample particles:")
        for i in range(samples):
            print("  Particle", i, ": pos(", self.x[i], ",", self.y[i], ",", self.z[i], 
                  ") vel(", self.vx[i], ",", self.vy[i], ",", self.vz[i], ") mass:", self.mass[i])

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b

fn soa_demo():
    """Demonstrate Struct of Arrays memory layout."""
    print("=== Struct of Arrays (SoA) Pattern ===")
    
    var system = ParticleSystemSoA(1000)
    
    # Memory layout visualization
    print("Memory Layout: x:[x0,x1,x2...] y:[y0,y1,y2...] vx:[vx0,vx1,vx2...]")
    print("Characteristics:")
    print("  - Good: Excellent vectorization, cache-friendly for bulk operations")
    print("  - Bad: Less intuitive, scattered access for single particle operations")
    
    system.print_sample(3)
    
    # Simulate operations
    var dt: Float32 = 0.1
    
    # Compare regular vs SIMD operations
    print("\nRegular update:")
    system.update_all_positions(dt)
    
    print("SIMD-optimized update:")
    system.update_positions_simd(dt)
    
    var total_energy = system.total_kinetic_energy_simd()
    print("Total kinetic energy (SIMD):", total_energy)

fn main():
    """Main function for SoA demonstration."""
    soa_demo()
```

### File: `13_performance_comparison.mojo`

**Execution:** `pixi run mojo 13_performance_comparison.mojo`

```mojo
from memory import UnsafePointer

# Tensor-like structure using SoA for optimal GPU performance
struct TensorSoA[dtype: DType]:
    """GPU-optimized tensor using Struct of Arrays layout."""
    var data: UnsafePointer[Scalar[dtype]]
    var gradients: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var ndim: Int
    var size: Int
    var requires_grad: Bool
    
    fn __init__(out self, shape_list: List[Int], requires_grad: Bool = False):
        self.ndim = len(shape_list)
        self.requires_grad = requires_grad
        
        # Allocate shape and stride arrays
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        
        # Calculate size and strides
        self.size = 1
        for i in range(self.ndim):
            self.shape[i] = shape_list[i]
            self.size *= shape_list[i]
        
        # Calculate strides (row-major order)
        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
        
        # Allocate data arrays
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        if self.requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            # Initialize gradients to zero
            for i in range(self.size):
                self.gradients[i] = Scalar[dtype](0)
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor copying."""
        self.ndim = existing.ndim
        self.size = existing.size
        self.requires_grad = existing.requires_grad
        
        # Allocate and copy shape
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.strides[i] = existing.strides[i]
        
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
        self.shape.free()
        self.strides.free()
    
    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with a constant value."""
        for i in range(self.size):
            self.data[i] = value
    
    fn add_elementwise(self, other: TensorSoA[dtype]) -> TensorSoA[dtype]:
        """Element-wise addition optimized for vectorization."""
        # Create result tensor
        var shape_list = List[Int]()
        for i in range(self.ndim):
            shape_list.append(self.shape[i])
        
        var result = TensorSoA[dtype](shape_list, self.requires_grad or other.requires_grad)
        
        # Vectorized addition using manual SIMD operations
        var simd_size = 4
        var simd_count = (self.size // simd_size) * simd_size
        
        # Process in SIMD chunks
        for i in range(0, simd_count, simd_size):
            # Manual load from memory
            var a_vals = SIMD[dtype, 4](0)
            var b_vals = SIMD[dtype, 4](0)
            for j in range(4):
                a_vals[j] = self.data[i + j]
                b_vals[j] = other.data[i + j]
            
            # SIMD addition
            var result_vals = a_vals + b_vals
            
            # Manual store to memory
            for j in range(4):
                result.data[i + j] = result_vals[j]
        
        # Handle remaining elements
        for i in range(simd_count, self.size):
            result.data[i] = self.data[i] + other.data[i]
        
        return result
    
    fn print_info(self):
        """Print tensor information."""
        print("Tensor info:")
        print("  Shape: [", end="")
        for i in range(self.ndim):
            print(self.shape[i], end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        print("  Size:", self.size)
        print("  Requires grad:", self.requires_grad)
        print("  First few elements:", self.data[0], self.data[1], self.data[2])

fn memory_layout_comparison():
    """Compare different memory layouts for tensor operations."""
    print("=== Memory Layout Performance Comparison ===")
    
    # Create tensors with different patterns
    var shape_list = List[Int]()
    shape_list.append(1000)
    shape_list.append(100)
    
    print("\nCreating tensors for performance comparison...")
    
    # SoA-style tensor optimized for vectorization
    var tensor_a = TensorSoA[DType.float32](shape_list, True)
    var tensor_b = TensorSoA[DType.float32](shape_list, True)
    
    # Fill with test data
    tensor_a.fill(2.5)
    tensor_b.fill(1.5)
    
    print("Tensor A:")
    tensor_a.print_info()
    
    print("Tensor B:")
    tensor_b.print_info()
    
    # Perform vectorized operations
    print("\nPerforming element-wise addition...")
    var result = tensor_a.add_elementwise(tensor_b)
    
    print("Result tensor:")
    result.print_info()
    
    print("\nMemory Layout Analysis:")
    print("SoA Layout Benefits:")
    print("  ✓ Excellent cache locality for bulk operations")
    print("  ✓ SIMD vectorization friendly")
    print("  ✓ GPU memory coalescing")
    print("  ✓ Optimal for parallel processing")
    
    print("SoA Layout Considerations:")
    print("  - More complex indexing for individual elements")
    print("  - Requires careful memory management")
    print("  - Less intuitive than object-oriented approach")

fn main():
    """Main function for performance comparison."""
    memory_layout_comparison()
```

### File: `14_layout_patterns_complete.mojo`

**Execution:** `pixi run mojo 14_layout_patterns_complete.mojo`

```mojo
from memory import UnsafePointer

# AoS Pattern Example
struct Point3D:
    var x: Float32
    var y: Float32 
    var z: Float32
    
    fn __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

# SoA Pattern Example  
struct Points3DSoA:
    var x: UnsafePointer[Float32]
    var y: UnsafePointer[Float32]
    var z: UnsafePointer[Float32]
    var count: Int
    
    fn __init__(out self, count: Int):
        self.count = count
        self.x = UnsafePointer[Float32].alloc(count)
        self.y = UnsafePointer[Float32].alloc(count)
        self.z = UnsafePointer[Float32].alloc(count)
    
    fn __del__(owned self):
        self.x.free()
        self.y.free()
        self.z.free()
    
    fn set_point(self, index: Int, x: Float32, y: Float32, z: Float32):
        self.x[index] = x
        self.y[index] = y
        self.z[index] = z
    
    fn distance_sum_simd(self) -> Float32:
        """Calculate sum of distances from origin using SIMD."""
        var total: Float32 = 0.0
        var simd_count = (self.count // 4) * 4
        
        var sum_vec = SIMD[DType.float32, 4](0.0)
        
        for i in range(0, simd_count, 4):
            # Manual load from memory
            var x_vals = SIMD[DType.float32, 4](0)
            var y_vals = SIMD[DType.float32, 4](0)
            var z_vals = SIMD[DType.float32, 4](0)
            
            for j in range(4):
                x_vals[j] = self.x[i + j]
                y_vals[j] = self.y[i + j]
                z_vals[j] = self.z[i + j]
            
            var dist_sq = x_vals * x_vals + y_vals * y_vals + z_vals * z_vals
            sum_vec += dist_sq
        
        # Sum SIMD vector elements
        for i in range(4):
            total += sum_vec[i]
        
        # Handle remaining elements
        for i in range(simd_count, self.count):
            var dist_sq = self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i]
            total += dist_sq
        
        return total

fn complete_layout_demo():
    """Complete demonstration of memory layout patterns."""
    print("=== Complete Memory Layout Patterns Demo ===")
    
    var num_points = 1000
    
    # AoS approach
    print("Array of Structs (AoS):")
    print("  Memory: [x,y,z][x,y,z][x,y,z]...")
    print("  Use case: Individual point operations")
    
    var aos_points = UnsafePointer[Point3D].alloc(num_points)
    for i in range(num_points):
        aos_points[i] = Point3D(Float32(i), Float32(i) * 2, Float32(i) * 3)
    
    print("  Sample AoS point:", aos_points[0].x, aos_points[0].y, aos_points[0].z)
    aos_points.free()
    
    # SoA approach
    print("\nStruct of Arrays (SoA):")
    print("  Memory: x:[x0,x1,x2...] y:[y0,y1,y2...] z:[z0,z1,z2...]")
    print("  Use case: Bulk vectorized operations")
    
    var soa_points = Points3DSoA(num_points)
    for i in range(num_points):
        soa_points.set_point(i, Float32(i), Float32(i) * 2, Float32(i) * 3)
    
    print("  Sample SoA point:", soa_points.x[0], soa_points.y[0], soa_points.z[0])
    
    var distance_sum = soa_points.distance_sum_simd()
    print("  SIMD distance sum:", distance_sum)
    
    print("\nPerformance Characteristics:")
    print("AoS Benefits:")
    print("  + Cache-friendly for single object operations")
    print("  + Intuitive object-oriented design")
    print("  + Good spatial locality for related data")
    
    print("SoA Benefits:")
    print("  + Excellent vectorization (SIMD/GPU)")
    print("  + Cache-friendly for bulk operations")
    print("  + Memory bandwidth optimization")
    print("  + Parallel processing friendly")
    
    print("\nTensor Framework Choice: SoA")
    print("Reason: Automatic differentiation requires bulk operations on gradients,")
    print("        making SoA optimal for GPU acceleration and SIMD vectorization.")

fn main():
    """Complete demonstration of memory layout patterns."""
    complete_layout_demo()
```

### Expected Output for `14_layout_patterns_complete.mojo`
```
=== Complete Memory Layout Patterns Demo ===
Array of Structs (AoS):
  Memory: [x,y,z][x,y,z][x,y,z]...
  Use case: Individual point operations
  Sample AoS point: 0.0 0.0 0.0

Struct of Arrays (SoA):
  Memory: x:[x0,x1,x2...] y:[y0,y1,y2...] z:[z0,z1,z2...]
  Use case: Bulk vectorized operations
  Sample SoA point: 0.0 0.0 0.0
  SIMD distance sum: [calculated value]

Performance Characteristics:
AoS Benefits:
  + Cache-friendly for single object operations
  + Intuitive object-oriented design
  + Good spatial locality for related data

SoA Benefits:
  + Excellent vectorization (SIMD/GPU)
  + Cache-friendly for bulk operations
  + Memory bandwidth optimization
  + Parallel processing friendly

Tensor Framework Choice: SoA
Reason: Automatic differentiation requires bulk operations on gradients,
        making SoA optimal for GPU acceleration and SIMD vectorization.
```

---

## Key Memory Layout Patterns Summary

**1. Array of Structs (AoS)** - Traditional object-oriented layout
- **Memory pattern**: `[x,y,z,vx,vy,vz][x,y,z,vx,vy,vz][...]`
- **Best for**: Single object operations, intuitive code structure
- **Cache behavior**: Good locality for related fields of same object

**2. Struct of Arrays (SoA)** - Performance-optimized layout
- **Memory pattern**: `x:[x0,x1,x2...] y:[y0,y1,y2...] vx:[vx0,vx1,vx2...]`
- **Best for**: Bulk operations, SIMD vectorization, GPU computing
- **Cache behavior**: Excellent for processing same field across many objects

**3. SIMD Optimization** - Leveraging SoA for vectorization
- **Vector operations**: Process 4/8/16 elements simultaneously
- **Memory coalescing**: Adjacent memory access for optimal bandwidth
- **GPU readiness**: Perfect for CUDA-style parallel kernels

**4. Tensor Framework Design** - Why SoA wins for AD
- **Gradient computations**: Bulk operations on gradient arrays
- **Backpropagation**: Vector operations across many tensors
- **GPU acceleration**: Memory access patterns optimized for parallel computing
