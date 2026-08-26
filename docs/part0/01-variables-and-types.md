# 0.1 Variable Declaration and Type System

# Part 0: Mojo Programming Fundamentals

## 0.1 Variable Declaration and Type System

### File: `01_basic_variables.mojo`

**Execution:** `pixi run mojo 01_basic_variables.mojo`

```mojo
from memory import UnsafePointer
from collections import List

fn basic_variables_demo():
    """Demonstrate basic Mojo variable declaration patterns."""
    print("=== Basic Variable Declaration ===")
    
    # Basic numeric types
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828
    
    print("Integer x:", x)
    print("Float32 y:", y)
    print("Float64 z:", z)
    
    # Type inference
    var a = 10        # Inferred as Int
    var b = 5.5       # Inferred as Float64
    var c = True      # Inferred as Bool
    
    print("Inferred types - a:", a, "b:", b, "c:", c)
    
    # Constants with alias
    alias PI = 3.14159265359
    alias MAX_SIZE = 1024
    
    print("Constants - PI:", PI, "MAX_SIZE:", MAX_SIZE)
    
    # Built-in math functions (no import needed)
    var sqrt_result = pow(4.0, 0.5)  # Square root using pow
    var abs_result = abs(-42)        # Absolute value
    var max_result = max(10, 20)     # Maximum of two values
    var min_result = min(10, 20)     # Minimum of two values
    
    print("Built-in math functions:")
    print("  sqrt(4):", sqrt_result)
    print("  abs(-42):", abs_result) 
    print("  max(10,20):", max_result)
    print("  min(10,20):", min_result)

fn main():
    """Main function for basic variables demonstration."""
    basic_variables_demo()
```

### File: `02_advanced_types.mojo`

**Execution:** `pixi run mojo 02_advanced_types.mojo`

```mojo
fn advanced_types_demo():
    """Demonstrate advanced Mojo type features."""
    print("\n=== Advanced Type Features ===")
    
    # Parametric types
    alias dtype = DType.float32
    var value: Scalar[dtype] = 42.0
    print("Parametric type value:", value)
    
    # SIMD types for vectorization
    var simd_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("SIMD vector:", simd_vec)
    
    # Operations on SIMD vectors
    var simd_squared = simd_vec * simd_vec
    print("SIMD squared:", simd_squared)
    
    # SIMD power operation using built-in pow
    var simd_cubed = pow(simd_vec, 3.0)
    print("SIMD cubed:", simd_cubed)
    
    # List collection
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print("List size:", len(numbers))
    print("List elements:", numbers[0], numbers[1], numbers[2])

fn main():
    """Main function for advanced types demonstration."""
    advanced_types_demo()
```

### File: `03_memory_patterns.mojo`

**Execution:** `pixi run mojo 03_memory_patterns.mojo`

```mojo
from memory import UnsafePointer
from collections import List

fn basic_variables_demo():
    """Demonstrate basic Mojo variable declaration patterns."""
    print("=== Basic Variable Declaration ===")
    
    # Basic numeric types
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828
    
    print("Integer x:", x)
    print("Float32 y:", y)
    print("Float64 z:", z)
    
    # Type inference
    var a = 10        # Inferred as Int
    var b = 5.5       # Inferred as Float64
    var c = True      # Inferred as Bool
    
    print("Inferred types - a:", a, "b:", b, "c:", c)
    
    # Constants with alias
    alias PI = 3.14159265359
    alias MAX_SIZE = 1024
    
    print("Constants - PI:", PI, "MAX_SIZE:", MAX_SIZE)
    
    # Built-in math functions (no import needed)
    var sqrt_result = pow(4.0, 0.5)  # Square root using pow
    var abs_result = abs(-42)        # Absolute value
    var max_result = max(10, 20)     # Maximum of two values
    var min_result = min(10, 20)     # Minimum of two values
    
    print("Built-in math functions:")
    print("  sqrt(4):", sqrt_result)
    print("  abs(-42):", abs_result) 
    print("  max(10,20):", max_result)
    print("  min(10,20):", min_result)

fn advanced_types_demo():
    """Demonstrate advanced Mojo type features."""
    print("\n=== Advanced Type Features ===")
    
    # Parametric types
    alias dtype = DType.float32
    var value: Scalar[dtype] = 42.0
    print("Parametric type value:", value)
    
    # SIMD types for vectorization
    var simd_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("SIMD vector:", simd_vec)
    
    # Operations on SIMD vectors
    var simd_squared = simd_vec * simd_vec
    print("SIMD squared:", simd_squared)
    
    # SIMD power operation using built-in pow
    var simd_cubed = pow(simd_vec, 3.0)
    print("SIMD cubed:", simd_cubed)
    
    # List collection
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print("List size:", len(numbers))
    print("List elements:", numbers[0], numbers[1], numbers[2])

fn memory_patterns_demo():
    """Demonstrate memory management patterns in Mojo."""
    print("\n=== Memory Management Patterns ===")
    
    # Heap allocation with UnsafePointer
    var size = 5
    var heap_ptr = UnsafePointer[Int].alloc(size)
    
    # Initialize heap memory
    for i in range(size):
        heap_ptr[i] = i * i
    
    print("Heap values:")
    for i in range(size):
        var i_str: String = String(i)
        var value_str: String = String(heap_ptr[i])
        var message: String = "  heap_ptr[" + i_str + "] = " + value_str
        print(message)
    
    # Manual cleanup (RAII pattern will automate this later)
    heap_ptr.free()
    
    print("Memory management demo completed")

fn main():
    """Complete demonstration of all basic Mojo patterns."""
    basic_variables_demo()
    advanced_types_demo()
    memory_patterns_demo()
```

### File: `04_complete_basics.mojo`

**Execution:** `pixi run mojo 04_complete_basics.mojo`

```mojo
from memory import UnsafePointer
from collections import List

fn basic_variables_demo():
    """Demonstrate basic Mojo variable declaration patterns."""
    print("=== Basic Variable Declaration ===")
    
    # Basic numeric types
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828
    
    print("Integer x:", x)
    print("Float32 y:", y)
    print("Float64 z:", z)
    
    # Type inference
    var a = 10        # Inferred as Int
    var b = 5.5       # Inferred as Float64
    var c = True      # Inferred as Bool
    
    print("Inferred types - a:", a, "b:", b, "c:", c)
    
    # Constants with alias
    alias PI = 3.14159265359
    alias MAX_SIZE = 1024
    
    print("Constants - PI:", PI, "MAX_SIZE:", MAX_SIZE)

fn advanced_types_demo():
    """Demonstrate advanced Mojo type features."""
    print("\n=== Advanced Type Features ===")
    
    # Parametric types
    alias dtype = DType.float32
    var value: Scalar[dtype] = 42.0
    print("Parametric type value:", value)
    
    # SIMD types for vectorization
    var simd_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("SIMD vector:", simd_vec)
    
    # Operations on SIMD vectors
    var simd_squared = simd_vec * simd_vec
    print("SIMD squared:", simd_squared)
    
    # List collection
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print("List size:", len(numbers))
    print("List elements:", numbers[0], numbers[1], numbers[2])

fn memory_patterns_demo():
    """Demonstrate memory management patterns in Mojo."""
    print("=== Memory Management Patterns ===")
    
    # Heap allocation with UnsafePointer
    var size = 5
    var heap_ptr = UnsafePointer[Int].alloc(size)
    
    # Initialize heap memory
    for i in range(size):
        heap_ptr[i] = i * i
    
    print("Heap values:")
    for i in range(size):
        print("  heap_ptr[", i, "] =", heap_ptr[i])
    
    # Manual cleanup (RAII pattern will automate this later)
    heap_ptr.free()
    
    print("Memory management demo completed")

fn main():
    """Complete demonstration of all basic Mojo patterns."""
    basic_variables_demo()
    advanced_types_demo()
    memory_patterns_demo()
```

### File: `05_standalone_complete.mojo`

**Execution:** `pixi run mojo 05_standalone_complete.mojo`

```mojo
from memory import UnsafePointer
from collections import List

fn basic_variables_demo():
    """Demonstrate basic Mojo variable declaration patterns."""
    print("=== Basic Variable Declaration ===")
    
    # Basic numeric types
    var x: Int = 42
    var y: Float32 = 3.14159
    var z: Float64 = 2.71828
    
    print("Integer x:", x)
    print("Float32 y:", y)
    print("Float64 z:", z)
    
    # Type inference
    var a = 10        # Inferred as Int
    var b = 5.5       # Inferred as Float64
    var c = True      # Inferred as Bool
    
    print("Inferred types - a:", a, "b:", b, "c:", c)
    
    # Constants with alias
    alias PI = 3.14159265359
    alias MAX_SIZE = 1024
    
    print("Constants - PI:", PI, "MAX_SIZE:", MAX_SIZE)
    
    # Built-in math functions (no import needed)
    var sqrt_result = pow(4.0, 0.5)  # Square root using pow
    var abs_result = abs(-42)        # Absolute value
    var max_result = max(10, 20)     # Maximum of two values
    var min_result = min(10, 20)     # Minimum of two values
    
    print("Built-in math functions:")
    print("  sqrt(4):", sqrt_result)
    print("  abs(-42):", abs_result) 
    print("  max(10,20):", max_result)
    print("  min(10,20):", min_result)

fn advanced_types_demo():
    """Demonstrate advanced Mojo type features."""
    print("\n=== Advanced Type Features ===")
    
    # Parametric types
    alias dtype = DType.float32
    var value: Scalar[dtype] = 42.0
    print("Parametric type value:", value)
    
    # SIMD types for vectorization
    var simd_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("SIMD vector:", simd_vec)
    
    # Operations on SIMD vectors
    var simd_squared = simd_vec * simd_vec
    print("SIMD squared:", simd_squared)
    
    # SIMD power operation using built-in pow
    var simd_cubed = pow(simd_vec, 3.0)
    print("SIMD cubed:", simd_cubed)
    
    # List collection
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print("List size:", len(numbers))
    print("List elements:", numbers[0], numbers[1], numbers[2])

fn memory_patterns_demo():
    """Demonstrate memory management patterns in Mojo."""
    print("\n=== Memory Management Patterns ===")
    
    # Heap allocation with UnsafePointer
    var size = 5
    var heap_ptr = UnsafePointer[Int].alloc(size)
    
    # Initialize heap memory
    for i in range(size):
        heap_ptr[i] = i * i
    
    print("Heap values:")
    for i in range(size):
        var i_str: String = String(i)
        var value_str: String = String(heap_ptr[i])
        var message: String = "  heap_ptr[" + i_str + "] = " + value_str
        print(message)
    
    # Manual cleanup (RAII pattern will automate this later)
    heap_ptr.free()
    
    print("Memory management demo completed")

fn main():
    """Complete demonstration of all basic Mojo patterns."""
    basic_variables_demo()
    advanced_types_demo()
    memory_patterns_demo()
```

### Expected Output for `05_standalone_complete.mojo`
```
=== Basic Variable Declaration ===
Integer x: 42
Float32 y: 3.14159
Float64 z: 2.71828
Inferred types - a: 10 b: 5.5 c: True
Constants - PI: 3.14159265359 MAX_SIZE: 1024
Built-in math functions:
  sqrt(4): 2.0
  abs(-42): 42
  max(10,20): 20
  min(10,20): 10

=== Advanced Type Features ===
Parametric type value: 42.0
SIMD vector: [1.0, 2.0, 3.0, 4.0]
SIMD squared: [1.0, 4.0, 9.0, 16.0]
SIMD cubed: [1.0, 8.0, 27.0, 64.0]
List size: 3
List elements: 1 2 3

=== Memory Management Patterns ===
Heap values:
  heap_ptr[0] = 0
  heap_ptr[1] = 1
  heap_ptr[2] = 4
  heap_ptr[3] = 9
  heap_ptr[4] = 16
Memory management demo completed
```

<!-- END: Part 0.1 -->
