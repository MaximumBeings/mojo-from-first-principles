# 0.2 Struct Design Patterns

# Mojo Struct Design Patterns - Part 0.2

This section demonstrates essential struct design patterns in Mojo for building high-performance automatic differentiation frameworks. These patterns form the foundation for tensor operations, memory management, and GPU computing.

---

## 0.2 Struct Design Patterns

### File: `06_basic_structs.mojo`

**Execution:** `pixi run mojo 06_basic_structs.mojo`

```mojo
from memory import UnsafePointer

struct Point2D:
    """Basic struct representing a 2D point."""
    var x: Float64
    var y: Float64
    
    fn __init__(out self, x: Float64, y: Float64):
        """Initialize a 2D point with x and y coordinates."""
        self.x = x
        self.y = y
    
    fn __init__(out self):
        """Default constructor - initialize to origin."""
        self.x = 0.0
        self.y = 0.0
    
    fn distance_from_origin(self) -> Float64:
        """Calculate distance from origin using Pythagorean theorem."""
        return pow(self.x * self.x + self.y * self.y, 0.5)
    
    fn distance_to(self, other: Point2D) -> Float64:
        """Calculate distance to another point."""
        var dx = self.x - other.x
        var dy = self.y - other.y
        return pow(dx * dx + dy * dy, 0.5)
    
    fn __str__(self) -> String:
        """String representation of the point."""
        return "Point2D(" + String(self.x) + ", " + String(self.y) + ")"

fn basic_struct_demo():
    """Demonstrate basic struct usage patterns."""
    print("=== Basic Struct Patterns ===")
    
    # Create points using different constructors
    var origin = Point2D()
    var point1 = Point2D(3.0, 4.0)
    var point2 = Point2D(1.0, 1.0)
    
    print("Origin:", origin.__str__())
    print("Point1:", point1.__str__())
    print("Point2:", point2.__str__())
    
    # Calculate distances
    var dist_from_origin = point1.distance_from_origin()
    var dist_between_points = point1.distance_to(point2)
    
    print("Point1 distance from origin:", dist_from_origin)
    print("Distance between point1 and point2:", dist_between_points)

fn main():
    """Main function for basic struct demonstration."""
    basic_struct_demo()
```

### File: `07_parametric_structs.mojo`

**Execution:** `pixi run mojo 07_parametric_structs.mojo`

```mojo
from memory import UnsafePointer

struct Vector[dtype: DType, size: Int]:
    """Parametric struct for SIMD-optimized vectors."""
    var data: SIMD[dtype, size]
    
    fn __init__(out self):
        """Initialize vector with zeros."""
        self.data = SIMD[dtype, size](0)
    
    fn __init__(out self, value: Scalar[dtype]):
        """Initialize vector with single value (broadcast)."""
        self.data = SIMD[dtype, size](value)
    
    fn __init__(out self, simd_data: SIMD[dtype, size]):
        """Initialize vector with SIMD data."""
        self.data = simd_data
    
    fn get(self, index: Int) -> Scalar[dtype]:
        """Get element at index."""
        return self.data[index]
    
    fn set(mut self, index: Int, value: Scalar[dtype]):
        """Set element at index."""
        self.data[index] = value
    
    fn add(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        """Add two vectors element-wise."""
        return Vector[dtype, size](self.data + other.data)
    
    fn multiply(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        """Multiply two vectors element-wise."""
        return Vector[dtype, size](self.data * other.data)
    
    fn sum(self) -> Scalar[dtype]:
        """Sum all elements in the vector."""
        var result: Scalar[dtype] = 0
        for i in range(size):
            result += self.data[i]
        return result
    
    fn print_vector(self, name: String):
        """Print vector elements without string concatenation."""
        print(name + ":", self.data)

fn parametric_struct_demo():
    """Demonstrate parametric struct usage."""
    print("=== Parametric Struct Patterns ===")
    
    # Create different types of vectors (SIMD widths must be powers of 2)
    alias Float32Vec4 = Vector[DType.float32, 4]
    alias Float64Vec2 = Vector[DType.float64, 2]  # Changed from 3 to 2 (power of 2)
    
    # Float32 4-element vector
    var vec1 = Float32Vec4(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))
    var vec2 = Float32Vec4(SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))
    
    vec1.print_vector("Vector 1")
    vec2.print_vector("Vector 2")
    
    # Vector operations
    var vec_sum = vec1.add(vec2)
    var vec_product = vec1.multiply(vec2)
    
    vec_sum.print_vector("Sum")
    vec_product.print_vector("Product")
    print("Sum of all elements in vec1:", vec1.sum())
    
    # Float64 2-element vector (changed to power of 2)
    var vec3 = Float64Vec2(SIMD[DType.float64, 2](1.5, 2.5))
    vec3.print_vector("Float64 Vector")
    
    # Demonstrate zero initialization and broadcast with explicit types
    var zero_vec = Float32Vec4()
    zero_vec.print_vector("Zero Vector")
    
    # Use explicit Scalar type to avoid ambiguity
    var broadcast_value: Scalar[DType.float32] = 3.14
    var broadcast_vec = Float32Vec4(broadcast_value)
    broadcast_vec.print_vector("Broadcast Vector")

fn main():
    """Main function for parametric struct demonstration."""
    parametric_struct_demo()
```

### File: `08_memory_management_structs.mojo`

**Execution:** `pixi run mojo 08_memory_management_structs.mojo`

```mojo
from memory import UnsafePointer

struct DynamicArray[dtype: DType]:
    """Dynamic array with manual memory management."""
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var capacity: Int
    
    fn __init__(out self, initial_capacity: Int):
        """Initialize with given capacity."""
        self.capacity = initial_capacity
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]].alloc(initial_capacity)
    
    fn __init__(out self):
        """Initialize with default capacity."""
        self.capacity = 8
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]].alloc(8)
    
    fn __del__(owned self):
        """Destructor - free allocated memory."""
        self.data.free()
    
    fn append(mut self, value: Scalar[dtype]):
        """Add element to end of array."""
        if self.size >= self.capacity:
            self._resize()
        
        self.data[self.size] = value
        self.size += 1
    
    fn get(self, index: Int) -> Scalar[dtype]:
        """Get element at index."""
        return self.data[index]
    
    fn set(mut self, index: Int, value: Scalar[dtype]):
        """Set element at index."""
        self.data[index] = value
    
    fn _resize(mut self):
        """Private method to resize array when capacity is exceeded."""
        var new_capacity = self.capacity * 2
        var new_data = UnsafePointer[Scalar[dtype]].alloc(new_capacity)
        
        # Copy existing data
        for i in range(self.size):
            new_data[i] = self.data[i]
        
        # Free old memory and update
        self.data.free()
        self.data = new_data
        self.capacity = new_capacity
    
    fn print_elements(self):
        """Print all elements in the array."""
        print("DynamicArray[", self.size, "/", self.capacity, "]: ", end="")
        for i in range(self.size):
            print(self.data[i], end="")
            if i < self.size - 1:
                print(", ", end="")
        print("")

fn memory_management_demo():
    """Demonstrate memory management in structs."""
    print("=== Memory Management in Structs ===")
    
    # Create dynamic arrays
    var int_array = DynamicArray[DType.int32](4)
    var float_array = DynamicArray[DType.float32]()
    
    # Add elements to trigger resizing
    print("Adding elements to int array:")
    for i in range(10):
        int_array.append(i * i)
        int_array.print_elements()
    
    print("\nAdding elements to float array:")
    for i in range(5):
        var value = Float32(i) * 1.5
        float_array.append(value)
        float_array.print_elements()
    
    # Access elements
    print("\nAccessing elements:")
    print("int_array[3] =", int_array.get(3))
    print("float_array[2] =", float_array.get(2))

fn main():
    """Main function for memory management demonstration."""
    memory_management_demo()
```

### File: `09_trait_structs.mojo`

**Execution:** `pixi run mojo 09_trait_structs.mojo`

```mojo
from memory import UnsafePointer

trait Drawable:
    """Trait for objects that can be drawn."""
    fn draw(self):
        """Draw the object."""
        pass
    
    fn area(self) -> Float64:
        """Calculate area of the object."""
        pass

struct Rectangle(Drawable):
    """Rectangle that implements Drawable trait."""
    var width: Float64
    var height: Float64
    
    fn __init__(out self, width: Float64, height: Float64):
        self.width = width
        self.height = height
    
    fn draw(self):
        """Draw the rectangle."""
        print("Drawing rectangle: " + String(self.width) + " x " + String(self.height))
    
    fn area(self) -> Float64:
        """Calculate rectangle area."""
        return self.width * self.height

struct Circle(Drawable):
    """Circle that implements Drawable trait."""
    var radius: Float64
    
    fn __init__(out self, radius: Float64):
        self.radius = radius
    
    fn draw(self):
        """Draw the circle."""
        print("Drawing circle with radius: " + String(self.radius))
    
    fn area(self) -> Float64:
        """Calculate circle area."""
        alias PI = 3.14159265359
        return PI * self.radius * self.radius

fn draw_shape[T: Drawable](shape: T):
    """Generic function that works with any Drawable."""
    shape.draw()
    print("Area: " + String(shape.area()))

fn trait_demo():
    """Demonstrate trait usage with structs."""
    print("=== Trait-based Struct Patterns ===")
    
    # Create different shapes
    var rect = Rectangle(5.0, 3.0)
    var circle = Circle(2.5)
    
    # Use generic function with different types
    print("Rectangle:")
    draw_shape(rect)
    
    print("\nCircle:")
    draw_shape(circle)
    
    # Direct method calls
    print("\nDirect method calls:")
    rect.draw()
    circle.draw()

fn main():
    """Main function for trait demonstration."""
    trait_demo()
```

### File: `10_struct_patterns_complete.mojo`

**Execution:** `pixi run mojo 10_struct_patterns_complete.mojo`

```mojo
from memory import UnsafePointer

# Basic struct pattern
struct Point2D:
    """Basic struct representing a 2D point."""
    var x: Float64
    var y: Float64
    
    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y
    
    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0
    
    fn distance_from_origin(self) -> Float64:
        return pow(self.x * self.x + self.y * self.y, 0.5)

# Parametric struct pattern
struct Vector[dtype: DType, size: Int]:
    """Parametric struct for SIMD-optimized vectors."""
    var data: SIMD[dtype, size]
    
    fn __init__(out self, simd_data: SIMD[dtype, size]):
        self.data = simd_data
    
    fn add(self, other: Vector[dtype, size]) -> Vector[dtype, size]:
        return Vector[dtype, size](self.data + other.data)

# Memory management pattern
struct SimpleArray[dtype: DType]:
    """Simple array with manual memory management."""
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    
    fn __init__(out self, size: Int):
        self.size = size
        self.data = UnsafePointer[Scalar[dtype]].alloc(size)
    
    fn __del__(owned self):
        self.data.free()
    
    fn get(self, index: Int) -> Scalar[dtype]:
        return self.data[index]
    
    fn set(mut self, index: Int, value: Scalar[dtype]):
        self.data[index] = value

# Trait pattern
trait Printable:
    fn print_info(self):
        pass

struct Number(Printable):
    var value: Float64
    
    fn __init__(out self, value: Float64):
        self.value = value
    
    fn print_info(self):
        var value_str: String = String(self.value)
        var message: String = "Number value: " + value_str
        print(message)

fn complete_struct_demo():
    """Demonstrate all struct patterns together."""
    print("=== Complete Struct Patterns Demo ===")
    
    # Basic struct
    var point = Point2D(3.0, 4.0)
    print("Point distance from origin:", point.distance_from_origin())
    
    # Parametric struct
    alias FloatVec4 = Vector[DType.float32, 4]
    var vec1 = FloatVec4(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))
    var vec2 = FloatVec4(SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))
    var vec_sum = vec1.add(vec2)
    print("Vector addition result: ", vec_sum.data)
    
    # Memory management
    var array = SimpleArray[DType.float64](3)
    array.set(0, 1.1)
    array.set(1, 2.2)
    array.set(2, 3.3)
    print("Array elements:", array.get(0), array.get(1), array.get(2))
    
    # Trait usage
    var number = Number(42.5)
    number.print_info()

fn main():
    """Complete demonstration of all struct patterns."""
    complete_struct_demo()
```

### Expected Output for `10_struct_patterns_complete.mojo`
```
=== Complete Struct Patterns Demo ===
Point distance from origin: 5.0
Vector addition result: [6.0, 8.0, 10.0, 12.0]
Array elements: 1.1 2.2 3.3
Number value: 42.5
```

---

## Key Struct Design Patterns Summary

**1. Basic Structs** - Simple data containers with methods
- Multiple constructors (`__init__` overloading)
- Member functions and string representation
- Value semantics and direct member access

**2. Parametric Structs** - Generic structs with compile-time parameters
- Type parameters (`dtype: DType`, `size: Int`)
- SIMD optimization through parameterization
- Zero-cost abstractions at compile time

**3. Memory Management** - Manual memory control with RAII
- Constructor allocates (`__init__`)
- Destructor deallocates (`__del__`)
- Automatic cleanup and resource management

**4. Trait-based Design** - Polymorphism and generic programming
- Interface definitions with traits
- Implementation inheritance
- Generic functions working with trait bounds

**5. Performance Patterns** - Optimization-focused designs
- SIMD vector operations
- Cache-friendly memory layouts
- Compile-time optimization opportunities
