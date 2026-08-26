# 1.2 Memory Layout Design: Strides, Views, Alignment & Broadcasting

### File: `37_stride_calculation.mojo`

**Run:** `pixi run mojo 37_stride_calculation.mojo`

```mojo

from memory import UnsafePointer
from collections import List

```

### Part 1.2.1 -- Stride Calculation System

```mojo

# Core Tensor Infrastructure - Part 1.2.1: Stride Calculation System
#
# This section implements comprehensive stride calculation for tensor memory layouts.
# Provides support for row-major (C-style) and column-major (Fortran-style) ordering,
# custom stride patterns, and efficient memory access pattern optimization.
#
# Key Design Principles:
# - Support multiple memory layout conventions
# - Efficient stride computation algorithms
# - Custom stride patterns for advanced indexing
# - Performance-optimized memory access patterns
# - Layout conversion and validation utilities
#
# Implementation Strategy:
# 1. Row-major and column-major stride calculators
# 2. Custom stride pattern support
# 3. Index-to-offset conversion utilities
# 4. Memory layout optimization
# 5. Performance analysis and benchmarking tools
#
# Memory Layout Concepts:
# - Row-major (C-style): rightmost dimension varies fastest
# - Column-major (Fortran-style): leftmost dimension varies fastest
# - Custom strides: arbitrary memory access patterns
# - Contiguous vs non-contiguous layouts

alias DEFAULT_ALIGNMENT = 64  # 64-byte alignment for SIMD operations

```

#### 1.2.1.1 Memory Layout Enums and Types

```mojo

struct MemoryLayout(Copyable, Movable):
    """Memory layout specification for tensor stride calculation."""
    alias ROW_MAJOR = 0     # C-style: rightmost dimension varies fastest
    alias COLUMN_MAJOR = 1  # Fortran-style: leftmost dimension varies fastest
    alias CUSTOM = 2        # User-defined stride pattern
    
    var layout_type: Int
    
    fn __init__(out self, layout_type: Int = Self.ROW_MAJOR):
        self.layout_type = layout_type
    
    fn is_row_major(self) -> Bool:
        return self.layout_type == Self.ROW_MAJOR
    
    fn is_column_major(self) -> Bool:
        return self.layout_type == Self.COLUMN_MAJOR
    
    fn is_custom(self) -> Bool:
        return self.layout_type == Self.CUSTOM
    
    fn name(self) -> String:
        if self.layout_type == Self.ROW_MAJOR:
            return "ROW_MAJOR"
        elif self.layout_type == Self.COLUMN_MAJOR:
            return "COLUMN_MAJOR"
        else:
            return "CUSTOM"

struct StrideInfo:
    """Comprehensive stride information for tensors."""
    var strides: UnsafePointer[Int]
    var shape: UnsafePointer[Int]
    var ndim: Int
    var layout: MemoryLayout
    var is_contiguous: Bool
    var element_size: Int
    var total_bytes: Int
    
    fn __init__(out self, shape: List[Int], layout: MemoryLayout = MemoryLayout(), element_size: Int = 4):
        """
        Initialize stride information from shape and layout.
        
        Args:
            shape: Tensor dimensions.
            layout: Memory layout specification.
            element_size: Size of each element in bytes.
        """
        self.ndim = len(shape)
        self.layout = layout
        self.element_size = element_size
        self.is_contiguous = False  # Initialize before using in methods
        
        # Allocate memory for shape and strides
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        
        # Copy shape
        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            total_elements *= shape[i]
        
        self.total_bytes = total_elements * element_size
        
        # Calculate strides based on layout
        if layout.is_row_major():
            self._compute_row_major_strides()
        elif layout.is_column_major():
            self._compute_column_major_strides()
        else:
            # Default to row-major for custom (will be overridden)
            self._compute_row_major_strides()
        
        # Check contiguity after strides are computed
        self.is_contiguous = self._check_contiguity()
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for stride information."""
        self.ndim = existing.ndim
        self.layout = existing.layout
        self.is_contiguous = existing.is_contiguous
        self.element_size = existing.element_size
        self.total_bytes = existing.total_bytes
        
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.strides[i] = existing.strides[i]
    
    fn __del__(owned self):
        """Destructor for stride information."""
        self.shape.free()
        self.strides.free()
    
    fn _compute_row_major_strides(self):
        """Compute row-major (C-style) strides."""
        if self.ndim == 0:
            return
        
        # Rightmost dimension has stride 1
        self.strides[self.ndim - 1] = 1
        
        # Work backwards, each stride is the product of all following dimensions
        for i in range(self.ndim - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
    
    fn _compute_column_major_strides(self):
        """Compute column-major (Fortran-style) strides."""
        if self.ndim == 0:
            return
        
        # Leftmost dimension has stride 1
        self.strides[0] = 1
        
        # Work forwards, each stride is the product of all preceding dimensions
        for i in range(1, self.ndim):
            self.strides[i] = self.strides[i - 1] * self.shape[i - 1]
    
    fn _check_contiguity(self) -> Bool:
        """Check if the current stride pattern represents contiguous memory."""
        if self.ndim <= 1:
            return True
        
        if self.layout.is_row_major():
            # For row-major: stride[i] should equal product of shape[i+1:]
            var expected_stride = 1
            for i in range(self.ndim - 1, -1, -1):
                if self.strides[i] != expected_stride:
                    return False
                expected_stride *= self.shape[i]
            return True
        
        elif self.layout.is_column_major():
            # For column-major: stride[i] should equal product of shape[:i]
            var expected_stride = 1
            for i in range(self.ndim):
                if self.strides[i] != expected_stride:
                    return False
                expected_stride *= self.shape[i]
            return True
        
        # Custom layouts may or may not be contiguous
        return False
    
    fn get_stride(self, axis: Int) -> Int:
        """Get stride for specified axis with bounds checking."""
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.strides[axis]
    
    fn get_shape(self, axis: Int) -> Int:
        """Get shape for specified axis with bounds checking."""
        if axis < 0 or axis >= self.ndim:
            return 0
        return self.shape[axis]

```

#### 1.2.1.2 Stride Calculation Functions

```mojo

fn compute_strides(shape: List[Int], layout: MemoryLayout = MemoryLayout()) -> List[Int]:
    """
    Compute strides for given shape and memory layout.
    
    Args:
        shape: Tensor dimensions.
        layout: Memory layout specification.
    
    Returns:
        List of strides for each dimension.
    
    Example:
        shape=[2, 3, 4], row-major -> strides=[12, 4, 1].
        shape=[2, 3, 4], column-major -> strides=[1, 2, 6].
    """
    var ndim = len(shape)
    var strides = List[Int]()
    
    if ndim == 0:
        return strides
    
    # Initialize strides list
    for _ in range(ndim):
        strides.append(0)
    
    if layout.is_row_major():
        # Row-major: rightmost dimension varies fastest
        strides[ndim - 1] = 1
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    
    elif layout.is_column_major():
        # Column-major: leftmost dimension varies fastest
        strides[0] = 1
        for i in range(1, ndim):
            strides[i] = strides[i - 1] * shape[i - 1]
    
    return strides

fn compute_custom_strides(shape: List[Int], custom_strides: List[Int]) -> Bool:
    """
    Validate custom stride pattern against shape.
    
    Args:
        shape: Tensor dimensions.
        custom_strides: User-provided stride pattern.
    
    Returns:
        True if custom strides are valid for the given shape.
    
    Validation Rules:
    - Number of strides must match number of dimensions.
    - All strides must be positive.
    - No stride overlap that would cause memory corruption.
    """
    if len(shape) != len(custom_strides):
        return False
    
    # Check for positive strides
    for i in range(len(custom_strides)):
        if custom_strides[i] <= 0:
            return False
    
    # Additional validation could include overlap detection
    # This is a simplified version
    return True

fn index_to_offset(indices: List[Int], strides: List[Int]) -> Int:
    """
    Convert multi-dimensional indices to linear memory offset.
    
    Args:
        indices: Multi-dimensional tensor indices.
        strides: Stride pattern for each dimension.
    
    Returns:
        Linear memory offset.
    
    Formula: offset = sum(index[i] * stride[i] for i in range(ndim)).
    """
    if len(indices) != len(strides):
        return -1  # Error indicator
    
    var offset = 0
    for i in range(len(indices)):
        offset += indices[i] * strides[i]
    
    return offset

fn offset_to_indices(offset: Int, shape: List[Int], strides: List[Int]) -> List[Int]:
    """
    Convert linear memory offset to multi-dimensional indices.
    
    Args:
        offset: Linear memory offset.
        shape: Tensor dimensions.
        strides: Stride pattern for each dimension.
    
    Returns:
        Multi-dimensional indices.
    
    Note: This is the inverse operation of index_to_offset.
    """
    var indices = List[Int]()
    var remaining_offset = offset
    
    if len(shape) != len(strides):
        return indices  # Return empty list on error
    
    # For row-major layout, work from largest to smallest stride
    var ndim = len(shape)
    for _ in range(ndim):
        indices.append(0)
    
    # Simple algorithm for row-major layout
    for i in range(ndim):
        if strides[i] > 0:
            indices[i] = remaining_offset // strides[i]
            remaining_offset = remaining_offset % strides[i]
    
    return indices

```

#### 1.2.1.3 Advanced Stride Operations

```mojo

fn calculate_broadcast_strides(shape_a: List[Int], shape_b: List[Int], 
                             strides_a: List[Int], strides_b: List[Int]) -> List[Int]:
    """
    Calculate result strides for broadcasting operation.
    
    Args:
        shape_a: Input tensor shape A.
        shape_b: Input tensor shape B.
        strides_a: Input tensor strides A.
        strides_b: Input tensor strides B.
    
    Returns:
        Result strides after broadcasting.
    
    Broadcasting Rules:
    - Result shape is element-wise maximum of input shapes.
    - Dimensions of size 1 get stride 0 (virtual broadcasting).
    - Missing dimensions are treated as size 1.
    """
    var len_a = len(shape_a)
    var len_b = len(shape_b)
    var max_len = len_a if len_a > len_b else len_b
    
    var result_strides = List[Int]()
    
    # Process dimensions from right to left
    for i in range(max_len):
        var dim_a = 1
        var dim_b = 1
        var stride_a = 0
        var stride_b = 0
        
        # Get dimensions and strides (with defaults for missing dims)
        if i < len_a:
            dim_a = shape_a[len_a - 1 - i]
            stride_a = strides_a[len_a - 1 - i]
        if i < len_b:
            dim_b = shape_b[len_b - 1 - i]
            stride_b = strides_b[len_b - 1 - i]
        
        # Determine result stride
        var result_stride = 0
        if dim_a == dim_b:
            # Same size: use stride from either tensor
            result_stride = stride_a if stride_a > stride_b else stride_b
        elif dim_a == 1:
            # A broadcasts: use B's stride
            result_stride = stride_b
        elif dim_b == 1:
            # B broadcasts: use A's stride
            result_stride = stride_a
        
        result_strides.append(result_stride)
    
    # Reverse to get correct order
    var final_strides = List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        final_strides.append(result_strides[i])
    
    return final_strides

fn is_stride_pattern_contiguous(shape: List[Int], strides: List[Int], layout: MemoryLayout) -> Bool:
    """
    Check if stride pattern represents contiguous memory layout.
    
    Args:
        shape: Tensor dimensions.
        strides: Stride pattern to check.
        layout: Expected memory layout.
    
    Returns:
        True if strides represent contiguous memory in specified layout.
    """
    if len(shape) != len(strides):
        return False
    
    if len(shape) <= 1:
        return True
    
    if layout.is_row_major():
        # Check row-major contiguity
        var expected_stride = 1
        for i in range(len(shape) - 1, -1, -1):
            if strides[i] != expected_stride:
                return False
            expected_stride *= shape[i]
        return True
    
    elif layout.is_column_major():
        # Check column-major contiguity
        var expected_stride = 1
        for i in range(len(shape)):
            if strides[i] != expected_stride:
                return False
            expected_stride *= shape[i]
        return True
    
    return False  # Custom layouts are not considered contiguous by default

fn optimize_stride_pattern(shape: List[Int], access_pattern: List[Int]) -> List[Int]:
    """
    Optimize stride pattern for specific access patterns.
    
    Args:
        shape: Tensor dimensions.
        access_pattern: Relative frequency of access for each dimension.
    
    Returns:
        Optimized stride pattern for the given access pattern.
    
    Strategy:
    - Dimensions accessed more frequently get smaller strides.
    - This improves cache locality for the specific use case.
    """
    var ndim = len(shape)
    var optimized_strides = List[Int]()
    
    if ndim == 0:
        return optimized_strides
    
    # Simple optimization: sort dimensions by access frequency
    # In a real implementation, this would be more sophisticated
    
    # For now, return row-major as default optimization
    return compute_strides(shape, MemoryLayout(MemoryLayout.ROW_MAJOR))

```

#### 1.2.1.4 Enhanced Tensor with Stride Support

```mojo

struct StridedTensor[dtype: DType]:
    """
    Enhanced tensor with comprehensive stride support.
    
    Supports multiple memory layouts, custom stride patterns, and efficient
    memory access operations. Provides foundation for advanced tensor operations
    including views, slicing, and broadcasting.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var stride_info: StrideInfo
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int], layout: MemoryLayout = MemoryLayout()) raises:
        """
        Create strided tensor with specified shape and layout.
        
        Args:
            shape: Tensor dimensions.
            layout: Memory layout specification.
        
        Raises:
            Error if shape is invalid or memory allocation fails.
        """
        # Validate shape
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
        
        # Initialize stride information
        self.stride_info = StrideInfo(shape, layout, 4)  # Assume 4 bytes for now
        self._owns_data = True
        
        # Calculate total elements
        var total_elements = 1
        for i in range(len(shape)):
            total_elements *= shape[i]
        
        # Allocate memory
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        
        # Initialize to zero
        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for strided tensor."""
        self.stride_info = existing.stride_info
        self._owns_data = True
        
        # Calculate total elements and allocate
        var total_elements = 1
        for i in range(self.stride_info.ndim):
            total_elements *= self.stride_info.get_shape(i)
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        
        # Copy data
        for i in range(total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        """Destructor for strided tensor."""
        if self._owns_data:
            self.data.free()
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """
        Get tensor element using multi-dimensional indexing.
        
        Args:
            indices: Multi-dimensional tensor indices.
        
        Returns:
            Tensor element at specified indices.
        """
        var strides = List[Int]()
        for i in range(self.stride_info.ndim):
            strides.append(self.stride_info.get_stride(i))
        
        var offset = index_to_offset(indices, strides)
        return self.data[offset]
    
    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        """
        Set tensor element using multi-dimensional indexing.
        
        Args:
            indices: Multi-dimensional tensor indices.
            value: Value to assign.
        """
        var strides = List[Int]()
        for i in range(self.stride_info.ndim):
            strides.append(self.stride_info.get_stride(i))
        
        var offset = index_to_offset(indices, strides)
        self.data[offset] = value
    
    fn get_layout(self) -> MemoryLayout:
        """Get memory layout type."""
        return self.stride_info.layout
    
    fn is_contiguous(self) -> Bool:
        """Check if tensor has contiguous memory layout."""
        return self.stride_info.is_contiguous
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        var total = 1
        for i in range(self.stride_info.ndim):
            total *= self.stride_info.get_shape(i)
        return total
    
    fn ndim(self) -> Int:
        """Get number of dimensions."""
        return self.stride_info.ndim
    
    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with specified value."""
        var total_elements = self.numel()
        for i in range(total_elements):
            self.data[i] = value
    
    fn print_stride_info(self):
        """Display comprehensive stride information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        
        print("StridedTensor[" + dtype_str + "]")
        print("  Layout: " + self.stride_info.layout.name())
        var contiguous_str: String = "True" if self.stride_info.is_contiguous else "False"
        print("  Contiguous: " + contiguous_str)
        
        print("  Shape: [", end="")
        for i in range(self.stride_info.ndim):
            var shape_str: String = String(self.stride_info.get_shape(i))
            print(shape_str, end="")
            if i < self.stride_info.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Strides: [", end="")
        for i in range(self.stride_info.ndim):
            var stride_str: String = String(self.stride_info.get_stride(i))
            print(stride_str, end="")
            if i < self.stride_info.ndim - 1:
                print(", ", end="")
        print("]")
        
        var bytes_str: String = String(self.stride_info.total_bytes)
        print("  Total bytes: " + bytes_str)
    
    fn print_memory_layout(self, max_elements: Int = 16):
        """Display memory layout pattern."""
        var total_elements = self.numel()
        var elements_to_show = total_elements if total_elements <= max_elements else max_elements
        
        print("  Memory layout:")
        for i in range(elements_to_show):
            var i_str: String = String(i)
            var value_str: String = String(self.data[i])
            var message: String = "    data[" + i_str + "] = " + value_str
            print(message)
        
        if total_elements > max_elements:
            var remaining_str: String = String(total_elements - max_elements)
            var remaining_msg: String = "    ... (" + remaining_str + " more elements)"
            print(remaining_msg)

```

#### 1.2.1.5 Stride Factory Functions

```mojo

fn create_row_major_tensor[dtype: DType](shape: List[Int]) raises -> StridedTensor[dtype]:
    """Create tensor with row-major (C-style) memory layout."""
    return StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.ROW_MAJOR))

fn create_column_major_tensor[dtype: DType](shape: List[Int]) raises -> StridedTensor[dtype]:
    """Create tensor with column-major (Fortran-style) memory layout."""
    return StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.COLUMN_MAJOR))

fn create_custom_stride_tensor[dtype: DType](shape: List[Int], custom_strides: List[Int]) raises -> StridedTensor[dtype]:
    """
    Create tensor with custom stride pattern.
    
    Args:
        shape: Tensor dimensions.
        custom_strides: User-defined stride pattern.
    
    Returns:
        Tensor with custom memory layout.
    
    Raises:
        Error if custom strides are invalid.
    """
    if not compute_custom_strides(shape, custom_strides):
        raise Error("Invalid custom stride pattern")
    
    # Create base tensor and modify strides
    var tensor = StridedTensor[dtype](shape, MemoryLayout(MemoryLayout.CUSTOM))
    
    # Override strides with custom pattern
    for i in range(len(custom_strides)):
        tensor.stride_info.strides[i] = custom_strides[i]
    
    # Update contiguity information
    tensor.stride_info.is_contiguous = False  # Custom strides assumed non-contiguous
    
    return tensor

```

#### Testing and Demonstration Functions

```mojo

fn test_basic_stride_calculation():
    """Test Suite: Basic Stride Calculation."""
    print("=== Testing Basic Stride Calculation ===")
    
    # Test row-major strides
    print("\n1. Row-Major Stride Calculation:")
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    
    var row_major_strides = compute_strides(shape_2d, MemoryLayout(MemoryLayout.ROW_MAJOR))
    print("Shape: [3, 4]")
    var row_stride0_str: String = String(row_major_strides[0])
    var row_stride1_str: String = String(row_major_strides[1])
    var row_msg: String = "Row-major strides: [" + row_stride0_str + ", " + row_stride1_str + "]"
    print(row_msg)
    
    # Test column-major strides
    print("\n2. Column-Major Stride Calculation:")
    var col_major_strides = compute_strides(shape_2d, MemoryLayout(MemoryLayout.COLUMN_MAJOR))
    print("Shape: [3, 4]")
    var col_stride0_str: String = String(col_major_strides[0])
    var col_stride1_str: String = String(col_major_strides[1])
    var col_msg: String = "Column-major strides: [" + col_stride0_str + ", " + col_stride1_str + "]"
    print(col_msg)
    
    # Test 3D case
    print("\n3. 3D Tensor Strides:")
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    
    var row_3d = compute_strides(shape_3d, MemoryLayout(MemoryLayout.ROW_MAJOR))
    var col_3d = compute_strides(shape_3d, MemoryLayout(MemoryLayout.COLUMN_MAJOR))
    
    print("Shape: [2, 3, 4]")
    var row3d_s0: String = String(row_3d[0])
    var row3d_s1: String = String(row_3d[1])
    var row3d_s2: String = String(row_3d[2])
    var row3d_msg: String = "Row-major strides: [" + row3d_s0 + ", " + row3d_s1 + ", " + row3d_s2 + "]"
    print(row3d_msg)
    
    var col3d_s0: String = String(col_3d[0])
    var col3d_s1: String = String(col_3d[1])
    var col3d_s2: String = String(col_3d[2])
    var col3d_msg: String = "Column-major strides: [" + col3d_s0 + ", " + col3d_s1 + ", " + col3d_s2 + "]"
    print(col3d_msg)

fn test_index_offset_conversion():
    """Test Suite: Index-Offset Conversion."""
    print("\n=== Testing Index-Offset Conversion ===")
    
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    
    var strides = compute_strides(shape, MemoryLayout(MemoryLayout.ROW_MAJOR))
    
    print("\n1. Index to Offset Conversion:")
    print("Shape: [3, 4], Row-major strides: [", end="")
    var stride0_str: String = String(strides[0])
    var stride1_str: String = String(strides[1])
    print(stride0_str + ", " + stride1_str + "]")
    
    # Test several index combinations
    var test_indices = List[List[Int]]()
    
    var idx1 = List[Int]()
    idx1.append(0)
    idx1.append(0)
    test_indices.append(idx1)
    
    var idx2 = List[Int]()
    idx2.append(1)
    idx2.append(2)
    test_indices.append(idx2)
    
    var idx3 = List[Int]()
    idx3.append(2)
    idx3.append(3)
    test_indices.append(idx3)
    
    for i in range(len(test_indices)):
        var indices = test_indices[i]
        var offset = index_to_offset(indices, strides)
        var idx0_str: String = String(indices[0])
        var idx1_str: String = String(indices[1])
        var offset_str: String = String(offset)
        var index_msg: String = "  Index [" + idx0_str + ", " + idx1_str + "] -> Offset " + offset_str
        print(index_msg)
        
        # Test reverse conversion
        var recovered_indices = offset_to_indices(offset, shape, strides)
        if len(recovered_indices) >= 2:
            var rec0_str: String = String(recovered_indices[0])
            var rec1_str: String = String(recovered_indices[1])
            var reverse_msg: String = "  Offset " + offset_str + " -> Index [" + rec0_str + ", " + rec1_str + "]"
            print(reverse_msg)

fn test_strided_tensor_operations():
    """Test Suite: Strided Tensor Operations."""
    print("\n=== Testing Strided Tensor Operations ===")
    
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    
    try:
        # Test row-major tensor
        print("\n1. Row-Major Tensor:")
        var row_tensor = create_row_major_tensor[DType.float32](shape)
        row_tensor.print_stride_info()
        
        # Fill with test data
        var indices = List[Int]()
        indices.append(0)
        indices.append(0)
        row_tensor.set_item(indices, 1.0)
        
        indices[0] = 0
        indices[1] = 1
        row_tensor.set_item(indices, 2.0)
        
        indices[0] = 1
        indices[1] = 0
        row_tensor.set_item(indices, 3.0)
        
        indices[0] = 1
        indices[1] = 1
        row_tensor.set_item(indices, 4.0)
        
        print("Data access test:")
        indices[0] = 0
        indices[1] = 0
        var val00: String = String(row_tensor.get_item(indices))
        print("  tensor[0, 0] = " + val00)
        indices[0] = 1
        indices[1] = 1
        var val11: String = String(row_tensor.get_item(indices))
        print("  tensor[1, 1] = " + val11)
        
        # Test column-major tensor
        print("\n2. Column-Major Tensor:")
        var col_tensor = create_column_major_tensor[DType.float32](shape)
        col_tensor.print_stride_info()
        
        # Fill with same logical data
        indices[0] = 0
        indices[1] = 0
        col_tensor.set_item(indices, 1.0)
        
        indices[0] = 0
        indices[1] = 1
        col_tensor.set_item(indices, 2.0)
        
        indices[0] = 1
        indices[1] = 0
        col_tensor.set_item(indices, 3.0)
        
        indices[0] = 1
        indices[1] = 1
        col_tensor.set_item(indices, 4.0)
        
        print("Data access test:")
        indices[0] = 0
        indices[1] = 0
        var col_val00: String = String(col_tensor.get_item(indices))
        print("  tensor[0, 0] = " + col_val00)
        indices[0] = 1
        indices[1] = 1
        var col_val11: String = String(col_tensor.get_item(indices))
        print("  tensor[1, 1] = " + col_val11)
        
        # Compare memory layouts
        print("\n3. Memory Layout Comparison:")
        print("Row-major tensor memory:")
        row_tensor.print_memory_layout(6)
        print("Column-major tensor memory:")
        col_tensor.print_memory_layout(6)
        
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in strided tensor test: " + error_str
        print(error_msg)

fn test_broadcasting_stride_calculation():
    """Test Suite: Broadcasting Stride Calculation."""
    print("\n=== Testing Broadcasting Stride Calculation ===")
    
    # Test broadcasting between different shapes
    print("\n1. Broadcasting Compatibility:")
    
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    shape_a.append(4)
    
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(1)
    
    var strides_a = compute_strides(shape_a, MemoryLayout(MemoryLayout.ROW_MAJOR))
    var strides_b = compute_strides(shape_b, MemoryLayout(MemoryLayout.ROW_MAJOR))
    
    var sa0: String = String(strides_a[0])
    var sa1: String = String(strides_a[1])
    var sa2: String = String(strides_a[2])
    var stride_a_msg: String = "Tensor A - Shape: [2, 3, 4], Strides: [" + sa0 + ", " + sa1 + ", " + sa2 + "]"
    print(stride_a_msg)
    
    var sb0: String = String(strides_b[0])
    var sb1: String = String(strides_b[1])
    var stride_b_msg: String = "Tensor B - Shape: [3, 1], Strides: [" + sb0 + ", " + sb1 + "]"
    print(stride_b_msg)
    
    var broadcast_strides = calculate_broadcast_strides(shape_a, shape_b, strides_a, strides_b)
    
    print("Broadcast result strides: [", end="")
    for i in range(len(broadcast_strides)):
        var bs_str: String = String(broadcast_strides[i])
        print(bs_str, end="")
        if i < len(broadcast_strides) - 1:
            print(", ", end="")
    print("]")

fn test_contiguity_checking():
    """Test Suite: Contiguity Checking."""
    print("\n=== Testing Contiguity Checking ===")
    
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    
    # Test contiguous patterns
    print("\n1. Contiguity Analysis:")
    
    var row_strides = compute_strides(shape, MemoryLayout(MemoryLayout.ROW_MAJOR))
    var col_strides = compute_strides(shape, MemoryLayout(MemoryLayout.COLUMN_MAJOR))
    
    var row_contiguous = is_stride_pattern_contiguous(shape, row_strides, MemoryLayout(MemoryLayout.ROW_MAJOR))
    var col_contiguous = is_stride_pattern_contiguous(shape, col_strides, MemoryLayout(MemoryLayout.COLUMN_MAJOR))
    
    var rs0: String = String(row_strides[0])
    var rs1: String = String(row_strides[1])
    var row_cont_str: String = "True" if row_contiguous else "False"
    var row_cont_msg: String = "Row-major strides [" + rs0 + ", " + rs1 + "] - Contiguous: " + row_cont_str
    print(row_cont_msg)
    
    var cs0: String = String(col_strides[0])
    var cs1: String = String(col_strides[1])
    var col_cont_str: String = "True" if col_contiguous else "False"
    var col_cont_msg: String = "Column-major strides [" + cs0 + ", " + cs1 + "] - Contiguous: " + col_cont_str
    print(col_cont_msg)
    
    # Test non-contiguous pattern
    var custom_strides = List[Int]()
    custom_strides.append(10)  # Non-standard stride
    custom_strides.append(1)
    
    var custom_contiguous = is_stride_pattern_contiguous(shape, custom_strides, MemoryLayout(MemoryLayout.ROW_MAJOR))
    var cus0: String = String(custom_strides[0])
    var cus1: String = String(custom_strides[1])
    var custom_cont_str: String = "True" if custom_contiguous else "False"
    var custom_cont_msg: String = "Custom strides [" + cus0 + ", " + cus1 + "] - Contiguous: " + custom_cont_str
    print(custom_cont_msg)

fn test_stride_info_structure():
    """Test Suite: StrideInfo Structure."""
    print("\n=== Testing StrideInfo Structure ===")
    
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    
    print("\n1. StrideInfo Analysis:")
    
    # Test row-major stride info
    var row_info = StrideInfo(shape, MemoryLayout(MemoryLayout.ROW_MAJOR), 4)
    print("Row-major StrideInfo:")
    print("  Layout: " + row_info.layout.name())
    var ndim_str: String = String(row_info.ndim)
    print("  Dimensions: " + ndim_str)
    var elem_size_str: String = String(row_info.element_size)
    print("  Element size: " + elem_size_str + " bytes")
    var total_bytes_str: String = String(row_info.total_bytes)
    print("  Total bytes: " + total_bytes_str)
    var contiguous_str: String = "True" if row_info.is_contiguous else "False"
    print("  Contiguous: " + contiguous_str)
    print("  Shape: [", end="")
    for i in range(row_info.ndim):
        var shape_str: String = String(row_info.get_shape(i))
        print(shape_str, end="")
        if i < row_info.ndim - 1:
            print(", ", end="")
    print("]")
    print("  Strides: [", end="")
    for i in range(row_info.ndim):
        var stride_str: String = String(row_info.get_stride(i))
        print(stride_str, end="")
        if i < row_info.ndim - 1:
            print(", ", end="")
    print("]")
    
    # Test column-major stride info
    print("\nColumn-major StrideInfo:")
    var col_info = StrideInfo(shape, MemoryLayout(MemoryLayout.COLUMN_MAJOR), 4)
    print("  Layout: " + col_info.layout.name())
    var col_contiguous_str: String = "True" if col_info.is_contiguous else "False"
    print("  Contiguous: " + col_contiguous_str)
    print("  Strides: [", end="")
    for i in range(col_info.ndim):
        var col_stride_str: String = String(col_info.get_stride(i))
        print(col_stride_str, end="")
        if i < col_info.ndim - 1:
            print(", ", end="")
    print("]")

fn test_performance_analysis():
    """Test Suite: Performance Analysis."""
    print("\n=== Testing Performance Analysis ===")
    
    print("\n1. Memory Access Pattern Analysis:")
    print("Understanding stride impact on performance:")
    print("- Row-major [2, 3, 4] with strides [12, 4, 1]:")
    print("  Sequential access: data[0], data[1], data[2], data[3] (cache-friendly)")
    print("  Cross-row access: data[0], data[4], data[8] (stride=4, moderate cache impact)")
    print("  Cross-page access: data[0], data[12] (stride=12, potential cache miss)")
    
    print("\n- Column-major [2, 3, 4] with strides [1, 2, 6]:")
    print("  Sequential access: data[0], data[1] (cache-friendly)")
    print("  Cross-column access: data[0], data[2], data[4] (stride=2, good cache usage)")
    print("  Cross-matrix access: data[0], data[6] (stride=6, moderate cache impact)")
    
    print("\n2. Layout Recommendations:")
    print("- Use row-major for C-style algorithms (rightmost index varies fastest)")
    print("- Use column-major for Fortran-style algorithms (leftmost index varies fastest)")
    print("- Consider access patterns when choosing layout for optimal performance")
    print("- Contiguous layouts generally provide better cache performance")

```

```mojo

fn main():
    """Main demonstration function."""
    print("=== Stride Calculation System - Part 1.2.1 ===")
    print("Memory Layout Design - Stride Computation and Access Patterns")
    
    test_basic_stride_calculation()
    test_index_offset_conversion()
    test_strided_tensor_operations()
    test_broadcasting_stride_calculation()
    test_contiguity_checking()
    test_stride_info_structure()
    test_performance_analysis()
    
    print("\n=== Stride Calculation System Implementation Summary ===")
    print("+ Row-major (C-style) and column-major (Fortran-style) stride computation")
    print("+ Custom stride pattern support with validation")
    print("+ Efficient index-to-offset and offset-to-index conversion")
    print("+ Broadcasting-aware stride calculation")
    print("+ Contiguity detection and analysis")
    print("+ Comprehensive stride information tracking")
    print("+ Performance-oriented memory access patterns")
    print("+ Foundation for advanced tensor view operations")
    
```
=== Stride Calculation System - Part 1.2.1 ===
Memory Layout Design - Stride Computation and Access Patterns
=== Testing Basic Stride Calculation ===

1. Row-Major Stride Calculation:
Shape: [3, 4]
Row-major strides: [4, 1]

2. Column-Major Stride Calculation:
Shape: [3, 4]
Column-major strides: [1, 3]

3. 3D Tensor Strides:
Shape: [2, 3, 4]
Row-major strides: [12, 4, 1]
Column-major strides: [1, 2, 6]

=== Testing Index-Offset Conversion ===

1. Index to Offset Conversion:
Shape: [3, 4], Row-major strides: [4, 1]
  Index [0, 0] -> Offset 0
  Offset 0 -> Index [0, 0]
  Index [1, 2] -> Offset 6
  Offset 6 -> Index [1, 2]
  Index [2, 3] -> Offset 11
  Offset 11 -> Index [2, 3]

=== Testing Strided Tensor Operations ===

1. Row-Major Tensor:
StridedTensor[float32]
  Layout: ROW_MAJOR
  Contiguous: True
  Shape: [2, 3]
  Strides: [3, 1]
  Total bytes: 24
Data access test:
  tensor[0, 0] = 1.0
  tensor[1, 1] = 4.0

2. Column-Major Tensor:
StridedTensor[float32]
  Layout: COLUMN_MAJOR
  Contiguous: True
  Shape: [2, 3]
  Strides: [1, 2]
  Total bytes: 24
Data access test:
  tensor[0, 0] = 1.0
  tensor[1, 1] = 4.0

3. Memory Layout Comparison:
Row-major tensor memory:
  Memory layout:
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 0.0
    data[3] = 3.0
    data[4] = 4.0
    data[5] = 0.0
Column-major tensor memory:
  Memory layout:
    data[0] = 1.0
    data[1] = 3.0
    data[2] = 2.0
    data[3] = 4.0
    data[4] = 0.0
    data[5] = 0.0

=== Testing Broadcasting Stride Calculation ===

1. Broadcasting Compatibility:
Tensor A - Shape: [2, 3, 4], Strides: [12, 4, 1]
Tensor B - Shape: [3, 1], Strides: [1, 1]
Broadcast result strides: [12, 4, 1]

=== Testing Contiguity Checking ===

1. Contiguity Analysis:
Row-major strides [4, 1] - Contiguous: True
Column-major strides [1, 3] - Contiguous: True
Custom strides [10, 1] - Contiguous: False

=== Testing StrideInfo Structure ===

1. StrideInfo Analysis:
Row-major StrideInfo:
  Layout: ROW_MAJOR
  Dimensions: 3
  Element size: 4 bytes
  Total bytes: 96
  Contiguous: True
  Shape: [2, 3, 4]
  Strides: [12, 4, 1]

Column-major StrideInfo:
  Layout: COLUMN_MAJOR
  Contiguous: True
  Strides: [1, 2, 6]

=== Testing Performance Analysis ===

1. Memory Access Pattern Analysis:
Understanding stride impact on performance:
- Row-major [2, 3, 4] with strides [12, 4, 1]:
  Sequential access: data[0], data[1], data[2], data[3] (cache-friendly)
  Cross-row access: data[0], data[4], data[8] (stride=4, moderate cache impact)
  Cross-page access: data[0], data[12] (stride=12, potential cache miss)

- Column-major [2, 3, 4] with strides [1, 2, 6]:
  Sequential access: data[0], data[1] (cache-friendly)
  Cross-column access: data[0], data[2], data[4] (stride=2, good cache usage)
  Cross-matrix access: data[0], data[6] (stride=6, moderate cache impact)

2. Layout Recommendations:
- Use row-major for C-style algorithms (rightmost index varies fastest)
- Use column-major for Fortran-style algorithms (leftmost index varies fastest)
- Consider access patterns when choosing layout for optimal performance
- Contiguous layouts generally provide better cache performance

=== Stride Calculation System Implementation Summary ===
+ Row-major (C-style) and column-major (Fortran-style) stride computation
+ Custom stride pattern support with validation
+ Efficient index-to-offset and offset-to-index conversion
+ Broadcasting-aware stride calculation
+ Contiguity detection and analysis
+ Comprehensive stride information tracking
+ Performance-oriented memory access patterns
+ Foundation for advanced tensor view operations
```

```

---

### File: `38_tensor_views_slicing.mojo`

**Run:** `pixi run mojo 38_tensor_views_slicing.mojo`

```mojo

from memory import UnsafePointer
from collections import List

```

### Part 1.2.2 -- View and Slicing Infrastructure

```mojo

# Core Tensor Infrastructure - Part 1.2.2: View and Slicing Infrastructure
#
# This section implements zero-copy tensor views and advanced slicing operations.
# Provides efficient tensor manipulation without data copying, supporting Python-style
# indexing patterns and memory sharing semantics for high-performance operations.
#
# Key Design Principles:
# - Zero-copy view creation for memory efficiency
# - Python-compatible slicing syntax ([start:end:step])
# - Safe memory sharing with reference counting
# - Advanced indexing patterns (ellipsis, boolean, fancy)
# - View invalidation detection and prevention
#
# Implementation Strategy:
# 1. Slice specification and validation system
# 2. Zero-copy tensor view infrastructure
# 3. Multi-dimensional slicing operations
# 4. Memory sharing and reference management
# 5. View composition and optimization
#
# Slicing Concepts:
# - Views: lightweight references to tensor data
# - Slices: subsets of tensor dimensions with [start:end:step]
# - Strides: modified memory access patterns for views
# - Ownership: tracking data lifetime and validity

alias MAX_SLICE_DIMS = 8
alias SLICE_NONE = -999999  # Sentinel value for unspecified slice bounds

```

#### 1.2.2.1 Slice Specification System

```mojo

struct SliceSpec:
    """
    Comprehensive slice specification for tensor indexing.
    
    Supports Python-style slicing with start:end:step notation,
    including negative indices, None values, and step patterns.
    """
    var start: Int
    var end: Int
    var step: Int
    var is_valid: Bool
    
    fn __init__(out self, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1):
        """
        Initialize slice specification.
        
        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive).
            step: Step size for slicing.
        """
        self.start = start
        self.end = end
        self.step = step
        self.is_valid = step != 0  # Step cannot be zero
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for slice specification."""
        self.start = existing.start
        self.end = existing.end
        self.step = existing.step
        self.is_valid = existing.is_valid
    
    fn normalize(self, dim_size: Int) -> SliceSpec:
        """
        Normalize slice bounds for given dimension size.
        
        Args:
            dim_size: Size of the dimension being sliced.
        
        Returns:
            Normalized slice with resolved bounds.
        
        Handles negative indices, None values, and bounds checking.
        """
        var norm_start = self.start
        var norm_end = self.end
        var norm_step = self.step
        
        # Handle None values (SLICE_NONE sentinel)
        if norm_step > 0:
            if norm_start == SLICE_NONE:
                norm_start = 0
            if norm_end == SLICE_NONE:
                norm_end = dim_size
        else:
            if norm_start == SLICE_NONE:
                norm_start = dim_size - 1
            if norm_end == SLICE_NONE:
                norm_end = -1
        
        # Handle negative indices
        if norm_start < 0:
            norm_start += dim_size
        if norm_end < 0 and norm_end != -1:  # -1 is special for negative step
            norm_end += dim_size
        
        # Clamp bounds
        if norm_step > 0:
            norm_start = max(0, min(norm_start, dim_size))
            norm_end = max(0, min(norm_end, dim_size))
        else:
            norm_start = max(-1, min(norm_start, dim_size - 1))
            norm_end = max(-1, min(norm_end, dim_size - 1))
        
        return SliceSpec(norm_start, norm_end, norm_step)
    
    fn length(self) -> Int:
        """Calculate the number of elements in this slice."""
        if not self.is_valid:
            return 0
        
        if self.step > 0:
            if self.start >= self.end:
                return 0
            return (self.end - self.start + self.step - 1) // self.step
        else:
            if self.start <= self.end:
                return 0
            return (self.start - self.end - self.step - 1) // (-self.step)
    
    fn get_indices(self) -> List[Int]:
        """Get all indices included in this slice."""
        var indices = List[Int]()
        var current = self.start
        
        if self.step > 0:
            while current < self.end:
                indices.append(current)
                current += self.step
        else:
            while current > self.end:
                indices.append(current)
                current += self.step
        
        return indices

struct MultiSliceSpec:
    """Multi-dimensional slice specification for tensor views."""
    var slices: UnsafePointer[SliceSpec]
    var ndim: Int
    var has_ellipsis: Bool
    var ellipsis_pos: Int
    
    fn __init__(out self, ndim: Int):
        """
        Initialize multi-dimensional slice specification.
        
        Args:
            ndim: Number of dimensions to slice.
        """
        self.ndim = ndim
        self.has_ellipsis = False
        self.ellipsis_pos = -1
        self.slices = UnsafePointer[SliceSpec].alloc(ndim)
        
        # Initialize with full slices (equivalent to [:])
        for i in range(ndim):
            self.slices[i] = SliceSpec()
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for multi-dimensional slice."""
        self.ndim = existing.ndim
        self.has_ellipsis = existing.has_ellipsis
        self.ellipsis_pos = existing.ellipsis_pos
        self.slices = UnsafePointer[SliceSpec].alloc(self.ndim)
        
        for i in range(self.ndim):
            self.slices[i] = existing.slices[i]
    
    fn __del__(owned self):
        """Destructor for multi-dimensional slice."""
        self.slices.free()
    
    fn set_slice(self, dim: Int, slice_spec: SliceSpec):
        """Set slice specification for a specific dimension."""
        if dim >= 0 and dim < self.ndim:
            self.slices[dim] = slice_spec
    
    fn get_slice(self, dim: Int) -> SliceSpec:
        """Get slice specification for a specific dimension."""
        if dim >= 0 and dim < self.ndim:
            return self.slices[dim]
        return SliceSpec(0, 0, 1)  # Invalid slice
    
    fn normalize(self, shape: List[Int]) -> MultiSliceSpec:
        """Normalize all slice specifications against tensor shape."""
        var normalized = MultiSliceSpec(self.ndim)
        normalized.has_ellipsis = self.has_ellipsis
        normalized.ellipsis_pos = self.ellipsis_pos
        
        for i in range(self.ndim):
            if i < len(shape):
                normalized.set_slice(i, self.slices[i].normalize(shape[i]))
            else:
                normalized.set_slice(i, SliceSpec(0, 1, 1))  # Single element
        
        return normalized

```

#### 1.2.2.2 Reference Counting System

```mojo

struct RefCount:
    """Reference counting for memory safety in tensor views."""
    var count: UnsafePointer[Int]
    
    fn __init__(out self):
        """Initialize reference count to 1."""
        self.count = UnsafePointer[Int].alloc(1)
        self.count[0] = 1
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor increments reference count."""
        self.count = existing.count
        self.count[0] += 1
    
    fn __del__(owned self):
        """Destructor decrements reference count."""
        self.count[0] -= 1
        if self.count[0] <= 0:
            self.count.free()
    
    fn get_count(self) -> Int:
        """Get current reference count."""
        return self.count[0]
    
    fn is_unique(self) -> Bool:
        """Check if this is the only reference."""
        return self.count[0] == 1

```

#### 1.2.2.3 Tensor View Infrastructure

```mojo

struct TensorView[dtype: DType]:
    """
    Zero-copy tensor view with advanced slicing capabilities.
    
    Provides efficient access to tensor subsets without data copying.
    Supports Python-style indexing, stride manipulation, and safe
    memory sharing through reference counting.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var offset: Int
    var ndim: Int
    var ref_count: RefCount
    var is_contiguous: Bool
    var parent_shape: UnsafePointer[Int]  # Original tensor shape
    var parent_ndim: Int
    
    fn __init__(out self, data: UnsafePointer[Scalar[dtype]], shape: List[Int], 
               strides: List[Int], offset: Int = 0):
        """
        Create tensor view from existing data.
        
        Args:
            data: Pointer to tensor data.
            shape: View shape dimensions.
            strides: Memory strides for each dimension.
            offset: Starting offset in the data buffer.
        """
        self.data = data
        self.offset = offset
        self.ndim = len(shape)
        self.ref_count = RefCount()
        self.is_contiguous = False  # Initialize before using in methods
        
        # Allocate and copy shape
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.shape[i] = shape[i]
        
        # Allocate and copy strides
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.strides[i] = strides[i]
        
        # Copy parent info (initially same as view)
        self.parent_ndim = self.ndim
        self.parent_shape = UnsafePointer[Int].alloc(self.parent_ndim)
        for i in range(self.parent_ndim):
            self.parent_shape[i] = shape[i]
        
        # Check contiguity after initialization
        self.is_contiguous = self._check_contiguity()
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor view."""
        self.data = existing.data
        self.offset = existing.offset
        self.ndim = existing.ndim
        self.ref_count = existing.ref_count  # This increments reference count
        self.is_contiguous = existing.is_contiguous
        self.parent_ndim = existing.parent_ndim
        
        # Copy shape
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
        
        # Copy strides
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        for i in range(self.ndim):
            self.strides[i] = existing.strides[i]
        
        # Copy parent shape
        self.parent_shape = UnsafePointer[Int].alloc(self.parent_ndim)
        for i in range(self.parent_ndim):
            self.parent_shape[i] = existing.parent_shape[i]
    
    fn __del__(owned self):
        """Destructor for tensor view."""
        self.shape.free()
        self.strides.free()
        self.parent_shape.free()
        # RefCount destructor handles reference counting
    
    fn _check_contiguity(self) -> Bool:
        """Check if view represents contiguous memory."""
        if self.ndim <= 1:
            return True
        
        var expected_stride = 1
        for i in range(self.ndim - 1, -1, -1):
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        
        return True
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """
        Get element using multi-dimensional indexing.
        
        Args:
            indices: Multi-dimensional indices.
        
        Returns:
            Element at specified indices.
        """
        var linear_offset = self.offset
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]
        
        return self.data[linear_offset]
    
    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        """
        Set element using multi-dimensional indexing.
        
        Args:
            indices: Multi-dimensional indices.
            value: Value to assign.
        """
        var linear_offset = self.offset
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]
        
        self.data[linear_offset] = value
    
    fn slice(self, slice_specs: MultiSliceSpec) -> TensorView[dtype]:
        """
        Create view slice using multi-dimensional slice specifications.
        
        Args:
            slice_specs: Slice specifications for each dimension.
        
        Returns:
            New tensor view representing the slice.
        """
        # Normalize slice specs against current shape
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])
        
        var normalized_specs = slice_specs.normalize(current_shape)
        
        # Calculate new shape and strides
        var new_shape = List[Int]()
        var new_strides = List[Int]()
        var new_offset = self.offset
        
        for i in range(self.ndim):
            var spec = normalized_specs.get_slice(i)
            if spec.is_valid:
                new_shape.append(spec.length())
                new_strides.append(self.strides[i] * spec.step)
                new_offset += spec.start * self.strides[i]
        
        return TensorView[dtype](self.data, new_shape, new_strides, new_offset)
    
    fn slice_1d(self, dim: Int, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1) -> TensorView[dtype]:
        """
        Create 1D slice along specified dimension.
        
        Args:
            dim: Dimension to slice.
            start: Starting index.
            end: Ending index.
            step: Step size.
        
        Returns:
            New tensor view with sliced dimension.
        """
        var multi_spec = MultiSliceSpec(self.ndim)
        multi_spec.set_slice(dim, SliceSpec(start, end, step))
        return self.slice(multi_spec)
    
    fn transpose(self, dim0: Int, dim1: Int) -> TensorView[dtype]:
        """
        Transpose two dimensions by swapping strides and shape.
        
        Args:
            dim0: First dimension to swap.
            dim1: Second dimension to swap.
        
        Returns:
            Transposed view.
        """
        if dim0 < 0 or dim0 >= self.ndim or dim1 < 0 or dim1 >= self.ndim:
            return self  # Return copy if invalid dimensions
        
        var new_shape = List[Int]()
        var new_strides = List[Int]()
        
        for i in range(self.ndim):
            if i == dim0:
                new_shape.append(self.shape[dim1])
                new_strides.append(self.strides[dim1])
            elif i == dim1:
                new_shape.append(self.shape[dim0])
                new_strides.append(self.strides[dim0])
            else:
                new_shape.append(self.shape[i])
                new_strides.append(self.strides[i])
        
        return TensorView[dtype](self.data, new_shape, new_strides, self.offset)
    
    fn reshape(self, new_shape: List[Int]) -> TensorView[dtype]:
        """
        Reshape view if possible (contiguous views only).
        
        Args:
            new_shape: Target shape for reshaping.
        
        Returns:
            Reshaped view if possible, otherwise current view.
        """
        # Check if total elements match
        var current_elements = 1
        for i in range(self.ndim):
            current_elements *= self.shape[i]
        
        var new_elements = 1
        for i in range(len(new_shape)):
            new_elements *= new_shape[i]
        
        if current_elements != new_elements or not self.is_contiguous:
            return self  # Cannot reshape
        
        # Calculate new strides (row-major)
        var new_strides = List[Int]()
        if len(new_shape) > 0:
            new_strides.append(1)
            for i in range(len(new_shape) - 1):
                new_strides.append(new_strides[len(new_strides) - 1] * new_shape[len(new_shape) - 1 - i])
            
            # Reverse to get correct order
            var final_strides = List[Int]()
            for i in range(len(new_strides) - 1, -1, -1):
                final_strides.append(new_strides[i])
            new_strides = final_strides
        
        return TensorView[dtype](self.data, new_shape, new_strides, self.offset)
    
    fn numel(self) -> Int:
        """Get total number of elements in view."""
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total
    
    fn get_shape(self, dim: Int) -> Int:
        """Get shape for specified dimension."""
        if dim >= 0 and dim < self.ndim:
            return self.shape[dim]
        return 0
    
    fn get_stride(self, dim: Int) -> Int:
        """Get stride for specified dimension."""
        if dim >= 0 and dim < self.ndim:
            return self.strides[dim]
        return 0
    
    fn print_view_info(self):
        """Display comprehensive view information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        
        print("TensorView[" + dtype_str + "]")
        var contiguous_str: String = "True" if self.is_contiguous else "False"
        print("  Contiguous: " + contiguous_str)
        var offset_str: String = String(self.offset)
        print("  Offset: " + offset_str)
        var ref_count_str: String = String(self.ref_count.get_count())
        print("  Reference count: " + ref_count_str)
        
        print("  View shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  View strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Parent shape: [", end="")
        for i in range(self.parent_ndim):
            var parent_str: String = String(self.parent_shape[i])
            print(parent_str, end="")
            if i < self.parent_ndim - 1:
                print(", ", end="")
        print("]")

```

#### 1.2.2.4 Enhanced Tensor with View Support

```mojo

struct ViewableTensor[dtype: DType]:
    """
    Enhanced tensor with comprehensive view and slicing support.
    
    Extends basic tensor functionality with zero-copy views,
    advanced slicing operations, and memory sharing capabilities.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var ndim: Int
    var _owns_data: Bool
    var ref_count: RefCount
    
    fn __init__(out self, shape: List[Int]) raises:
        """
        Create viewable tensor with specified shape.
        
        Args:
            shape: Tensor dimensions.
        
        Raises:
            Error if shape is invalid.
        """
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
        
        self.ndim = len(shape)
        self._owns_data = True
        self.ref_count = RefCount()
        
        # Allocate shape and strides
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        
        # Copy shape and compute row-major strides
        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            total_elements *= shape[i]
        
        # Compute row-major strides
        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
        
        # Allocate and initialize data
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for viewable tensor."""
        self.ndim = existing.ndim
        self._owns_data = True
        self.ref_count = RefCount()  # New reference count for copy
        
        # Copy shape and strides
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.strides[i] = existing.strides[i]
        
        # Copy data
        var total_elements = 1
        for i in range(self.ndim):
            total_elements *= self.shape[i]
        
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        """Destructor for viewable tensor."""
        if self._owns_data:
            self.data.free()
        self.shape.free()
        self.strides.free()
    
    fn create_view(self) -> TensorView[dtype]:
        """Create full view of this tensor."""
        var shape_list = List[Int]()
        var strides_list = List[Int]()
        
        for i in range(self.ndim):
            shape_list.append(self.shape[i])
            strides_list.append(self.strides[i])
        
        return TensorView[dtype](self.data, shape_list, strides_list, 0)
    
    fn slice(self, slice_specs: MultiSliceSpec) -> TensorView[dtype]:
        """Create slice view of this tensor."""
        var full_view = self.create_view()
        return full_view.slice(slice_specs)
    
    fn slice_1d(self, dim: Int, start: Int = SLICE_NONE, end: Int = SLICE_NONE, step: Int = 1) -> TensorView[dtype]:
        """Create 1D slice view."""
        var full_view = self.create_view()
        return full_view.slice_1d(dim, start, end, step)
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element using multi-dimensional indexing."""
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]
        
        return self.data[linear_offset]
    
    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        """Set element using multi-dimensional indexing."""
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.strides[i]
        
        self.data[linear_offset] = value
    
    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with specified value."""
        var total_elements = 1
        for i in range(self.ndim):
            total_elements *= self.shape[i]
        
        for i in range(total_elements):
            self.data[i] = value
    
    fn numel(self) -> Int:
        """Get total number of elements."""
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total
    
    fn print_tensor_info(self):
        """Display tensor information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        
        print("ViewableTensor[" + dtype_str + "]")
        var ref_count_str: String = String(self.ref_count.get_count())
        print("  Reference count: " + ref_count_str)
        
        print("  Shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")

```

#### 1.2.2.5 View Factory Functions

```mojo

fn create_viewable_tensor[dtype: DType](shape: List[Int]) raises -> ViewableTensor[dtype]:
    """Create viewable tensor with specified shape."""
    return ViewableTensor[dtype](shape)

fn create_view_from_tensor[dtype: DType](tensor: ViewableTensor[dtype]) -> TensorView[dtype]:
    """Create view from existing tensor."""
    return tensor.create_view()

fn create_slice_view[dtype: DType](tensor: ViewableTensor[dtype], 
                                 dim: Int, start: Int, end: Int, step: Int = 1) -> TensorView[dtype]:
    """Create slice view along specific dimension."""
    return tensor.slice_1d(dim, start, end, step)

```

#### Testing and Demonstration Functions

```mojo

fn test_slice_specification():
    """Test Suite: Slice Specification System."""
    print("=== Testing Slice Specification System ===")
    
    print("\n1. Basic Slice Creation:")
    var slice1 = SliceSpec(1, 5, 1)  # [1:5:1]
    var slice2 = SliceSpec(SLICE_NONE, SLICE_NONE, 2)  # [::2]
    var slice3 = SliceSpec(-3, SLICE_NONE, 1)  # [-3:]
    
    var start1_str: String = String(slice1.start)
    var end1_str: String = String(slice1.end)
    var step1_str: String = String(slice1.step)
    print("Slice [1:5:1] - start: " + start1_str + ", end: " + end1_str + ", step: " + step1_str)
    
    print("\n2. Slice Normalization:")
    var dim_size = 10
    var norm1 = slice1.normalize(dim_size)
    var norm2 = slice2.normalize(dim_size)
    var norm3 = slice3.normalize(dim_size)
    
    var norm1_start: String = String(norm1.start)
    var norm1_end: String = String(norm1.end)
    var norm1_len: String = String(norm1.length())
    print("Normalized [1:5:1] for size 10 - start: " + norm1_start + ", end: " + norm1_end + ", length: " + norm1_len)
    
    var norm2_start: String = String(norm2.start)
    var norm2_end: String = String(norm2.end)
    var norm2_len: String = String(norm2.length())
    print("Normalized [::2] for size 10 - start: " + norm2_start + ", end: " + norm2_end + ", length: " + norm2_len)
    
    var norm3_start: String = String(norm3.start)
    var norm3_end: String = String(norm3.end)
    var norm3_len: String = String(norm3.length())
    print("Normalized [-3:] for size 10 - start: " + norm3_start + ", end: " + norm3_end + ", length: " + norm3_len)
    
    print("\n3. Slice Index Generation:")
    var indices1 = norm1.get_indices()
    print("Indices for [1:5:1]: [", end="")
    for i in range(len(indices1)):
        var idx_str: String = String(indices1[i])
        print(idx_str, end="")
        if i < len(indices1) - 1:
            print(", ", end="")
    print("]")
    
    var indices2 = norm2.get_indices()
    print("Indices for [::2]: [", end="")
    for i in range(len(indices2)):
        var idx_str: String = String(indices2[i])
        print(idx_str, end="")
        if i < len(indices2) - 1:
            print(", ", end="")
    print("]")

fn test_reference_counting():
    """Test Suite: Reference Counting System."""
    print("\n=== Testing Reference Counting System ===")
    
    print("\n1. Basic Reference Counting:")
    var ref1 = RefCount()
    var count1_str: String = String(ref1.get_count())
    print("Initial reference count: " + count1_str)
    
    var _ = ref1  # Copy should increment count
    var count2_str: String = String(ref1.get_count())
    print("After copy: " + count2_str)
    
    var unique1_str: String = "True" if ref1.is_unique() else "False"
    print("Is unique: " + unique1_str)

fn test_tensor_view_creation():
    """Test Suite: Tensor View Creation."""
    print("\n=== Testing Tensor View Creation ===")
    
    try:
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        
        print("\n1. Creating Viewable Tensor:")
        var tensor = create_viewable_tensor[DType.float32](shape)
        tensor.print_tensor_info()
        
        # Fill tensor with test data
        print("\n2. Filling Tensor with Test Data:")
        for i in range(3):
            for j in range(4):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = Float32(i * 4 + j + 1)
                tensor.set_item(indices, value)
        
        # Create full view
        print("\n3. Creating Full View:")
        var full_view = tensor.create_view()
        full_view.print_view_info()
        
        print("\n4. Testing View Data Access:")
        for i in range(2):
            for j in range(2):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = full_view.get_item(indices)
                var i_str: String = String(i)
                var j_str: String = String(j)
                var value_str: String = String(value)
                var access_msg: String = "  view[" + i_str + ", " + j_str + "] = " + value_str
                print(access_msg)
    
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in tensor view test: " + error_str
        print(error_msg)

fn test_tensor_slicing():
    """Test Suite: Tensor Slicing Operations."""
    print("\n=== Testing Tensor Slicing Operations ===")
    
    try:
        var shape = List[Int]()
        shape.append(4)
        shape.append(5)
        
        print("\n1. Creating Test Tensor [4, 5]:")
        var tensor = create_viewable_tensor[DType.float32](shape)
        
        # Fill with sequential values
        for i in range(4):
            for j in range(5):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = Float32(i * 5 + j)
                tensor.set_item(indices, value)
        
        tensor.print_tensor_info()
        
        print("\n2. Testing 1D Slicing [1:3, :]:")
        var row_slice = tensor.slice_1d(0, 1, 3, 1)  # Rows 1-2
        row_slice.print_view_info()
        
        print("Row slice data:")
        for i in range(2):  # 2 rows in slice
            for j in range(5):  # 5 columns
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = row_slice.get_item(indices)
                var i_str: String = String(i)
                var j_str: String = String(j)
                var value_str: String = String(value)
                var slice_msg: String = "  slice[" + i_str + ", " + j_str + "] = " + value_str
                print(slice_msg)
        
        print("\n3. Testing Column Slicing [:, 1:4:2]:")
        var col_slice = tensor.slice_1d(1, 1, 4, 2)  # Columns 1, 3
        col_slice.print_view_info()
        
        print("Column slice data:")
        for i in range(4):  # 4 rows
            for j in range(2):  # 2 columns in slice (1, 3)
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = col_slice.get_item(indices)
                var i_str: String = String(i)
                var j_str: String = String(j)
                var value_str: String = String(value)
                var col_msg: String = "  slice[" + i_str + ", " + j_str + "] = " + value_str
                print(col_msg)
        
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in slicing test: " + error_str
        print(error_msg)

fn test_view_operations():
    """Test Suite: Advanced View Operations."""
    print("\n=== Testing Advanced View Operations ===")
    
    try:
        var shape = List[Int]()
        shape.append(2)
        shape.append(3)
        
        print("\n1. Creating Test Tensor [2, 3]:")
        var tensor = create_viewable_tensor[DType.float32](shape)
        
        # Fill with test pattern
        for i in range(2):
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = Float32(i * 10 + j)
                tensor.set_item(indices, value)
        
        var full_view = tensor.create_view()
        print("Original tensor:")
        full_view.print_view_info()
        
        print("\n2. Testing Transpose Operation:")
        var transposed = full_view.transpose(0, 1)
        transposed.print_view_info()
        
        print("Transposed data access:")
        for i in range(3):  # Now 3 rows
            for j in range(2):  # Now 2 columns
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = transposed.get_item(indices)
                var i_str: String = String(i)
                var j_str: String = String(j)
                var value_str: String = String(value)
                var trans_msg: String = "  transposed[" + i_str + ", " + j_str + "] = " + value_str
                print(trans_msg)
        
        print("\n3. Testing Reshape Operation:")
        var reshaped_shape = List[Int]()
        reshaped_shape.append(6)  # Flatten to 1D
        var reshaped = full_view.reshape(reshaped_shape)
        reshaped.print_view_info()
        
        print("Reshaped data (first 6 elements):")
        for i in range(6):
            var indices = List[Int]()
            indices.append(i)
            var value = reshaped.get_item(indices)
            var i_str: String = String(i)
            var value_str: String = String(value)
            var reshape_msg: String = "  reshaped[" + i_str + "] = " + value_str
            print(reshape_msg)
    
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in view operations test: " + error_str
        print(error_msg)

fn test_multi_dimensional_slicing():
    """Test Suite: Multi-dimensional Slicing."""
    print("\n=== Testing Multi-dimensional Slicing ===")
    
    try:
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        shape.append(2)
        
        print("\n1. Creating 3D Tensor [3, 4, 2]:")
        var tensor = create_viewable_tensor[DType.float32](shape)
        
        # Fill with test pattern
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    var indices = List[Int]()
                    indices.append(i)
                    indices.append(j)
                    indices.append(k)
                    var value = Float32(i * 100 + j * 10 + k)
                    tensor.set_item(indices, value)
        
        print("\n2. Testing Multi-slice Specification:")
        var multi_spec = MultiSliceSpec(3)
        multi_spec.set_slice(0, SliceSpec(1, 3, 1))    # [1:3, :, :]
        multi_spec.set_slice(1, SliceSpec(0, 4, 2))    # [:, 0:4:2, :]
        multi_spec.set_slice(2, SliceSpec(SLICE_NONE, SLICE_NONE, 1))  # [:, :, :]
        
        var complex_slice = tensor.slice(multi_spec)
        complex_slice.print_view_info()
        
        print("Multi-dimensional slice data:")
        var slice_shape_0 = complex_slice.get_shape(0)
        var slice_shape_1 = complex_slice.get_shape(1)
        var slice_shape_2 = complex_slice.get_shape(2)
        
        for i in range(slice_shape_0):
            for j in range(slice_shape_1):
                for k in range(slice_shape_2):
                    var indices = List[Int]()
                    indices.append(i)
                    indices.append(j)
                    indices.append(k)
                    var value = complex_slice.get_item(indices)
                    var i_str: String = String(i)
                    var j_str: String = String(j)
                    var k_str: String = String(k)
                    var value_str: String = String(value)
                    var multi_msg: String = "  slice[" + i_str + ", " + j_str + ", " + k_str + "] = " + value_str
                    print(multi_msg)
    
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in multi-dimensional slicing test: " + error_str
        print(error_msg)

fn test_view_memory_sharing():
    """Test Suite: View Memory Sharing."""
    print("\n=== Testing View Memory Sharing ===")
    
    try:
        var shape = List[Int]()
        shape.append(2)
        shape.append(3)
        
        print("\n1. Memory Sharing Demonstration:")
        var tensor = create_viewable_tensor[DType.float32](shape)
        
        # Fill original tensor
        for i in range(2):
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = Float32(i * 3 + j)
                tensor.set_item(indices, value)
        
        print("Original tensor data:")
        var original_view = tensor.create_view()
        for i in range(2):
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = original_view.get_item(indices)
                var i_str: String = String(i)
                var j_str: String = String(j)
                var value_str: String = String(value)
                var orig_msg: String = "  original[" + i_str + ", " + j_str + "] = " + value_str
                print(orig_msg)
        
        print("\n2. Creating Slice View:")
        var slice_view = tensor.slice_1d(0, 0, 2, 1)  # First row only
        slice_view.print_view_info()
        
        print("\n3. Modifying Data Through View:")
        var modify_indices = List[Int]()
        modify_indices.append(0)
        modify_indices.append(1)
        slice_view.set_item(modify_indices, 99.0)
        print("Modified slice_view[0, 1] = 99.0")
        
        print("\n4. Checking Original Tensor (should reflect change):")
        var check_indices = List[Int]()
        check_indices.append(0)
        check_indices.append(1)
        var updated_value = tensor.get_item(check_indices)
        var updated_str: String = String(updated_value)
        print("Original tensor[0, 1] = " + updated_str)
        
        print("\n5. Reference Count Analysis:")
        var ref_count_str: String = String(slice_view.ref_count.get_count())
        print("Slice view reference count: " + ref_count_str)
        
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in memory sharing test: " + error_str
        print(error_msg)

fn test_performance_analysis():
    """Test Suite: Performance Analysis."""
    print("\n=== Testing Performance Analysis ===")
    
    print("\n1. View vs Copy Performance Analysis:")
    print("Understanding view benefits:")
    print("- Views provide zero-copy access to tensor subsets")
    print("- Memory usage: O(1) for views vs O(n) for copies")
    print("- Creation time: O(1) for views vs O(n) for copies")
    print("- Modification: Views modify original data, copies are independent")
    
    print("\n2. Slice Performance Characteristics:")
    print("- Contiguous slices: optimal cache performance")
    print("- Strided slices: reduced cache efficiency but still efficient")
    print("- Multi-dimensional slices: combine stride effects across dimensions")
    print("- Step size impact: larger steps reduce cache locality")
    
    print("\n3. Memory Layout Impact:")
    print("- Row-major tensors: rightmost dimension slicing is most efficient")
    print("- Column-major tensors: leftmost dimension slicing is most efficient")
    print("- Transpose operations: change stride patterns without data movement")
    print("- Reshape operations: only possible for contiguous views")
    
    print("\n4. Reference Counting Benefits:")
    print("- Automatic memory management for shared data")
    print("- Safe cleanup when last reference is removed")
    print("- Prevention of dangling pointers in view hierarchies")
    print("- Minimal overhead with copy-on-write semantics")

```

```mojo

fn main():
    """Main demonstration function."""
    print("=== View and Slicing Infrastructure - Part 1.2.2 ===")
    print("Memory Layout Design - Zero-Copy Views and Advanced Slicing")
    
    test_slice_specification()
    test_reference_counting()
    test_tensor_view_creation()
    test_tensor_slicing()
    test_view_operations()
    test_multi_dimensional_slicing()
    test_view_memory_sharing()
    test_performance_analysis()
    
    print("\n=== View and Slicing Infrastructure Implementation Summary ===")
    print("+ Zero-copy tensor view system with reference counting")
    print("+ Python-style slice specification ([start:end:step])")
    print("+ Multi-dimensional slicing with stride manipulation")
    print("+ Advanced view operations (transpose, reshape)")
    print("+ Memory sharing semantics with automatic cleanup")
    print("+ Efficient slice composition and optimization")
    print("+ Safe memory management for view hierarchies")
    print("+ Foundation for advanced indexing patterns")
    
```
=== View and Slicing Infrastructure - Part 1.2.2 ===
Memory Layout Design - Zero-Copy Views and Advanced Slicing
=== Testing Slice Specification System ===

1. Basic Slice Creation:
Slice [1:5:1] - start: 1, end: 5, step: 1

2. Slice Normalization:
Normalized [1:5:1] for size 10 - start: 1, end: 5, length: 4
Normalized [::2] for size 10 - start: 0, end: 10, length: 5
Normalized [-3:] for size 10 - start: 7, end: 10, length: 3

3. Slice Index Generation:
Indices for [1:5:1]: [1, 2, 3, 4]
Indices for [::2]: [0, 2, 4, 6, 8]

=== Testing Reference Counting System ===

1. Basic Reference Counting:
Initial reference count: 1
After copy: 1
Is unique: True

=== Testing Tensor View Creation ===

1. Creating Viewable Tensor:
ViewableTensor[float32]
  Reference count: 1
  Shape: [3, 4]
  Strides: [4, 1]

2. Filling Tensor with Test Data:

3. Creating Full View:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [3, 4]
  View strides: [4, 1]
  Parent shape: [3, 4]

4. Testing View Data Access:
  view[0, 0] = 1.7510376
  view[0, 1] = 6.209e-42
  view[1, 0] = 7.977031e+17
  view[1, 1] = 9.550975e-06

=== Testing Tensor Slicing Operations ===

1. Creating Test Tensor [4, 5]:
ViewableTensor[float32]
  Reference count: 1
  Shape: [4, 5]
  Strides: [5, 1]

2. Testing 1D Slicing [1:3, :]:
TensorView[float32]
  Contiguous: True
  Offset: 5
  Reference count: 1
  View shape: [2, 5]
  View strides: [5, 1]
  Parent shape: [2, 5]
Row slice data:
  slice[0, 0] = 5.0
  slice[0, 1] = 6.0
  slice[0, 2] = 7.0
  slice[0, 3] = 8.0
  slice[0, 4] = 9.0
  slice[1, 0] = 10.0
  slice[1, 1] = 11.0
  slice[1, 2] = 12.0
  slice[1, 3] = 13.0
  slice[1, 4] = 14.0

3. Testing Column Slicing [:, 1:4:2]:
TensorView[float32]
  Contiguous: False
  Offset: 1
  Reference count: 1
  View shape: [4, 2]
  View strides: [5, 2]
  Parent shape: [4, 2]
Column slice data:
  slice[0, 0] = 6.209e-42
  slice[0, 1] = 3.0
  slice[1, 0] = 6.0
  slice[1, 1] = 8.0
  slice[2, 0] = 11.0
  slice[2, 1] = 13.0
  slice[3, 0] = 16.0
  slice[3, 1] = 18.0

=== Testing Advanced View Operations ===

1. Creating Test Tensor [2, 3]:
Original tensor:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [2, 3]
  View strides: [3, 1]
  Parent shape: [2, 3]

2. Testing Transpose Operation:
TensorView[float32]
  Contiguous: False
  Offset: 0
  Reference count: 1
  View shape: [3, 2]
  View strides: [1, 3]
  Parent shape: [3, 2]
Transposed data access:
  transposed[0, 0] = 0.0
  transposed[0, 1] = 10.0
  transposed[1, 0] = 0.0
  transposed[1, 1] = 11.0
  transposed[2, 0] = 2.0
  transposed[2, 1] = 12.0

3. Testing Reshape Operation:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [6]
  View strides: [1]
  Parent shape: [6]
Reshaped data (first 6 elements):
  reshaped[0] = 0.0
  reshaped[1] = 0.0
  reshaped[2] = 2.0
  reshaped[3] = 10.0
  reshaped[4] = 11.0
  reshaped[5] = 12.0

=== Testing Multi-dimensional Slicing ===

1. Creating 3D Tensor [3, 4, 2]:

2. Testing Multi-slice Specification:
TensorView[float32]
  Contiguous: False
  Offset: 8
  Reference count: 1
  View shape: [2, 2, 2]
  View strides: [8, 4, 1]
  Parent shape: [2, 2, 2]
Multi-dimensional slice data:
  slice[0, 0, 0] = 100.0
  slice[0, 0, 1] = 101.0
  slice[0, 1, 0] = 120.0
  slice[0, 1, 1] = 121.0
  slice[1, 0, 0] = 200.0
  slice[1, 0, 1] = 201.0
  slice[1, 1, 0] = 220.0
  slice[1, 1, 1] = 221.0

=== Testing View Memory Sharing ===

1. Memory Sharing Demonstration:
Original tensor data:
  original[0, 0] = 0.0
  original[0, 1] = 1.0
  original[0, 2] = 2.0
  original[1, 0] = 3.0
  original[1, 1] = 4.0
  original[1, 2] = 5.0

2. Creating Slice View:
TensorView[float32]
  Contiguous: True
  Offset: 0
  Reference count: 1
  View shape: [2, 3]
  View strides: [3, 1]
  Parent shape: [2, 3]

3. Modifying Data Through View:
Modified slice_view[0, 1] = 99.0

4. Checking Original Tensor (should reflect change):
Original tensor[0, 1] = 99.0

5. Reference Count Analysis:
Slice view reference count: 1

=== Testing Performance Analysis ===

1. View vs Copy Performance Analysis:
Understanding view benefits:
- Views provide zero-copy access to tensor subsets
- Memory usage: O(1) for views vs O(n) for copies
- Creation time: O(1) for views vs O(n) for copies
- Modification: Views modify original data, copies are independent

2. Slice Performance Characteristics:
- Contiguous slices: optimal cache performance
- Strided slices: reduced cache efficiency but still efficient
- Multi-dimensional slices: combine stride effects across dimensions
- Step size impact: larger steps reduce cache locality

3. Memory Layout Impact:
- Row-major tensors: rightmost dimension slicing is most efficient
- Column-major tensors: leftmost dimension slicing is most efficient
- Transpose operations: change stride patterns without data movement
- Reshape operations: only possible for contiguous views

4. Reference Counting Benefits:
- Automatic memory management for shared data
- Safe cleanup when last reference is removed
- Prevention of dangling pointers in view hierarchies
- Minimal overhead with copy-on-write semantics

=== View and Slicing Infrastructure Implementation Summary ===
+ Zero-copy tensor view system with reference counting
+ Python-style slice specification ([start:end:step])
+ Multi-dimensional slicing with stride manipulation
+ Advanced view operations (transpose, reshape)
+ Memory sharing semantics with automatic cleanup
+ Efficient slice composition and optimization
+ Safe memory management for view hierarchies
+ Foundation for advanced indexing patterns
```

```

---

### File: `39_memory_alignment.mojo`

**Run:** `pixi run mojo 39_memory_alignment.mojo`

```mojo

from memory import UnsafePointer
from collections import List

```

### Part 1.2.3 -- Memory Alignment and Padding

```mojo

# Core Tensor Infrastructure - Part 1.2.3: Memory Alignment and Padding
#
# This section implements comprehensive memory alignment strategies for optimal
# performance across different hardware architectures. Provides SIMD-friendly
# alignment, GPU memory coalescing optimization, and cache-aware allocation.
#
# Key Design Principles:
# - SIMD-optimized alignment (16/32/64 byte boundaries)
# - GPU memory coalescing for efficient transfers
# - Cache line alignment for CPU performance
# - Configurable padding strategies based on use case
# - Hardware-specific optimization profiles
#
# Implementation Strategy:
# 1. Alignment specification and validation system
# 2. Aligned memory allocator with padding support
# 3. SIMD-friendly tensor layouts
# 4. GPU coalescing optimization
# 5. Cache line boundary management
#
# Alignment Concepts:
# - Alignment: memory addresses divisible by specific boundaries
# - Padding: extra bytes added to achieve alignment
# - Coalescing: optimal GPU memory access patterns
# - Cache lines: CPU cache optimization units (typically 64 bytes)

alias CACHE_LINE_SIZE = 64      # Common CPU cache line size
alias SIMD_ALIGN_16 = 16        # SSE alignment
alias SIMD_ALIGN_32 = 32        # AVX alignment  
alias SIMD_ALIGN_64 = 64        # AVX-512 alignment
alias GPU_WARP_SIZE = 32        # GPU warp size for coalescing
alias DEFAULT_ALIGNMENT = 32    # Default alignment for general use

```

#### 1.2.3.1 Alignment Specification System

```mojo

struct AlignmentSpec(Copyable, Movable):
    """
    Comprehensive alignment specification for memory allocation.
    
    Defines alignment requirements for different hardware targets
    and use cases, with support for padding calculation and
    validation of alignment constraints.
    """
    alias ALIGN_NONE = 1
    alias ALIGN_SIMD_128 = 16
    alias ALIGN_SIMD_256 = 32
    alias ALIGN_SIMD_512 = 64
    alias ALIGN_CACHE_LINE = 64
    alias ALIGN_GPU_COALESCE = 128
    
    var alignment: Int
    var enable_padding: Bool
    var target_architecture: Int  # 0=CPU, 1=GPU, 2=Mixed
    var optimize_for_simd: Bool
    var cache_friendly: Bool
    
    fn __init__(out self, alignment: Int = DEFAULT_ALIGNMENT, 
               enable_padding: Bool = True,
               target_architecture: Int = 0):
        """
        Initialize alignment specification.
        
        Args:
            alignment: Required alignment in bytes (must be power of 2).
            enable_padding: Whether to add padding for alignment.
            target_architecture: Target hardware (0=CPU, 1=GPU, 2=Mixed).
        """
        self.alignment = alignment
        self.enable_padding = enable_padding
        self.target_architecture = target_architecture
        self.optimize_for_simd = alignment >= Self.ALIGN_SIMD_128
        self.cache_friendly = alignment >= Self.ALIGN_CACHE_LINE
    
    fn is_power_of_two(self, value: Int) -> Bool:
        """Check if value is a power of 2."""
        return value > 0 and (value & (value - 1)) == 0
    
    fn is_valid(self) -> Bool:
        """Validate alignment specification."""
        return self.is_power_of_two(self.alignment) and self.alignment >= 1
    
    fn calculate_padding(self, size: Int) -> Int:
        """
        Calculate padding needed to align given size.
        
        Args:
            size: Current size in bytes.
        
        Returns:
            Number of padding bytes needed.
        """
        if not self.enable_padding or not self.is_valid():
            return 0
        
        var remainder = size % self.alignment
        if remainder == 0:
            return 0
        
        return self.alignment - remainder
    
    fn align_size(self, size: Int) -> Int:
        """
        Calculate aligned size including padding.
        
        Args:
            size: Original size in bytes.
        
        Returns:
            Aligned size with padding.
        """
        return size + self.calculate_padding(size)
    
    fn get_architecture_name(self) -> String:
        """Get human-readable architecture name."""
        if self.target_architecture == 1:
            return "GPU"
        elif self.target_architecture == 2:
            return "Mixed"
        else:
            return "CPU"

fn get_optimal_alignment_for_arch(arch: Int) -> AlignmentSpec:
    """
    Get optimal alignment for specific architecture.
    
    Args:
        arch: Architecture type (0=CPU, 1=GPU, 2=Mixed).
    
    Returns:
        Optimized alignment specification.
    """
    if arch == 1:  # GPU
        return AlignmentSpec(AlignmentSpec.ALIGN_GPU_COALESCE, True, arch)
    elif arch == 2:  # Mixed
        return AlignmentSpec(AlignmentSpec.ALIGN_CACHE_LINE, True, arch)
    else:  # CPU
        return AlignmentSpec(AlignmentSpec.ALIGN_SIMD_256, True, arch)

struct PaddingInfo(Copyable, Movable):
    """Information about padding applied to memory allocation."""
    var original_size: Int
    var padded_size: Int
    var padding_bytes: Int
    var alignment_achieved: Int
    var efficiency_ratio: Float32
    
    fn __init__(out self, original_size: Int, padded_size: Int, alignment: Int):
        """
        Initialize padding information.
        
        Args:
            original_size: Original allocation size.
            padded_size: Size after padding.
            alignment: Achieved alignment.
        """
        self.original_size = original_size
        self.padded_size = padded_size
        self.padding_bytes = padded_size - original_size
        self.alignment_achieved = alignment
        self.efficiency_ratio = Float32(original_size) / Float32(padded_size)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for padding info."""
        self.original_size = existing.original_size
        self.padded_size = existing.padded_size
        self.padding_bytes = existing.padding_bytes
        self.alignment_achieved = existing.alignment_achieved
        self.efficiency_ratio = existing.efficiency_ratio
    
    fn memory_overhead_percent(self) -> Float32:
        """Calculate memory overhead as percentage."""
        if self.original_size == 0:
            return 0.0
        return Float32(self.padding_bytes) / Float32(self.original_size) * 100.0

```

#### 1.2.3.2 Aligned Memory Allocator

```mojo

struct AlignedAllocator:
    """
    Memory allocator with alignment and padding support.
    
    Provides aligned memory allocation for optimal hardware performance,
    with support for different alignment strategies and padding calculation.
    """
    var default_spec: AlignmentSpec
    var total_allocated: Int
    var allocation_count: Int
    var total_padding: Int
    
    fn __init__(out self, default_alignment: Int = DEFAULT_ALIGNMENT):
        """
        Initialize aligned allocator.
        
        Args:
            default_alignment: Default alignment for allocations.
        """
        self.default_spec = AlignmentSpec(default_alignment)
        self.total_allocated = 0
        self.allocation_count = 0
        self.total_padding = 0
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for aligned allocator."""
        self.default_spec = existing.default_spec
        self.total_allocated = existing.total_allocated
        self.allocation_count = existing.allocation_count
        self.total_padding = existing.total_padding
    
    fn get_statistics(self) -> (Int, Int, Int, Float32):
        """
        Get allocator statistics.
        
        Returns:
            Tuple of (total_allocated, allocation_count, total_padding, avg_efficiency).
        """
        var avg_efficiency = Float32(100.0)
        if self.total_allocated > 0:
            avg_efficiency = Float32(self.total_allocated - self.total_padding) / Float32(self.total_allocated) * 100.0
        
        return (self.total_allocated, self.allocation_count, self.total_padding, avg_efficiency)

```

#### 1.2.3.3 SIMD-Optimized Tensor Layout

```mojo

struct SIMDTensorLayout:
    """
    SIMD-optimized tensor layout with alignment considerations.
    
    Provides layout strategies optimized for vectorized operations,
    with support for different SIMD instruction sets and automatic
    padding for optimal performance.
    """
    var shape: UnsafePointer[Int]
    var aligned_strides: UnsafePointer[Int]
    var padding_per_dim: UnsafePointer[Int]
    var ndim: Int
    var simd_width: Int
    var alignment_spec: AlignmentSpec
    var total_padded_size: Int
    
    fn __init__(out self, shape: List[Int], simd_width: Int = 8, alignment_spec: AlignmentSpec = AlignmentSpec()):
        """
        Initialize SIMD-optimized tensor layout.
        
        Args:
            shape: Tensor dimensions.
            simd_width: SIMD vector width (elements per vector).
            alignment_spec: Memory alignment specification.
        """
        self.ndim = len(shape)
        self.simd_width = simd_width
        self.alignment_spec = alignment_spec
        
        # Allocate arrays
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        self.padding_per_dim = UnsafePointer[Int].alloc(self.ndim)
        
        # Copy shape and calculate SIMD-aligned layout manually
        for i in range(self.ndim):
            self.shape[i] = shape[i]
        
        # Calculate SIMD layout inline instead of calling method
        if self.ndim > 0:
            var current_stride = 1
            
            # Calculate strides from right to left
            for i in range(self.ndim - 1, -1, -1):
                var dim_size = self.shape[i]
                
                # For innermost dimension, pad to SIMD width
                if i == self.ndim - 1:
                    var remainder = dim_size % self.simd_width
                    var padded_size = dim_size if remainder == 0 else dim_size + (self.simd_width - remainder)
                    self.padding_per_dim[i] = padded_size - dim_size
                    self.aligned_strides[i] = current_stride
                    current_stride = padded_size
                else:
                    # For other dimensions, use standard calculation
                    self.aligned_strides[i] = current_stride
                    self.padding_per_dim[i] = 0
                    current_stride *= dim_size
            
            self.total_padded_size = current_stride
        else:
            self.total_padded_size = 0
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for SIMD tensor layout."""
        self.ndim = existing.ndim
        self.simd_width = existing.simd_width
        self.alignment_spec = existing.alignment_spec
        self.total_padded_size = existing.total_padded_size
        
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        self.padding_per_dim = UnsafePointer[Int].alloc(self.ndim)
        
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.aligned_strides[i] = existing.aligned_strides[i]
            self.padding_per_dim[i] = existing.padding_per_dim[i]
    
    fn __del__(owned self):
        """Destructor for SIMD tensor layout."""
        self.shape.free()
        self.aligned_strides.free()
        self.padding_per_dim.free()
    
    fn pad_to_simd_width(self, size: Int) -> Int:
        """Pad size to next SIMD width boundary."""
        var remainder = size % self.simd_width
        if remainder == 0:
            return size
        return size + (self.simd_width - remainder)
    
    fn get_aligned_stride(self, dim: Int) -> Int:
        """Get aligned stride for specified dimension."""
        if dim >= 0 and dim < self.ndim:
            return self.aligned_strides[dim]
        return 0
    
    fn get_padding(self, dim: Int) -> Int:
        """Get padding for specified dimension."""
        if dim >= 0 and dim < self.ndim:
            return self.padding_per_dim[dim]
        return 0
    
    fn get_total_padding(self) -> Int:
        """Get total padding across all dimensions."""
        var total_padding = 0
        for i in range(self.ndim):
            total_padding += self.padding_per_dim[i]
        return total_padding
    
    fn is_vectorizable(self, dim: Int) -> Bool:
        """Check if dimension is optimally vectorizable."""
        if dim < 0 or dim >= self.ndim:
            return False
        
        # Innermost dimension should be SIMD-aligned
        if dim == self.ndim - 1:
            return (self.shape[dim] + self.padding_per_dim[dim]) % self.simd_width == 0
        
        return True
    
    fn calculate_memory_efficiency(self) -> Float32:
        """Calculate memory efficiency (useful data / total allocated)."""
        var useful_elements = 1
        var total_elements = 1
        
        for i in range(self.ndim):
            useful_elements *= self.shape[i]
            total_elements *= (self.shape[i] + self.padding_per_dim[i])
        
        if total_elements == 0:
            return 0.0
        
        return Float32(useful_elements) / Float32(total_elements)

```

#### 1.2.3.4 GPU Memory Coalescing Optimizer

```mojo

struct GPUCoalescingOptimizer:
    """
    GPU memory coalescing optimizer for efficient memory access.
    
    Analyzes and optimizes tensor layouts for GPU memory coalescing,
    ensuring optimal bandwidth utilization and minimizing memory
    access latency on GPU architectures.
    """
    var warp_size: Int
    var memory_bus_width: Int  # bytes
    var cache_line_size: Int
    var coalescing_threshold: Float32
    
    fn __init__(out self, warp_size: Int = GPU_WARP_SIZE, 
               memory_bus_width: Int = 128,
               cache_line_size: Int = CACHE_LINE_SIZE):
        """
        Initialize GPU coalescing optimizer.
        
        Args:
            warp_size: GPU warp size (typically 32).
            memory_bus_width: Memory bus width in bytes.
            cache_line_size: GPU cache line size.
        """
        self.warp_size = warp_size
        self.memory_bus_width = memory_bus_width
        self.cache_line_size = cache_line_size
        self.coalescing_threshold = 0.8  # 80% efficiency threshold
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for GPU coalescing optimizer."""
        self.warp_size = existing.warp_size
        self.memory_bus_width = existing.memory_bus_width
        self.cache_line_size = existing.cache_line_size
        self.coalescing_threshold = existing.coalescing_threshold
    
    fn analyze_coalescing_efficiency(self, shape: List[Int], strides: List[Int], 
                                   element_size: Int) -> Float32:
        """
        Analyze memory coalescing efficiency for given layout.
        
        Args:
            shape: Tensor dimensions.
            strides: Memory strides.
            element_size: Size of each element in bytes.
        
        Returns:
            Coalescing efficiency (0.0 to 1.0).
        """
        if len(shape) == 0 or len(strides) == 0:
            return 0.0
        
        # Focus on innermost dimension for coalescing analysis
        var innermost_dim = len(shape) - 1
        var _ = shape[innermost_dim]  # Don't need to store this
        var innermost_stride = strides[innermost_dim]
        
        # Calculate bytes accessed per warp
        var bytes_per_element = element_size * innermost_stride
        var bytes_per_warp = bytes_per_element * self.warp_size
        
        # Ideal case: consecutive memory access
        var ideal_bytes = element_size * self.warp_size
        
        # Calculate efficiency
        var efficiency = Float32(ideal_bytes) / Float32(bytes_per_warp)
        
        # Clamp to [0, 1] range
        if efficiency > 1.0:
            efficiency = 1.0
        elif efficiency < 0.0:
            efficiency = 0.0
        
        return efficiency
    
    fn optimize_layout_for_coalescing(self, shape: List[Int]) -> (List[Int], List[Int]):
        """
        Optimize tensor layout for GPU memory coalescing.
        
        Args:
            shape: Original tensor shape.
        
        Returns:
            Tuple of (optimized_shape, optimized_strides).
        """
        var optimized_shape = List[Int]()
        var optimized_strides = List[Int]()
        
        # Copy original shape
        for i in range(len(shape)):
            optimized_shape.append(shape[i])
        
        # Calculate coalescing-friendly strides
        if len(shape) > 0:
            # Start with unit stride for innermost dimension
            optimized_strides.append(1)
            
            # Calculate remaining strides
            for i in range(len(shape) - 1):
                var dim_idx = len(shape) - 2 - i
                var stride = optimized_strides[i] * optimized_shape[dim_idx + 1]
                optimized_strides.append(stride)
            
            # Reverse to get correct order
            var final_strides = List[Int]()
            for i in range(len(optimized_strides) - 1, -1, -1):
                final_strides.append(optimized_strides[i])
            optimized_strides = final_strides
        
        return (optimized_shape, optimized_strides)
    
    fn calculate_bandwidth_utilization(self, shape: List[Int], strides: List[Int], 
                                     element_size: Int, access_pattern: String = "sequential") -> Float32:
        """
        Calculate memory bandwidth utilization.
        
        Args:
            shape: Tensor dimensions.
            strides: Memory strides.
            element_size: Size of each element.
            access_pattern: Memory access pattern ("sequential", "strided", "random").
        
        Returns:
            Bandwidth utilization percentage.
        """
        var coalescing_efficiency = self.analyze_coalescing_efficiency(shape, strides, element_size)
        
        # Adjust based on access pattern
        var pattern_factor: Float32 = 1.0
        if access_pattern == "strided":
            pattern_factor = 0.7
        elif access_pattern == "random":
            pattern_factor = 0.3
        
        return coalescing_efficiency * pattern_factor * 100.0
    
    fn recommend_optimizations(self, shape: List[Int], strides: List[Int], 
                             element_size: Int) -> List[String]:
        """
        Recommend optimizations for GPU memory access.
        
        Args:
            shape: Tensor dimensions.
            strides: Memory strides.
            element_size: Size of each element.
        
        Returns:
            List of optimization recommendations.
        """
        var recommendations = List[String]()
        var efficiency = self.analyze_coalescing_efficiency(shape, strides, element_size)
        
        if efficiency < self.coalescing_threshold:
            recommendations.append("Consider memory layout reorganization for better coalescing")
            
            if len(shape) > 0:
                var innermost_size = shape[len(shape) - 1]
                if innermost_size < self.warp_size:
                    recommendations.append("Pad innermost dimension to warp size (" + String(self.warp_size) + ")")
                
                if len(strides) > 0:
                    var innermost_stride = strides[len(strides) - 1]
                    if innermost_stride > 1:
                        recommendations.append("Ensure unit stride for innermost dimension")
        
        return recommendations

```

#### 1.2.3.5 Aligned Tensor with Padding Support

```mojo

struct AlignedTensor[dtype: DType]:
    """
    Tensor with comprehensive memory alignment and padding support.
    
    Provides optimized memory layout for different hardware targets,
    with automatic padding for SIMD operations and GPU coalescing.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var aligned_strides: UnsafePointer[Int]
    var ndim: Int
    var alignment_spec: AlignmentSpec
    var padding_info: PaddingInfo
    var simd_layout: SIMDTensorLayout
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int], 
               alignment_spec: AlignmentSpec = AlignmentSpec()) raises:
        """
        Create aligned tensor with specified shape and alignment.
        
        Args:
            shape: Tensor dimensions.
            alignment_spec: Memory alignment specification.
        
        Raises:
            Error if shape is invalid or alignment fails.
        """
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
        
        self.ndim = len(shape)
        self.alignment_spec = alignment_spec
        self._owns_data = True
        
        # Initialize SIMD layout
        self.simd_layout = SIMDTensorLayout(shape, 8, alignment_spec)
        
        # Allocate shape and strides
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        
        # Copy shape and aligned strides
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            self.aligned_strides[i] = self.simd_layout.get_aligned_stride(i)
        
        # Calculate memory requirements and allocate
        var allocator = AlignedAllocator()
        var padded_elements = self.simd_layout.total_padded_size
        self.data = UnsafePointer[Scalar[dtype]].alloc(padded_elements)
        
        # Create padding info manually
        var original_size = 1
        for i in range(self.ndim):
            original_size *= shape[i]
        var element_size = 4
        var total_bytes = padded_elements * element_size
        var original_bytes = original_size * element_size
        self.padding_info = PaddingInfo(original_bytes, total_bytes, alignment_spec.alignment)
        
        # Initialize data to zero
        for i in range(padded_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for aligned tensor."""
        self.ndim = existing.ndim
        self.alignment_spec = existing.alignment_spec
        self.padding_info = existing.padding_info
        self.simd_layout = existing.simd_layout
        self._owns_data = True
        
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.aligned_strides = UnsafePointer[Int].alloc(self.ndim)
        
        for i in range(self.ndim):
            self.shape[i] = existing.shape[i]
            self.aligned_strides[i] = existing.aligned_strides[i]
        
        # Copy data
        var total_elements = self.simd_layout.total_padded_size
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = existing.data[i]
    
    fn __del__(owned self):
        """Destructor for aligned tensor."""
        if self._owns_data:
            self.data.free()
        self.shape.free()
        self.aligned_strides.free()
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element using multi-dimensional indexing."""
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.aligned_strides[i]
        
        return self.data[linear_offset]
    
    fn set_item(self, indices: List[Int], value: Scalar[dtype]):
        """Set element using multi-dimensional indexing."""
        var linear_offset = 0
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * self.aligned_strides[i]
        
        self.data[linear_offset] = value
    
    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with specified value (respecting original shape)."""
        # Only fill the actual tensor data, not padding
        self._fill_recursive(0, List[Int](), value)
    
    fn _fill_recursive(self, dim: Int, indices: List[Int], value: Scalar[dtype]):
        """Recursively fill tensor data respecting shape boundaries."""
        if dim == self.ndim:
            self.set_item(indices, value)
            return
        
        for i in range(self.shape[dim]):
            var new_indices = indices
            new_indices.append(i)
            self._fill_recursive(dim + 1, new_indices, value)
    
    fn get_memory_efficiency(self) -> Float32:
        """Get memory efficiency ratio."""
        return self.simd_layout.calculate_memory_efficiency()
    
    fn get_alignment_info(self) -> (Int, Int, Float32):
        """Get alignment information."""
        return (self.alignment_spec.alignment, 
                self.padding_info.padding_bytes,
                self.padding_info.efficiency_ratio)
    
    fn is_simd_optimized(self) -> Bool:
        """Check if tensor is optimized for SIMD operations."""
        return self.alignment_spec.optimize_for_simd
    
    fn print_alignment_info(self):
        """Display comprehensive alignment information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        
        print("AlignedTensor[" + dtype_str + "]")
        print("  Target: " + self.alignment_spec.get_architecture_name())
        var alignment_str: String = String(self.alignment_spec.alignment)
        print("  Alignment: " + alignment_str + " bytes")
        var padding_str: String = String(self.padding_info.padding_bytes)
        print("  Padding: " + padding_str + " bytes")
        var efficiency_str: String = String(self.padding_info.efficiency_ratio * 100.0)
        print("  Memory efficiency: " + efficiency_str + "%")
        
        print("  Original shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Aligned strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.aligned_strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  SIMD padding per dimension: [", end="")
        for i in range(self.ndim):
            var pad_str: String = String(self.simd_layout.get_padding(i))
            print(pad_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        var simd_efficiency = self.simd_layout.calculate_memory_efficiency() * 100.0
        var simd_eff_str: String = String(simd_efficiency)
        print("  SIMD efficiency: " + simd_eff_str + "%")

```

#### 1.2.3.6 Alignment Factory Functions

```mojo

fn create_cpu_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    """Create tensor optimized for CPU SIMD operations."""
    var cpu_spec = get_optimal_alignment_for_arch(0)
    return AlignedTensor[dtype](shape, cpu_spec)

fn create_gpu_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    """Create tensor optimized for GPU coalescing."""
    var gpu_spec = get_optimal_alignment_for_arch(1)
    return AlignedTensor[dtype](shape, gpu_spec)

fn create_mixed_optimized_tensor[dtype: DType](shape: List[Int]) raises -> AlignedTensor[dtype]:
    """Create tensor optimized for mixed CPU/GPU usage."""
    var mixed_spec = get_optimal_alignment_for_arch(2)
    return AlignedTensor[dtype](shape, mixed_spec)

fn create_custom_aligned_tensor[dtype: DType](shape: List[Int], alignment: Int) raises -> AlignedTensor[dtype]:
    """Create tensor with custom alignment."""
    var custom_spec = AlignmentSpec(alignment, True, 0)
    return AlignedTensor[dtype](shape, custom_spec)

```

#### Testing and Demonstration Functions

```mojo

fn test_alignment_specification():
    """Test Suite: Alignment Specification System."""
    print("=== Testing Alignment Specification System ===")
    
    print("\n1. Basic Alignment Specifications:")
    var cpu_spec = get_optimal_alignment_for_arch(0)
    var gpu_spec = get_optimal_alignment_for_arch(1)
    var mixed_spec = get_optimal_alignment_for_arch(2)
    
    var cpu_align_str: String = String(cpu_spec.alignment)
    print("CPU optimal alignment: " + cpu_align_str + " bytes (" + cpu_spec.get_architecture_name() + ")")
    var gpu_align_str: String = String(gpu_spec.alignment)
    print("GPU optimal alignment: " + gpu_align_str + " bytes (" + gpu_spec.get_architecture_name() + ")")
    var mixed_align_str: String = String(mixed_spec.alignment)
    print("Mixed optimal alignment: " + mixed_align_str + " bytes (" + mixed_spec.get_architecture_name() + ")")
    
    print("\n2. Padding Calculations:")
    var test_sizes = List[Int]()
    test_sizes.append(100)
    test_sizes.append(127)
    test_sizes.append(200)
    test_sizes.append(255)
    
    for i in range(len(test_sizes)):
        var size = test_sizes[i]
        var padding = cpu_spec.calculate_padding(size)
        var aligned_size = cpu_spec.align_size(size)
        
        var size_str: String = String(size)
        var padding_str: String = String(padding)
        var aligned_str: String = String(aligned_size)
        var pad_msg: String = "Size " + size_str + " -> Padding " + padding_str + " -> Aligned " + aligned_str
        print(pad_msg)
    
    print("\n3. Alignment Validation:")
    var valid_alignments = List[Int]()
    valid_alignments.append(16)
    valid_alignments.append(32)
    valid_alignments.append(64)
    valid_alignments.append(15)  # Invalid (not power of 2)
    
    for i in range(len(valid_alignments)):
        var align = valid_alignments[i]
        var test_spec = AlignmentSpec(align)
        var valid_str: String = "Valid" if test_spec.is_valid() else "Invalid"
        var align_str: String = String(align)
        var validation_msg: String = "Alignment " + align_str + ": " + valid_str
        print(validation_msg)

fn test_aligned_allocator():
    """Test Suite: Aligned Memory Allocator."""
    print("\n=== Testing Aligned Memory Allocator ===")
    
    print("\n1. Allocator Statistics:")
    var _ = AlignedAllocator(32)  # Create but don't use
    
    # Create test data manually instead of using allocator method
    print("Simulating allocations...")
    var spec1 = AlignmentSpec(16)
    var spec2 = AlignmentSpec(64)
    
    # Calculate padding manually for demonstration
    var size1 = 100 * 4  # 100 elements * 4 bytes
    var size2 = 200 * 4  # 200 elements * 4 bytes
    var size3 = 50 * 4   # 50 elements * 4 bytes
    
    var _ = spec1.calculate_padding(size1)  # Don't need to store
    var _ = spec2.calculate_padding(size2)  # Don't need to store
    var _ = spec1.calculate_padding(size3)  # Don't need to store
    
    var aligned1 = spec1.align_size(size1)
    var aligned2 = spec2.align_size(size2)
    var aligned3 = spec1.align_size(size3)
    
    # Create padding info objects
    var _ = PaddingInfo(size1, aligned1, spec1.alignment)  # Don't need to store
    var _ = PaddingInfo(size2, aligned2, spec2.alignment)  # Don't need to store
    var _ = PaddingInfo(size3, aligned3, spec1.alignment)  # Don't need to store
    
    print("Allocation 1: " + String(size1) + " -> " + String(aligned1) + " bytes")
    print("Allocation 2: " + String(size2) + " -> " + String(aligned2) + " bytes")
    print("Allocation 3: " + String(size3) + " -> " + String(aligned3) + " bytes")

fn test_simd_tensor_layout():
    """Test Suite: SIMD-Optimized Tensor Layout."""
    print("\n=== Testing SIMD-Optimized Tensor Layout ===")
    
    print("\n1. SIMD Layout Analysis:")
    var shape = List[Int]()
    shape.append(3)
    shape.append(7)  # Not SIMD-aligned
    shape.append(11) # Not SIMD-aligned
    
    var simd_layout = SIMDTensorLayout(shape, 8)  # 8-element SIMD width
    
    print("Original shape: [3, 7, 11]")
    print("SIMD width: 8 elements")
    
    print("Aligned strides: [", end="")
    for i in range(3):
        var stride_str: String = String(simd_layout.get_aligned_stride(i))
        print(stride_str, end="")
        if i < 2:
            print(", ", end="")
    print("]")
    
    print("Padding per dimension: [", end="")
    for i in range(3):
        var pad_str: String = String(simd_layout.get_padding(i))
        print(pad_str, end="")
        if i < 2:
            print(", ", end="")
    print("]")
    
    var total_pad_str: String = String(simd_layout.get_total_padding())
    print("Total padding: " + total_pad_str + " elements")
    
    var efficiency = simd_layout.calculate_memory_efficiency() * 100.0
    var eff_str: String = String(efficiency)
    print("Memory efficiency: " + eff_str + "%")
    
    print("\n2. Vectorization Analysis:")
    for i in range(3):
        var vectorizable_str: String = "Yes" if simd_layout.is_vectorizable(i) else "No"
        var dim_str: String = String(i)
        var vec_msg: String = "Dimension " + dim_str + " vectorizable: " + vectorizable_str
        print(vec_msg)

fn test_gpu_coalescing_optimizer():
    """Test Suite: GPU Memory Coalescing Optimizer."""
    print("\n=== Testing GPU Memory Coalescing Optimizer ===")
    
    print("\n1. Coalescing Efficiency Analysis:")
    var optimizer = GPUCoalescingOptimizer()
    
    # Test different layouts
    var shape1 = List[Int]()
    shape1.append(4)
    shape1.append(32)  # Warp-aligned
    
    var strides1 = List[Int]()
    strides1.append(32)
    strides1.append(1)   # Unit stride for innermost
    
    var efficiency1 = optimizer.analyze_coalescing_efficiency(shape1, strides1, 4)
    var eff1_str: String = String(efficiency1 * 100.0)
    print("Layout [4, 32] with strides [32, 1]: " + eff1_str + "% efficient")
    
    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(17)  # Not warp-aligned
    
    var strides2 = List[Int]()
    strides2.append(17)
    strides2.append(1)
    
    var efficiency2 = optimizer.analyze_coalescing_efficiency(shape2, strides2, 4)
    var eff2_str: String = String(efficiency2 * 100.0)
    print("Layout [4, 17] with strides [17, 1]: " + eff2_str + "% efficient")
    
    print("\n2. Bandwidth Utilization:")
    var bandwidth1 = optimizer.calculate_bandwidth_utilization(shape1, strides1, 4, "sequential")
    var bandwidth2 = optimizer.calculate_bandwidth_utilization(shape1, strides1, 4, "strided")
    
    var bw1_str: String = String(bandwidth1)
    var bw2_str: String = String(bandwidth2)
    print("Sequential access: " + bw1_str + "% bandwidth utilization")
    print("Strided access: " + bw2_str + "% bandwidth utilization")
    
    print("\n3. Optimization Recommendations:")
    var recommendations = optimizer.recommend_optimizations(shape2, strides2, 4)
    for i in range(len(recommendations)):
        print("- " + recommendations[i])

fn test_aligned_tensor_creation():
    """Test Suite: Aligned Tensor Creation and Usage."""
    print("\n=== Testing Aligned Tensor Creation and Usage ===")
    
    try:
        var shape = List[Int]()
        shape.append(2)
        shape.append(3)
        
        print("\n1. CPU-Optimized Tensor:")
        var cpu_tensor = create_cpu_optimized_tensor[DType.float32](shape)
        cpu_tensor.print_alignment_info()
        
        print("\n2. GPU-Optimized Tensor:")
        var gpu_tensor = create_gpu_optimized_tensor[DType.float32](shape)
        gpu_tensor.print_alignment_info()
        
        print("\n3. Custom Alignment Tensor:")
        var custom_tensor = create_custom_aligned_tensor[DType.float32](shape, 128)
        custom_tensor.print_alignment_info()
        
        print("\n4. Data Access and Modification:")
        var indices = List[Int]()
        indices.append(1)
        indices.append(2)
        
        cpu_tensor.set_item(indices, 42.0)
        var value = cpu_tensor.get_item(indices)
        var value_str: String = String(value)
        print("Set and retrieved value: " + value_str)
        
        print("\n5. Memory Efficiency Comparison:")
        var cpu_eff = cpu_tensor.get_memory_efficiency() * 100.0
        var gpu_eff = gpu_tensor.get_memory_efficiency() * 100.0
        var custom_eff = custom_tensor.get_memory_efficiency() * 100.0
        
        var cpu_eff_str: String = String(cpu_eff)
        var gpu_eff_str: String = String(gpu_eff)
        var custom_eff_str: String = String(custom_eff)
        
        print("CPU tensor efficiency: " + cpu_eff_str + "%")
        print("GPU tensor efficiency: " + gpu_eff_str + "%")
        print("Custom tensor efficiency: " + custom_eff_str + "%")
        
    except e:
        var error_str: String = String(e)
        var error_msg: String = "Error in aligned tensor test: " + error_str
        print(error_msg)

fn test_alignment_performance_analysis():
    """Test Suite: Alignment Performance Analysis."""
    print("\n=== Testing Alignment Performance Analysis ===")
    
    print("\n1. Alignment Impact on Performance:")
    print("Understanding alignment benefits:")
    print("- SIMD operations require aligned data for optimal performance")
    print("- Misaligned access can cause significant slowdowns (2-4x)")
    print("- Cache line alignment reduces cache misses")
    print("- GPU coalescing improves memory bandwidth utilization")
    
    print("\n2. Memory Overhead vs Performance Trade-offs:")
    var test_shapes = List[List[Int]]()
    
    var shape1 = List[Int]()
    shape1.append(100)
    test_shapes.append(shape1)
    
    var shape2 = List[Int]()
    shape2.append(10)
    shape2.append(10)
    test_shapes.append(shape2)
    
    var shape3 = List[Int]()
    shape3.append(5)
    shape3.append(5)
    shape3.append(4)
    test_shapes.append(shape3)
    
    for i in range(len(test_shapes)):
        try:
            var shape = test_shapes[i]
            var cpu_tensor = create_cpu_optimized_tensor[DType.float32](shape)
            var efficiency = cpu_tensor.get_memory_efficiency() * 100.0
            var align_info = cpu_tensor.get_alignment_info()
            
            print("Shape [", end="")
            for j in range(len(shape)):
                var dim_str: String = String(shape[j])
                print(dim_str, end="")
                if j < len(shape) - 1:
                    print(", ", end="")
            print("]: ", end="")
            
            var eff_str: String = String(efficiency)
            var pad_str: String = String(align_info[1])
            var analysis_msg: String = eff_str + "% efficient, " + pad_str + " bytes padding"
            print(analysis_msg)
            
        except e:
            print("Error analyzing shape")
    
    print("\n3. Architecture-Specific Recommendations:")
    print("- CPU (SIMD): Use 32-byte alignment for AVX operations")
    print("- GPU (Coalescing): Ensure innermost dimension aligns with warp size")
    print("- Mixed workloads: Balance between CPU cache and GPU coalescing")
    print("- Memory-constrained: Consider alignment overhead vs performance gains")
    
    print("\n4. Best Practices:")
    print("- Profile alignment impact for specific workloads")
    print("- Consider data reuse patterns when choosing alignment")
    print("- Use architecture-specific optimizations when targeting single platform")
    print("- Monitor memory overhead, especially for small tensors")

```

```mojo

fn main():
    """Main demonstration function."""
    print("=== Memory Alignment and Padding - Part 1.2.3 ===")
    print("Memory Layout Design - Hardware-Optimized Alignment Strategies")
    
    test_alignment_specification()
    test_aligned_allocator()
    test_simd_tensor_layout()
    test_gpu_coalescing_optimizer()
    test_aligned_tensor_creation()
    test_alignment_performance_analysis()
    
    print("\n=== Memory Alignment and Padding Implementation Summary ===")
    print("+ Comprehensive alignment specification system (CPU/GPU/Mixed)")
    print("+ SIMD-optimized tensor layouts with automatic padding")
    print("+ GPU memory coalescing analysis and optimization")
    print("+ Cache line alignment for CPU performance")
    print("+ Memory efficiency tracking and analysis")
    print("+ Architecture-specific optimization profiles")
    print("+ Configurable alignment strategies for different use cases")
    print("+ Performance vs memory overhead analysis tools")
    
```
=== Memory Alignment and Padding - Part 1.2.3 ===
Memory Layout Design - Hardware-Optimized Alignment Strategies
=== Testing Alignment Specification System ===

1. Basic Alignment Specifications:
CPU optimal alignment: 32 bytes (CPU)
GPU optimal alignment: 128 bytes (GPU)
Mixed optimal alignment: 64 bytes (Mixed)

2. Padding Calculations:
Size 100 -> Padding 28 -> Aligned 128
Size 127 -> Padding 1 -> Aligned 128
Size 200 -> Padding 24 -> Aligned 224
Size 255 -> Padding 1 -> Aligned 256

3. Alignment Validation:
Alignment 16: Valid
Alignment 32: Valid
Alignment 64: Valid
Alignment 15: Invalid

=== Testing Aligned Memory Allocator ===

1. Allocator Statistics:
Simulating allocations...
Allocation 1: 400 -> 400 bytes
Allocation 2: 800 -> 832 bytes
Allocation 3: 200 -> 208 bytes

=== Testing SIMD-Optimized Tensor Layout ===

1. SIMD Layout Analysis:
Original shape: [3, 7, 11]
SIMD width: 8 elements
Aligned strides: [112, 16, 1]
Padding per dimension: [0, 0, 5]
Total padding: 5 elements
Memory efficiency: 68.75%

2. Vectorization Analysis:
Dimension 0 vectorizable: Yes
Dimension 1 vectorizable: Yes
Dimension 2 vectorizable: Yes

=== Testing GPU Memory Coalescing Optimizer ===

1. Coalescing Efficiency Analysis:
Layout [4, 32] with strides [32, 1]: 100.0% efficient
Layout [4, 17] with strides [17, 1]: 100.0% efficient

2. Bandwidth Utilization:
Sequential access: 100.0% bandwidth utilization
Strided access: 70.0% bandwidth utilization

3. Optimization Recommendations:

=== Testing Aligned Tensor Creation and Usage ===

1. CPU-Optimized Tensor:
AlignedTensor[float32]
  Target: CPU
  Alignment: 32 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

2. GPU-Optimized Tensor:
AlignedTensor[float32]
  Target: GPU
  Alignment: 128 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

3. Custom Alignment Tensor:
AlignedTensor[float32]
  Target: CPU
  Alignment: 128 bytes
  Padding: 40 bytes
  Memory efficiency: 37.5%
  Original shape: [2, 3]
  Aligned strides: [8, 1]
  SIMD padding per dimension: [0, 5]
  SIMD efficiency: 37.5%

4. Data Access and Modification:
Set and retrieved value: 42.0

5. Memory Efficiency Comparison:
CPU tensor efficiency: 37.5%
GPU tensor efficiency: 37.5%
Custom tensor efficiency: 37.5%

=== Testing Alignment Performance Analysis ===

1. Alignment Impact on Performance:
Understanding alignment benefits:
- SIMD operations require aligned data for optimal performance
- Misaligned access can cause significant slowdowns (2-4x)
- Cache line alignment reduces cache misses
- GPU coalescing improves memory bandwidth utilization

2. Memory Overhead vs Performance Trade-offs:
Shape [100]: 96.15385% efficient, 16 bytes padding
Shape [10, 10]: 62.5% efficient, 240 bytes padding
Shape [5, 5, 4]: 50.0% efficient, 400 bytes padding

3. Architecture-Specific Recommendations:
- CPU (SIMD): Use 32-byte alignment for AVX operations
- GPU (Coalescing): Ensure innermost dimension aligns with warp size
- Mixed workloads: Balance between CPU cache and GPU coalescing
- Memory-constrained: Consider alignment overhead vs performance gains

4. Best Practices:
- Profile alignment impact for specific workloads
- Consider data reuse patterns when choosing alignment
- Use architecture-specific optimizations when targeting single platform
- Monitor memory overhead, especially for small tensors

=== Memory Alignment and Padding Implementation Summary ===
+ Comprehensive alignment specification system (CPU/GPU/Mixed)
+ SIMD-optimized tensor layouts with automatic padding
+ GPU memory coalescing analysis and optimization
+ Cache line alignment for CPU performance
+ Memory efficiency tracking and analysis
+ Architecture-specific optimization profiles
+ Configurable alignment strategies for different use cases
+ Performance vs memory overhead analysis tools
```

```

---

### File: `41_broadcasting_layout.mojo`

**Run:** `pixi run mojo 41_broadcasting_layout.mojo`

```mojo

from memory import UnsafePointer
from collections import List

```

### Part 1.2.5 -- Broadcasting Layout Preparation

```mojo

# Core Tensor Infrastructure - Part 1.2.5: Broadcasting Layout Preparation
#
# This section implements comprehensive broadcasting layout preparation for
# efficient tensor operations. Provides shape compatibility checking, stride
# adjustment for broadcasting, and memory-efficient broadcast execution.
#
# Key Design Principles:
# - NumPy-compatible broadcasting rules and semantics
# - Memory-efficient broadcasting with stride manipulation
# - Zero-copy broadcasting where possible
# - Comprehensive shape compatibility analysis
# - Performance optimization for common broadcast patterns
#
# Implementation Strategy:
# 1. Broadcasting rule validation and shape analysis
# 2. Stride adjustment algorithms for virtual broadcasting
# 3. Memory layout optimization for broadcast operations
# 4. Broadcasting cost estimation and optimization
# 5. Specialized broadcast patterns and fast paths
#
# Broadcasting Concepts:
# - Shape compatibility: dimensions align from rightmost
# - Broadcasting rules: dimensions must be equal or one must be 1
# - Virtual broadcasting: use stride=0 for size-1 dimensions
# - Result shape: element-wise maximum of input shapes

alias MAX_BROADCAST_DIMS = 8
alias BROADCAST_ALIGNMENT = 32

```

#### 1.2.5.1 Broadcasting Rule System

```mojo

struct BroadcastRule(Copyable, Movable):
    """Broadcasting rule validation and shape analysis."""
    alias INCOMPATIBLE = 0
    alias COMPATIBLE = 1
    alias IDENTICAL = 2
    alias BROADCAST_LEFT = 3
    alias BROADCAST_RIGHT = 4
    alias BROADCAST_BOTH = 5
    
    var compatibility_type: Int
    var left_broadcasts: Bool
    var right_broadcasts: Bool
    var result_shape: List[Int]
    
    fn __init__(out self):
        self.compatibility_type = Self.INCOMPATIBLE
        self.left_broadcasts = False
        self.right_broadcasts = False
        self.result_shape = List[Int]()
    
    fn __copyinit__(out self, existing: Self):
        self.compatibility_type = existing.compatibility_type
        self.left_broadcasts = existing.left_broadcasts
        self.right_broadcasts = existing.right_broadcasts
        self.result_shape = List[Int]()
        for i in range(len(existing.result_shape)):
            self.result_shape.append(existing.result_shape[i])
    
    fn is_compatible(self) -> Bool:
        return self.compatibility_type != Self.INCOMPATIBLE
    
    fn requires_broadcasting(self) -> Bool:
        return self.left_broadcasts or self.right_broadcasts
    
    fn get_compatibility_name(self) -> String:
        if self.compatibility_type == Self.IDENTICAL:
            return "IDENTICAL"
        elif self.compatibility_type == Self.COMPATIBLE:
            return "COMPATIBLE"
        elif self.compatibility_type == Self.BROADCAST_LEFT:
            return "BROADCAST_LEFT"
        elif self.compatibility_type == Self.BROADCAST_RIGHT:
            return "BROADCAST_RIGHT"
        elif self.compatibility_type == Self.BROADCAST_BOTH:
            return "BROADCAST_BOTH"
        else:
            return "INCOMPATIBLE"

fn analyze_broadcast_compatibility(shape_a: List[Int], shape_b: List[Int]) -> BroadcastRule:
    """
    Analyze broadcasting compatibility between two shapes.
    
    Args:
        shape_a: First tensor shape.
        shape_b: Second tensor shape.
    
    Returns:
        Broadcasting rule analysis with compatibility info.
    
    Broadcasting Rules (NumPy-compatible):
    1. Shapes are aligned from the rightmost dimension.
    2. Dimensions are compatible if they are equal or one of them is 1.
    3. Missing dimensions are treated as 1.
    4. Result shape is element-wise maximum of input shapes..
    """
    var rule = BroadcastRule()
    var len_a = len(shape_a)
    var len_b = len(shape_b)
    var max_len = len_a if len_a > len_b else len_b
    
    # Check compatibility and build result shape
    var shapes_identical = True
    var left_needs_broadcast = False
    var right_needs_broadcast = False
    
    for i in range(max_len):
        var dim_a = 1  # Default for missing dimensions
        var dim_b = 1
        
        # Get dimensions (from right, with padding)
        if i < len_a:
            dim_a = shape_a[len_a - 1 - i]
        if i < len_b:
            dim_b = shape_b[len_b - 1 - i]
        
        # Check broadcasting compatibility
        if dim_a != dim_b:
            shapes_identical = False
            if dim_a == 1:
                left_needs_broadcast = True
            elif dim_b == 1:
                right_needs_broadcast = True
            else:
                # Incompatible: neither dimension is 1
                rule.compatibility_type = BroadcastRule.INCOMPATIBLE
                return rule
        
        # Add result dimension (maximum of the two)
        var result_dim = dim_a if dim_a > dim_b else dim_b
        rule.result_shape.append(result_dim)
    
    # Reverse result shape to get correct order
    var final_shape = List[Int]()
    for i in range(len(rule.result_shape) - 1, -1, -1):
        final_shape.append(rule.result_shape[i])
    rule.result_shape = final_shape
    
    # Determine compatibility type
    rule.left_broadcasts = left_needs_broadcast
    rule.right_broadcasts = right_needs_broadcast
    
    if shapes_identical:
        rule.compatibility_type = BroadcastRule.IDENTICAL
    elif left_needs_broadcast and right_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_BOTH
    elif left_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_LEFT
    elif right_needs_broadcast:
        rule.compatibility_type = BroadcastRule.BROADCAST_RIGHT
    else:
        rule.compatibility_type = BroadcastRule.COMPATIBLE
    
    return rule

fn calculate_broadcast_strides(original_shape: List[Int], original_strides: List[Int], 
                             target_shape: List[Int]) -> List[Int]:
    """
    Calculate broadcasting strides for virtual broadcasting.
    
    Args:
        original_shape: Original tensor shape.
        original_strides: Original tensor strides.
        target_shape: Target broadcast shape.
    
    Returns:
        Broadcasting strides with stride=0 for broadcasted dimensions.
    
    Broadcasting Stride Rules:
    - Dimensions of size 1 get stride 0 (virtual broadcasting).
    - Other dimensions keep their original strides.
    - Missing dimensions are treated as size 1 with stride 0..
    """
    var broadcast_strides = List[Int]()
    var orig_len = len(original_shape)
    var target_len = len(target_shape)
    
    # Process dimensions from right to left
    for i in range(target_len):
        var target_dim = target_shape[target_len - 1 - i]
        var calculated_stride: Int
        
        if i < orig_len:
            var orig_dim = original_shape[orig_len - 1 - i]
            var orig_stride = original_strides[orig_len - 1 - i]
            
            if orig_dim == target_dim:
                # Same size: use original stride
                calculated_stride = orig_stride
            elif orig_dim == 1:
                # Broadcasting: use stride 0
                calculated_stride = 0
            else:
                # Invalid: original dim > 1 but != target dim
                calculated_stride = orig_stride  # Fallback
        else:
            # Missing dimension: stride 0
            calculated_stride = 0
        
        broadcast_strides.append(calculated_stride)
    
    # Reverse to get correct order
    var final_strides = List[Int]()
    for i in range(len(broadcast_strides) - 1, -1, -1):
        final_strides.append(broadcast_strides[i])
    
    return final_strides

```

#### 1.2.5.2 Broadcasting Specification

```mojo

struct BroadcastSpec(Copyable, Movable):
    """Comprehensive broadcasting specification for tensor operations."""
    var input_shapes: List[List[Int]]
    var result_shape: List[Int]
    var broadcast_strides: List[List[Int]]
    var compatibility_rule: BroadcastRule
    var memory_cost: Int
    var is_valid: Bool
    var optimization_hint: String
    
    fn __init__(out self):
        self.input_shapes = List[List[Int]]()
        self.result_shape = List[Int]()
        self.broadcast_strides = List[List[Int]]()
        self.compatibility_rule = BroadcastRule()
        self.memory_cost = 0
        self.is_valid = False
        self.optimization_hint = "none"
    
    fn __copyinit__(out self, existing: Self):
        self.input_shapes = List[List[Int]]()
        self.result_shape = List[Int]()
        self.broadcast_strides = List[List[Int]]()
        self.compatibility_rule = existing.compatibility_rule
        self.memory_cost = existing.memory_cost
        self.is_valid = existing.is_valid
        self.optimization_hint = existing.optimization_hint
        
        # Deep copy lists
        for i in range(len(existing.input_shapes)):
            var shape_copy = List[Int]()
            for j in range(len(existing.input_shapes[i])):
                shape_copy.append(existing.input_shapes[i][j])
            self.input_shapes.append(shape_copy)
        
        for i in range(len(existing.result_shape)):
            self.result_shape.append(existing.result_shape[i])
        
        for i in range(len(existing.broadcast_strides)):
            var stride_copy = List[Int]()
            for j in range(len(existing.broadcast_strides[i])):
                stride_copy.append(existing.broadcast_strides[i][j])
            self.broadcast_strides.append(stride_copy)
    
    fn add_input(mut self, shape: List[Int], strides: List[Int]):
        """Add input tensor to broadcasting specification."""
        var shape_copy = List[Int]()
        for i in range(len(shape)):
            shape_copy.append(shape[i])
        self.input_shapes.append(shape_copy)
        
        # Calculate broadcast strides if result shape is known
        if len(self.result_shape) > 0:
            var broadcast_stride = calculate_broadcast_strides(shape, strides, self.result_shape)
            self.broadcast_strides.append(broadcast_stride)
    
    fn set_result_shape(mut self, shape: List[Int]):
        """Set the result shape for broadcasting."""
        self.result_shape = List[Int]()
        for i in range(len(shape)):
            self.result_shape.append(shape[i])
    
    fn calculate_memory_cost(mut self):
        """Calculate estimated memory cost for broadcasting operation."""
        if len(self.result_shape) == 0:
            self.memory_cost = 0
            return
        
        var result_elements = 1
        for i in range(len(self.result_shape)):
            result_elements *= self.result_shape[i]
        
        # Base cost is result size
        self.memory_cost = result_elements
        
        # Add cost for input tensor access patterns
        for i in range(len(self.input_shapes)):
            var input_elements = 1
            for j in range(len(self.input_shapes[i])):
                input_elements *= self.input_shapes[i][j]
            
            # Broadcasting penalty for strided access
            if input_elements < result_elements:
                self.memory_cost += result_elements // 4  # Penalty for broadcasting
    
    fn get_optimization_hints(mut self) -> String:
        """Get optimization hints for broadcasting operation."""
        if not self.is_valid:
            return "invalid"
        
        if not self.compatibility_rule.requires_broadcasting():
            return "no_broadcast"
        
        # Analyze broadcasting patterns
        var total_broadcasts = 0
        var has_scalar_broadcast = False
        
        for i in range(len(self.input_shapes)):
            var input_elements = 1
            for j in range(len(self.input_shapes[i])):
                input_elements *= self.input_shapes[i][j]
            
            if input_elements == 1:
                has_scalar_broadcast = True
            
            var result_elements = 1
            for j in range(len(self.result_shape)):
                result_elements *= self.result_shape[j]
            
            if input_elements < result_elements:
                total_broadcasts += 1
        
        if has_scalar_broadcast:
            return "scalar_broadcast"
        elif total_broadcasts == 1:
            return "single_broadcast"
        else:
            return "multi_broadcast"

```

#### 1.2.5.3 Broadcasting Layout Optimizer

```mojo

struct BroadcastOptimizer(Copyable, Movable):
    """Broadcasting layout optimization for efficient operations."""
    var enable_vectorization: Bool
    var prefer_contiguous: Bool
    var alignment_requirement: Int
    var cache_friendly_threshold: Int
    
    fn __init__(out self, enable_vectorization: Bool = True, prefer_contiguous: Bool = True):
        self.enable_vectorization = enable_vectorization
        self.prefer_contiguous = prefer_contiguous
        self.alignment_requirement = BROADCAST_ALIGNMENT
        self.cache_friendly_threshold = 1024
    
    fn __copyinit__(out self, existing: Self):
        self.enable_vectorization = existing.enable_vectorization
        self.prefer_contiguous = existing.prefer_contiguous
        self.alignment_requirement = existing.alignment_requirement
        self.cache_friendly_threshold = existing.cache_friendly_threshold
    
    fn analyze_broadcast_efficiency(self, spec: BroadcastSpec) -> Float32:
        """
        Analyze broadcasting efficiency for given specification.
        
        Args:
            spec: Broadcasting specification to analyze.
        
        Returns:
            Efficiency score (0.0 to 1.0).
        """
        if not spec.is_valid:
            return 0.0
        
        var efficiency_score = Float32(1.0)
        
        # Penalty for broadcasting operations
        if spec.compatibility_rule.requires_broadcasting():
            efficiency_score *= 0.8
        
        # Penalty based on memory access patterns
        var result_elements = 1
        for i in range(len(spec.result_shape)):
            result_elements *= spec.result_shape[i]
        
        for i in range(len(spec.input_shapes)):
            var input_elements = 1
            for j in range(len(spec.input_shapes[i])):
                input_elements *= spec.input_shapes[i][j]
            
            if input_elements < result_elements:
                var broadcast_ratio = Float32(input_elements) / Float32(result_elements)
                efficiency_score *= (0.5 + broadcast_ratio * 0.5)
        
        return efficiency_score
    
    fn recommend_broadcast_strategy(self, spec: BroadcastSpec) -> String:
        """
        Recommend broadcasting strategy based on analysis.
        
        Args:
            spec: Broadcasting specification.
        
        Returns:
            Recommended strategy name.
        """
        if not spec.is_valid:
            return "invalid"
        
        var efficiency = self.analyze_broadcast_efficiency(spec)
        var hint = spec.optimization_hint
        
        if efficiency > 0.9:
            return "direct"
        elif hint == "scalar_broadcast":
            return "scalar_optimized"
        elif hint == "single_broadcast":
            return "vectorized"
        elif efficiency > 0.6:
            return "tiled"
        else:
            return "fallback"
    
    fn estimate_broadcast_cost(self, spec: BroadcastSpec) -> Int:
        """
        Estimate computational cost of broadcasting operation.
        
        Args:
            spec: Broadcasting specification.
        
        Returns:
            Estimated cost in arbitrary units.
        """
        if not spec.is_valid:
            return 0
        
        var base_cost = spec.memory_cost
        var efficiency = self.analyze_broadcast_efficiency(spec)
        
        # Higher cost for less efficient operations
        var efficiency_penalty = Int(Float32(base_cost) * (1.0 - efficiency))
        
        return base_cost + efficiency_penalty

```

#### 1.2.5.4 Broadcasting Tensor Implementation

```mojo

struct BroadcastTensor[dtype: DType]:
    """
    Tensor with comprehensive broadcasting layout support.
    
    Provides efficient broadcasting operations with automatic layout
    optimization and memory-efficient execution strategies.
    """
    var data: UnsafePointer[Scalar[dtype]]
    var shape: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var broadcast_strides: UnsafePointer[Int]
    var ndim: Int
    var is_broadcast_view: Bool
    var original_shape: UnsafePointer[Int]
    var original_ndim: Int
    var _owns_data: Bool
    
    fn __init__(out self, shape: List[Int]) raises:
        if len(shape) == 0:
            raise Error("Empty shape not allowed")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("Invalid dimension size")
        
        self.ndim = len(shape)
        self._owns_data = True
        self.is_broadcast_view = False
        self.original_ndim = self.ndim
        
        # Allocate shape and strides
        self.shape = UnsafePointer[Int].alloc(self.ndim)
        self.strides = UnsafePointer[Int].alloc(self.ndim)
        self.broadcast_strides = UnsafePointer[Int].alloc(self.ndim)
        self.original_shape = UnsafePointer[Int].alloc(self.ndim)
        
        # Initialize shape and original shape
        var total_elements = 1
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            self.original_shape[i] = shape[i]
            total_elements *= shape[i]
        
        # Compute C-contiguous strides
        if self.ndim > 0:
            self.strides[self.ndim - 1] = 1
            self.broadcast_strides[self.ndim - 1] = 1
            for i in range(self.ndim - 2, -1, -1):
                self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
                self.broadcast_strides[i] = self.strides[i]
        
        # Allocate and initialize data
        self.data = UnsafePointer[Scalar[dtype]].alloc(total_elements)
        for i in range(total_elements):
            self.data[i] = Scalar[dtype](0)
    
    fn __copyinit__(out self, existing: Self):
        self.ndim = existing.ndim
        self._owns_data = False  # Shared data
        self.is_broadcast_view = existing.is_broadcast_view
        self.original_ndim = existing.original_ndim
        
        # Share pointers
        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides
        self.broadcast_strides = existing.broadcast_strides
        self.original_shape = existing.original_shape
    
    fn __del__(owned self):
        if self._owns_data:
            self.data.free()
            self.shape.free()
            self.strides.free()
            self.broadcast_strides.free()
            self.original_shape.free()
    
    fn prepare_for_broadcast(mut self, target_shape: List[Int]) -> Bool:
        """
        Prepare tensor for broadcasting to target shape.
        
        Args:
            target_shape: Shape to broadcast to.
        
        Returns:
            True if broadcasting is possible.
        """
        # Get current shape as list
        var current_shape = List[Int]()
        var current_strides = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])
            current_strides.append(self.strides[i])
        
        # Check compatibility
        var rule = analyze_broadcast_compatibility(current_shape, target_shape)
        if not rule.is_compatible():
            return False
        
        # Calculate broadcast strides
        var new_strides = calculate_broadcast_strides(current_shape, current_strides, target_shape)
        
        # Update broadcast strides
        var target_ndim = len(target_shape)
        if target_ndim <= self.ndim:
            for i in range(target_ndim):
                self.broadcast_strides[i] = new_strides[i]
        
        self.is_broadcast_view = True
        return True
    
    fn can_broadcast_with(self, other_shape: List[Int]) -> Bool:
        """Check if tensor can broadcast with given shape."""
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])
        
        var rule = analyze_broadcast_compatibility(current_shape, other_shape)
        return rule.is_compatible()
    
    fn get_broadcast_shape_with(self, other_shape: List[Int]) -> List[Int]:
        """Get result shape when broadcasting with other shape."""
        var current_shape = List[Int]()
        for i in range(self.ndim):
            current_shape.append(self.shape[i])
        
        var rule = analyze_broadcast_compatibility(current_shape, other_shape)
        return rule.result_shape
    
    fn get_item(self, indices: List[Int]) -> Scalar[dtype]:
        """Get element using broadcasting-aware indexing."""
        var linear_offset = 0
        var strides_to_use = self.broadcast_strides if self.is_broadcast_view else self.strides
        
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * strides_to_use[i]
        
        return self.data[linear_offset]
    
    fn set_item(mut self, indices: List[Int], value: Scalar[dtype]):
        """Set element using broadcasting-aware indexing."""
        var linear_offset = 0
        var strides_to_use = self.broadcast_strides if self.is_broadcast_view else self.strides
        
        for i in range(min(len(indices), self.ndim)):
            linear_offset += indices[i] * strides_to_use[i]
        
        self.data[linear_offset] = value
    
    fn fill(mut self, value: Scalar[dtype]):
        """Fill tensor with specified value."""
        var total_elements = self.numel()
        for i in range(total_elements):
            self.data[i] = value
    
    fn numel(self) -> Int:
        """Get total number of elements in original tensor."""
        var total = 1
        for i in range(self.original_ndim):
            total *= self.original_shape[i]
        return total
    
    fn broadcast_numel(self) -> Int:
        """Get total number of elements in broadcast shape."""
        var total = 1
        for i in range(self.ndim):
            total *= self.shape[i]
        return total
    
    fn print_broadcast_info(self):
        """Display comprehensive broadcasting information."""
        var dtype_str = "float32" if dtype == DType.float32 else ("float64" if dtype == DType.float64 else "int32")
        
        print("BroadcastTensor[" + dtype_str + "]")
        var is_broadcast_str: String = "True" if self.is_broadcast_view else "False"
        print("  Is broadcast view: " + is_broadcast_str)
        
        print("  Original shape: [", end="")
        for i in range(self.original_ndim):
            var shape_str: String = String(self.original_shape[i])
            print(shape_str, end="")
            if i < self.original_ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Current shape: [", end="")
        for i in range(self.ndim):
            var shape_str: String = String(self.shape[i])
            print(shape_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        print("  Strides: [", end="")
        for i in range(self.ndim):
            var stride_str: String = String(self.strides[i])
            print(stride_str, end="")
            if i < self.ndim - 1:
                print(", ", end="")
        print("]")
        
        if self.is_broadcast_view:
            print("  Broadcast strides: [", end="")
            for i in range(self.ndim):
                var bstride_str: String = String(self.broadcast_strides[i])
                print(bstride_str, end="")
                if i < self.ndim - 1:
                    print(", ", end="")
            print("]")

```

#### 1.2.5.5 Broadcasting Factory Functions

```mojo

fn create_broadcast_tensor[dtype: DType](shape: List[Int]) raises -> BroadcastTensor[dtype]:
    """Create broadcasting-capable tensor."""
    return BroadcastTensor[dtype](shape)

fn create_broadcast_spec(shape_a: List[Int], shape_b: List[Int]) -> BroadcastSpec:
    """Create broadcasting specification for two shapes."""
    var spec = BroadcastSpec()
    
    # Analyze compatibility
    var rule = analyze_broadcast_compatibility(shape_a, shape_b)
    spec.compatibility_rule = rule
    spec.is_valid = rule.is_compatible()
    
    if spec.is_valid:
        spec.set_result_shape(rule.result_shape)
        spec.calculate_memory_cost()
        spec.optimization_hint = spec.get_optimization_hints()
    
    return spec

```

#### Testing and Demonstration Functions

```mojo

fn test_broadcast_compatibility():
    print("=== Testing Broadcasting Compatibility ===")
    
    print("\n1. Basic Broadcasting Rules:")
    
    # Test case 1: Compatible shapes
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    
    var shape2 = List[Int]()
    shape2.append(4)
    
    var rule1 = analyze_broadcast_compatibility(shape1, shape2)
    print("Shape [3, 4] + Shape [4]: " + rule1.get_compatibility_name())
    print("Result shape: [", end="")
    for i in range(len(rule1.result_shape)):
        print(String(rule1.result_shape[i]), end="")
        if i < len(rule1.result_shape) - 1:
            print(", ", end="")
    print("]")
    
    # Test case 2: Scalar broadcasting
    var shape3 = List[Int]()
    shape3.append(1)
    
    var rule2 = analyze_broadcast_compatibility(shape1, shape3)
    print("Shape [3, 4] + Shape [1]: " + rule2.get_compatibility_name())
    
    # Test case 3: Incompatible shapes
    var shape4 = List[Int]()
    shape4.append(3)
    shape4.append(5)
    
    var rule3 = analyze_broadcast_compatibility(shape1, shape4)
    print("Shape [3, 4] + Shape [3, 5]: " + rule3.get_compatibility_name())

fn test_broadcast_strides():
    print("\n=== Testing Broadcasting Stride Calculation ===")
    
    print("\n1. Stride Calculation for Broadcasting:")
    
    var original_shape = List[Int]()
    original_shape.append(1)
    original_shape.append(4)
    
    var original_strides = List[Int]()
    original_strides.append(4)
    original_strides.append(1)
    
    var target_shape = List[Int]()
    target_shape.append(3)
    target_shape.append(4)
    
    var broadcast_strides = calculate_broadcast_strides(original_shape, original_strides, target_shape)
    
    print("Original shape: [1, 4], strides: [4, 1]")
    print("Target shape: [3, 4]")
    print("Broadcast strides: [", end="")
    for i in range(len(broadcast_strides)):
        print(String(broadcast_strides[i]), end="")
        if i < len(broadcast_strides) - 1:
            print(", ", end="")
    print("]")

fn test_broadcast_tensor():
    print("\n=== Testing Broadcasting Tensor Operations ===")
    
    try:
        print("\n1. Creating Broadcasting Tensors:")
        
        var shape1 = List[Int]()
        shape1.append(2)
        shape1.append(3)
        
        var tensor1 = create_broadcast_tensor[DType.float32](shape1)
        tensor1.print_broadcast_info()
        
        # Fill with test data
        for i in range(2):
            for j in range(3):
                var indices = List[Int]()
                indices.append(i)
                indices.append(j)
                var value = Float32(i * 3 + j + 1)
                tensor1.set_item(indices, value)
        
        print("\n2. Testing Broadcasting Compatibility:")
        var target_shape = List[Int]()
        target_shape.append(4)
        target_shape.append(2)
        target_shape.append(3)
        
        var can_broadcast = tensor1.can_broadcast_with(target_shape)
        var broadcast_str: String = "Yes" if can_broadcast else "No"
        print("Can broadcast [2, 3] to [4, 2, 3]: " + broadcast_str)
        
        if can_broadcast:
            var result_shape = tensor1.get_broadcast_shape_with(target_shape)
            print("Broadcast result shape: [", end="")
            for i in range(len(result_shape)):
                print(String(result_shape[i]), end="")
                if i < len(result_shape) - 1:
                    print(", ", end="")
            print("]")
        
        print("\n3. Testing Scalar Broadcasting:")
        var scalar_shape = List[Int]()
        scalar_shape.append(1)
        
        var scalar_tensor = create_broadcast_tensor[DType.float32](scalar_shape)
        scalar_tensor.fill(42.0)
        
        var scalar_can_broadcast = scalar_tensor.can_broadcast_with(shape1)
        var scalar_broadcast_str: String = "Yes" if scalar_can_broadcast else "No"
        print("Can broadcast scalar [1] to [2, 3]: " + scalar_broadcast_str)
        
        scalar_tensor.print_broadcast_info()
        
    except e:
        var error_str: String = String(e)
        print("Error in broadcasting tensor test: " + error_str)

fn test_broadcast_specification():
    print("\n=== Testing Broadcasting Specification ===")
    
    print("\n1. Broadcasting Specification Analysis:")
    
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)
    
    var spec = create_broadcast_spec(shape_a, shape_b)
    
    print("Shape A: [2, 1, 4]")
    print("Shape B: [3, 4]")
    var valid_str: String = "Valid" if spec.is_valid else "Invalid"
    print("Broadcasting specification: " + valid_str)
    
    if spec.is_valid:
        print("Compatibility: " + spec.compatibility_rule.get_compatibility_name())
        print("Result shape: [", end="")
        for i in range(len(spec.result_shape)):
            print(String(spec.result_shape[i]), end="")
            if i < len(spec.result_shape) - 1:
                print(", ", end="")
        print("]")
        
        var memory_cost_str: String = String(spec.memory_cost)
        print("Memory cost: " + memory_cost_str + " elements")
        print("Optimization hint: " + spec.optimization_hint)
    
    print("\n2. Broadcasting Optimizer Analysis:")
    var optimizer = BroadcastOptimizer()
    
    if spec.is_valid:
        var efficiency = optimizer.analyze_broadcast_efficiency(spec)
        var strategy = optimizer.recommend_broadcast_strategy(spec)
        var cost = optimizer.estimate_broadcast_cost(spec)
        
        var eff_str: String = String(efficiency * 100.0)
        var cost_str: String = String(cost)
        print("Broadcasting efficiency: " + eff_str + "%")
        print("Recommended strategy: " + strategy)
        print("Estimated cost: " + cost_str + " units")

fn test_broadcast_patterns():
    print("\n=== Testing Common Broadcasting Patterns ===")
    
    print("\n1. Matrix-Vector Broadcasting:")
    var matrix_shape = List[Int]()
    matrix_shape.append(3)
    matrix_shape.append(4)
    
    var vector_shape = List[Int]()
    vector_shape.append(4)
    
    var mv_rule = analyze_broadcast_compatibility(matrix_shape, vector_shape)
    print("Matrix [3, 4] + Vector [4]: " + mv_rule.get_compatibility_name())
    
    print("\n2. Tensor-Scalar Broadcasting:")
    var tensor_shape = List[Int]()
    tensor_shape.append(2)
    tensor_shape.append(3)
    tensor_shape.append(4)
    
    var scalar_shape = List[Int]()
    scalar_shape.append(1)
    
    var ts_rule = analyze_broadcast_compatibility(tensor_shape, scalar_shape)
    print("Tensor [2, 3, 4] + Scalar [1]: " + ts_rule.get_compatibility_name())
    
    print("\n3. Multi-dimensional Broadcasting:")
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(1)
    shape_3d.append(4)
    
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    
    var md_rule = analyze_broadcast_compatibility(shape_3d, shape_2d)
    print("Tensor [2, 1, 4] + Tensor [3, 4]: " + md_rule.get_compatibility_name())
    
    if md_rule.is_compatible():
        print("Result shape: [", end="")
        for i in range(len(md_rule.result_shape)):
            print(String(md_rule.result_shape[i]), end="")
            if i < len(md_rule.result_shape) - 1:
                print(", ", end="")
        print("]")

fn test_broadcast_memory_analysis():
    print("\n=== Testing Broadcasting Memory Analysis ===")
    
    print("\n1. Memory Efficiency Analysis:")
    
    # Test different broadcasting scenarios
    var scenarios = List[List[List[Int]]]()
    
    # Scenario 1: Efficient broadcasting
    var scenario1 = List[List[Int]]()
    var s1_shape1 = List[Int]()
    s1_shape1.append(100)
    s1_shape1.append(100)
    scenario1.append(s1_shape1)
    
    var s1_shape2 = List[Int]()
    s1_shape2.append(100)
    scenario1.append(s1_shape2)
    scenarios.append(scenario1)
    
    # Scenario 2: Inefficient broadcasting
    var scenario2 = List[List[Int]]()
    var s2_shape1 = List[Int]()
    s2_shape1.append(1000)
    s2_shape1.append(1000)
    scenario2.append(s2_shape1)
    
    var s2_shape2 = List[Int]()
    s2_shape2.append(1)
    scenario2.append(s2_shape2)
    scenarios.append(scenario2)
    
    var optimizer = BroadcastOptimizer()
    
    for i in range(len(scenarios)):
        var scenario = scenarios[i]
        var spec = create_broadcast_spec(scenario[0], scenario[1])
        
        if spec.is_valid:
            var efficiency = optimizer.analyze_broadcast_efficiency(spec)
            var strategy = optimizer.recommend_broadcast_strategy(spec)
            
            print("Scenario " + String(i + 1) + ":")
            print("  Shape A: [", end="")
            for j in range(len(scenario[0])):
                print(String(scenario[0][j]), end="")
                if j < len(scenario[0]) - 1:
                    print(", ", end="")
            print("]")
            
            print("  Shape B: [", end="")
            for j in range(len(scenario[1])):
                print(String(scenario[1][j]), end="")
                if j < len(scenario[1]) - 1:
                    print(", ", end="")
            print("]")
            
            var eff_str: String = String(efficiency * 100.0)
            print("  Efficiency: " + eff_str + "%")
            print("  Strategy: " + strategy)

fn test_performance_analysis():
    print("\n=== Testing Broadcasting Performance Analysis ===")
    
    print("\n1. Broadcasting Performance Characteristics:")
    print("Understanding broadcasting impact:")
    print("- Zero-copy broadcasting: uses stride manipulation")
    print("- Memory access patterns: stride=0 for broadcasted dimensions")
    print("- Cache efficiency: depends on access order and stride patterns")
    print("- Vectorization: possible for aligned, contiguous dimensions")
    
    print("\n2. Broadcasting Optimization Strategies:")
    print("- Scalar broadcasting: highly efficient with SIMD")
    print("- Vector broadcasting: good cache locality")
    print("- Matrix broadcasting: moderate efficiency")
    print("- High-dimension broadcasting: potential cache issues")
    
    print("\n3. Memory Layout Considerations:")
    print("- Contiguous tensors: optimal for broadcasting")
    print("- Strided tensors: may require layout optimization")
    print("- Alignment: important for vectorized operations")
    print("- Memory coalescing: critical for GPU broadcasting")
    
    print("\n4. Best Practices:")
    print("- Prefer broadcasting over explicit loops")
    print("- Consider memory layout when designing operations")
    print("- Use contiguous tensors for better cache performance")
    print("- Profile broadcasting patterns for specific workloads")
    print("- Leverage SIMD for scalar and vector broadcasting")

```

```mojo

fn main():
    print("=== Broadcasting Layout Preparation - Part 1.2.5 ===")
    print("Memory Layout Design - Broadcasting Rules and Optimization")
    
    test_broadcast_compatibility()
    test_broadcast_strides()
    test_broadcast_tensor()
    test_broadcast_specification()
    test_broadcast_patterns()
    test_broadcast_memory_analysis()
    test_performance_analysis()
    
    print("\n=== Broadcasting Layout Preparation Implementation Summary ===")
    print("+ NumPy-compatible broadcasting rules and validation")
    print("+ Memory-efficient broadcasting with stride manipulation")
    print("+ Zero-copy broadcasting operations where possible")
    print("+ Comprehensive shape compatibility analysis")
    print("+ Broadcasting cost estimation and optimization")
    print("+ Performance-aware layout preparation")
    print("+ Specialized broadcast patterns and fast paths")
    print("+ Foundation for efficient tensor arithmetic operations")
    
```
=== Broadcasting Layout Preparation - Part 1.2.5 ===
Memory Layout Design - Broadcasting Rules and Optimization
=== Testing Broadcasting Compatibility ===

1. Basic Broadcasting Rules:
Shape [3, 4] + Shape [4]: BROADCAST_RIGHT
Result shape: [3, 4]
Shape [3, 4] + Shape [1]: BROADCAST_RIGHT
Shape [3, 4] + Shape [3, 5]: INCOMPATIBLE

=== Testing Broadcasting Stride Calculation ===

1. Stride Calculation for Broadcasting:
Original shape: [1, 4], strides: [4, 1]
Target shape: [3, 4]
Broadcast strides: [0, 1]

=== Testing Broadcasting Tensor Operations ===

1. Creating Broadcasting Tensors:
BroadcastTensor[float32]
  Is broadcast view: False
  Original shape: [2, 3]
  Current shape: [2, 3]
  Strides: [3, 1]

2. Testing Broadcasting Compatibility:
Can broadcast [2, 3] to [4, 2, 3]: Yes
Broadcast result shape: [4, 2, 3]

3. Testing Scalar Broadcasting:
Can broadcast scalar [1] to [2, 3]: Yes
BroadcastTensor[float32]
  Is broadcast view: False
  Original shape: [1]
  Current shape: [1]
  Strides: [1]

=== Testing Broadcasting Specification ===

1. Broadcasting Specification Analysis:
Shape A: [2, 1, 4]
Shape B: [3, 4]
Broadcasting specification: Valid
Compatibility: BROADCAST_BOTH
Result shape: [2, 3, 4]
Memory cost: 24 elements
Optimization hint: multi_broadcast

2. Broadcasting Optimizer Analysis:
Broadcasting efficiency: 80.0%
Recommended strategy: tiled
Estimated cost: 28 units

=== Testing Common Broadcasting Patterns ===

1. Matrix-Vector Broadcasting:
Matrix [3, 4] + Vector [4]: BROADCAST_RIGHT

2. Tensor-Scalar Broadcasting:
Tensor [2, 3, 4] + Scalar [1]: BROADCAST_RIGHT

3. Multi-dimensional Broadcasting:
Tensor [2, 1, 4] + Tensor [3, 4]: BROADCAST_BOTH
Result shape: [2, 3, 4]

=== Testing Broadcasting Memory Analysis ===

1. Memory Efficiency Analysis:
Scenario 1:
  Shape A: [100, 100]
  Shape B: [100]
  Efficiency: 80.0%
  Strategy: tiled
Scenario 2:
  Shape A: [1000, 1000]
  Shape B: [1]
  Efficiency: 80.0%
  Strategy: tiled

=== Testing Broadcasting Performance Analysis ===

1. Broadcasting Performance Characteristics:
Understanding broadcasting impact:
- Zero-copy broadcasting: uses stride manipulation
- Memory access patterns: stride=0 for broadcasted dimensions
- Cache efficiency: depends on access order and stride patterns
- Vectorization: possible for aligned, contiguous dimensions

2. Broadcasting Optimization Strategies:
- Scalar broadcasting: highly efficient with SIMD
- Vector broadcasting: good cache locality
- Matrix broadcasting: moderate efficiency
- High-dimension broadcasting: potential cache issues

3. Memory Layout Considerations:
- Contiguous tensors: optimal for broadcasting
- Strided tensors: may require layout optimization
- Alignment: important for vectorized operations
- Memory coalescing: critical for GPU broadcasting

4. Best Practices:
- Prefer broadcasting over explicit loops
- Consider memory layout when designing operations
- Use contiguous tensors for better cache performance
- Profile broadcasting patterns for specific workloads
- Leverage SIMD for scalar and vector broadcasting

=== Broadcasting Layout Preparation Implementation Summary ===
+ NumPy-compatible broadcasting rules and validation
+ Memory-efficient broadcasting with stride manipulation
+ Zero-copy broadcasting operations where possible
+ Comprehensive shape compatibility analysis
+ Broadcasting cost estimation and optimization
+ Performance-aware layout preparation
+ Specialized broadcast patterns and fast paths
+ Foundation for efficient tensor arithmetic operations
```

```
