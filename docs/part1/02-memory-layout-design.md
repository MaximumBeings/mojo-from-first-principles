# 1.2 Memory Layout Design

Shape answers how many logical coordinates exist; layout answers where each coordinate lives. Keeping them separate lets a transpose, slice, or broadcast share storage while presenting a different logical tensor.

## 1.2.1 Contiguity is a checkable property

A row-major view is contiguous when its strides equal the canonical right-to-left products, ignoring axes of length one. Contiguity determines whether reshape can be metadata-only.

```mojo
def is_row_major_contiguous(shape: Shape, strides: List[Int]) -> Bool:
    var expected = 1
    var axis = len(shape.dims)
    while axis > 0:
        axis -= 1
        if shape.dims[axis] != 1 and strides[axis] != expected:
            return False
        expected *= shape.dims[axis]
    return True
```

**Manual worked example.** Shape `[2,3]` with strides `[3,1]` is contiguous: expected strides are 1 then 3. Strides `[1,2]` describe a transpose-like walk and fail at the last axis because 2 is not 1.

## 1.2.2 Views carry an offset and shared storage identity

A view must not own a second copy of the data. In a complete implementation the storage field is a reference-counted owner handle; this excerpt isolates the layout metadata the view adds.

```mojo
@fieldwise_init
struct ViewLayout(Copyable, Movable):
    var shape: Shape
    var strides: List[Int]
    var offset: Int

def view_offset(layout: ViewLayout, indices: List[Int]) raises -> Int:
    var result = layout.offset
    for axis in range(len(indices)):
        if indices[axis] < 0 or indices[axis] >= layout.shape.dims[axis]:
            raise Error("view index out of bounds")
        result += indices[axis] * layout.strides[axis]
    return result
```

**Manual worked example.** A 2×2 view into a 3×3 row-major parent can start at parent coordinate `(1,1)`, flat offset 4, with strides `[3,1]`. View coordinate `(1,1)` maps to `4+1×3+1=8`, the parent's bottom-right element.

## 1.2.3 Slicing changes offset, shape, and stride

A one-dimensional slice `[start:stop:step]` begins at `start`, contains the ceiling-divided span, and multiplies the parent stride by `step`. This version requires a positive step so the arithmetic stays explicit.

```mojo
def slice_1d(parent: ViewLayout, start: Int, stop: Int, step: Int) raises -> ViewLayout:
    if step <= 0 or start < 0 or stop < start or stop > parent.shape.dims[0]:
        raise Error("invalid slice")
    var length = (stop - start + step - 1) // step
    return ViewLayout(
        Shape([length]),
        List[Int](parent.strides[0] * step),
        parent.offset + start * parent.strides[0],
    )
```

**Manual worked example.** Slice six values with `start=1`, `stop=6`, `step=2`. Length is `(5+1)//2=3`, offset becomes 1, and stride becomes 2. Logical indices 0, 1, 2 read parent offsets 1, 3, 5: values `[1,3,5]`.

## 1.2.4 Broadcasting uses zero strides

Broadcasting aligns axes from the right. When an input extent is 1 and the result extent is larger, the broadcast view uses stride 0 so every logical position rereads the single physical value.

```mojo
def broadcast_stride(input_extent: Int, output_extent: Int, stride: Int) raises -> Int:
    if input_extent == output_extent:
        return stride
    if input_extent == 1:
        return 0
    raise Error("incompatible broadcast extents")
```

**Manual worked example.** Broadcasting shape `[1,3]` over `[2,3]` changes the row stride to 0 and keeps the column stride 1. Coordinates `(0,2)` and `(1,2)` both map to physical column offset 2, so `[10,20,30]` is reused without copying.

## 1.2.5 Alignment is measured in bytes

SIMD alignment is a byte constraint, not an element count. Padding a row rounds its byte length up to the next alignment boundary, then converts back to elements.

```mojo
def padded_float32_row(elements: Int, alignment: Int) -> Int:
    var bytes = elements * 4
    var padded_bytes = ((bytes + alignment - 1) // alignment) * alignment
    return padded_bytes // 4
```

**Manual worked example.** A row of 10 `Float32` values is 40 bytes. With 32-byte alignment, round up to 64 bytes, which is 16 floats. The physical row stride is 16 while the logical row length remains 10, so six padding elements are never exposed as data.

Layouts now explain transpose, slicing, broadcasting, and padding with the same offset equation. Later kernels may specialize these cases, but they must preserve this reference behavior.
