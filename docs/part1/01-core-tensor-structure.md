# 1.1 Core Tensor Structure

A tensor is not merely a pointer. It is an owner or view of numerical storage plus enough metadata to map logical coordinates to physical elements. This first implementation uses `List[Float32]` so ownership is safe and visible; raw pointers and device buffers appear only after the invariants are established.

## 1.1.1 Shape owns dimensional meaning

Shape validation rejects negative extents and computes element count with one rule. A zero extent is valid and produces an empty tensor; it is different from an invalid negative extent.

```mojo
@fieldwise_init
struct Shape(Copyable, Movable):
    var dims: List[Int]

    def numel(self) raises -> Int:
        var result = 1
        for extent in self.dims:
            if extent < 0:
                raise Error("tensor extents must be nonnegative")
            result *= extent
        return result
```

**Manual worked example.** Shape `[2,3,4]` has `2×3×4=24` elements. Shape `[2,0,4]` has 0 elements. Shape `[2,-1,4]` raises before allocation, preventing a negative extent from becoming a huge unsigned byte count.

## 1.1.2 Row-major strides map coordinates to offsets

For contiguous row-major storage, the last axis has stride 1 and each earlier stride is the product of all extents to its right.

```mojo
def row_major_strides(shape: Shape) -> List[Int]:
    var strides = List[Int](length=len(shape.dims), fill=0)
    var running = 1
    var axis = len(shape.dims)
    while axis > 0:
        axis -= 1
        strides[axis] = running
        running *= shape.dims[axis]
    return strides
```

**Manual worked example.** For `[2,3,4]`, work right to left: stride 2 is 1, stride 1 is `1×4=4`, and stride 0 is `4×3=12`. Coordinate `[1,2,3]` maps to `1×12+2×4+3×1=23`, the last of 24 elements.

## 1.1.3 Tensor storage has one clear owner

The foundational tensor stores owned data, shape, and strides. Autograd history is intentionally absent: Chapter 6's tape owns graph identity and gradients so copying a tensor value cannot split its history.

```mojo
struct Tensor:
    var data: List[Float32]
    var shape: Shape
    var strides: List[Int]

    def __init__(out self, var shape: Shape) raises:
        var count = shape.numel()
        self.data = List[Float32](length=count, fill=0)
        self.strides = row_major_strides(shape)
        self.shape = shape^
```

**Manual worked example.** Constructing shape `[2,3]` allocates six zeros and strides `[3,1]`. The state is consistent because data length 6 equals `numel=6`; a constructor that allocated only five elements would violate the invariant before any operation ran.

## 1.1.4 Checked indexing centralizes bounds logic

Every safe access passes through one coordinate-to-offset function. Optimized kernels may use an unchecked path only after a caller proves the same bounds.

```mojo
def offset_of(tensor: Tensor, indices: List[Int]) raises -> Int:
    if len(indices) != len(tensor.shape.dims):
        raise Error("index rank does not match tensor rank")
    var offset = 0
    for axis in range(len(indices)):
        if indices[axis] < 0 or indices[axis] >= tensor.shape.dims[axis]:
            raise Error("tensor index out of bounds")
        offset += indices[axis] * tensor.strides[axis]
    return offset
```

**Manual worked example.** For shape `[2,3]` and strides `[3,1]`, `[1,2]` maps to `1×3+2=5`. `[2,0]` fails because axis 0 permits only 0 and 1; `[1]` fails because its rank is 1 rather than 2.

## 1.1.5 Mutation uses the checked offset

Get and set share the same mapping, preventing subtly different indexing rules in read and write paths.

```mojo
def set(mut tensor: Tensor, indices: List[Int], value: Float32) raises:
    tensor.data[offset_of(tensor, indices)] = value

def get(tensor: Tensor, indices: List[Int]) raises -> Float32:
    return tensor.data[offset_of(tensor, indices)]
```

**Manual worked example.** Set `[1,2]` to 7 in a zero 2×3 tensor. Offset 5 changes the flat buffer from `[0,0,0,0,0,0]` to `[0,0,0,0,0,7]`; reading `[1,2]` recomputes offset 5 and returns 7.

This small representation is deliberately boring: one owner, validated shape, deterministic strides, and one indexing rule. Those invariants are the stable base for views, broadcasting, devices, and autograd.
