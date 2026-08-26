# Chapter 3: Element-wise Operations

Element-wise operations preserve a tensor's logical shape and apply one scalar rule at each coordinate. A simple scalar loop is the correctness reference; Chapter 9 changes the execution mapping, not the answers.

## 3.1 Addition and subtraction

Equal-shape addition checks element count once and visits each flat contiguous offset.

```mojo
def add(a: Tensor, b: Tensor) raises -> Tensor:
    if a.shape.dims != b.shape.dims:
        raise Error("add requires equal shapes")
    var out = Tensor(a.shape)
    for i in range(len(out.data)):
        out.data[i] = a.data[i] + b.data[i]
    return out
```

**Manual worked example.** `[1,2,3]+[10,20,30]=[11,22,33]`. The loop performs exactly those three independent additions; unequal shapes fail before any partial output is written.

## 3.2 Multiplication and safe division

Multiplication is direct. Division needs an explicit policy for zero denominators; this framework raises rather than silently producing infinities.

```mojo
def divide(a: Tensor, b: Tensor) raises -> Tensor:
    if a.shape.dims != b.shape.dims:
        raise Error("divide requires equal shapes")
    var out = Tensor(a.shape)
    for i in range(len(out.data)):
        if b.data[i] == 0:
            raise Error("division by zero")
        out.data[i] = a.data[i] / b.data[i]
    return out
```

**Manual worked example.** `[8,9,10]/[2,3,5]=[4,3,2]`. Replacing the last denominator by zero raises at index 2 rather than returning a partly trustworthy tensor.

## 3.3 Exponential and power rules

Unary functions read one input and write one output at the same coordinate. Their backward rules later reuse either the input or output value.

```mojo
def exp_tensor(input: Tensor) raises -> Tensor:
    var out = Tensor(input.shape)
    for i in range(len(out.data)):
        out.data[i] = exp(input.data[i])
    return out
```

**Manual worked example.** For `[0,1,2]`, outputs are `[1,e,e²]≈[1,2.71828,7.38906]`. Since `d(eˣ)/dx=eˣ`, these saved outputs are also the local derivatives used by Chapter 7.

## 3.4 Broadcasting reads repeated values through layout

Broadcasting does not expand storage. A zero stride makes multiple output coordinates map to one input value; the element-wise loop uses each operand's view layout to calculate offsets.

```mojo
def broadcast_add_2d(a: Tensor, row: Tensor) raises -> Tensor:
    if len(a.shape.dims) != 2 or row.shape.dims != [a.shape.dims[1]]:
        raise Error("expected matrix plus matching row")
    var out = Tensor(a.shape)
    for i in range(a.shape.dims[0]):
        for j in range(a.shape.dims[1]):
            out.data[i * a.shape.dims[1] + j] = a.data[i * a.shape.dims[1] + j] + row.data[j]
    return out
```

**Manual worked example.** Add row `[10,20,30]` to `[[1,2,3],[4,5,6]]`. Row 0 becomes `[11,22,33]`; row 1 becomes `[14,25,36]`. The three row values are read twice but stored once.

These scalar references define both forward behavior and the local derivatives introduced in Chapter 7.
