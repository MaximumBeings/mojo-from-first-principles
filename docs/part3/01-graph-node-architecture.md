# Chapter 6: Graph Node Architecture

Automatic differentiation needs two kinds of state: numerical values from the forward pass and a record of how those values were produced. This chapter deliberately starts with scalar `Float32` values. Removing tensor shape and allocation details makes the graph algorithm visible; Chapters 7–8 then extend the same value-ID design to gradients, and Parts 5–6 replace each scalar primitive with a tensor kernel.

!!! note "Mojo 1.0 design choice"
    The tape stores integer value IDs, not `Tensor` values inside graph nodes. This avoids self-referential ownership, makes shared subexpressions explicit, and ensures that every use of a value reaches the same gradient slot. Operations use an enum-like tag rather than a heterogeneous `Dict[String, Differentiable]`, because a bare trait name is not a runtime trait object in Mojo.

## 6.1 A node records dataflow, not values

A node only needs to answer three questions: which operation ran, which earlier values were its inputs, and which value did it create? The tape owns the actual numbers in one list. This separation is the central invariant of the design.

```mojo
alias ValueId = Int

@fieldwise_init
struct Node(Copyable, Movable):
    var op: OpKind
    var left: ValueId
    var right: ValueId
    var output: ValueId

@fieldwise_init
struct OpKind(Copyable, Movable, Equatable):
    var tag: UInt8

    comptime ADD = Self(0)
    comptime MUL = Self(1)
```

**Manual worked example.** Number the leaves `x=3` and `y=4` as values 0 and 1. Multiplication creates value 2, so its node is `(MUL, 0, 1, 2)`. Addition reuses `x` and creates value 3, so its node is `(ADD, 2, 0, 3)`. Both references to `x` contain the same ID, 0; there is exactly one future gradient slot for `x`.

## 6.2 The tape owns values and gradient policy

The tape is an append-only record during the forward pass. Each new value gets parallel entries for its number and whether it requires a gradient. Parallel arrays are intentionally plain here: for value ID `i`, all metadata lives at index `i`, so identity never depends on pointer aliasing or accidental copies.

```mojo
struct Tape:
    var values: List[Float32]
    var requires_grad: List[Bool]
    var nodes: List[Node]

    def __init__(out self):
        self.values = List[Float32]()
        self.requires_grad = List[Bool]()
        self.nodes = List[Node]()

    def leaf(mut self, value: Float32, requires_grad: Bool) -> ValueId:
        var result = len(self.values)
        self.values.append(value)
        self.requires_grad.append(requires_grad)
        return result
```

**Manual worked example.** `x = tape.leaf(3, True)` returns 0 and `y = tape.leaf(4, True)` returns 1. The lists become `values=[3,4]` and `requires_grad=[True,True]`. If `y` were a constant, only the second Boolean would change; the forward number would remain available, but backward would not accumulate into its slot.

## 6.3 Forward operations append both a value and a node

An operation first computes its result, then records the dependency. If neither input needs a gradient, it keeps the result but omits the node. That is dead-history elimination, not constant folding: the arithmetic still ran, but no backward work is retained.

```mojo
def record_binary(
    mut tape: Tape, op: OpKind, left: ValueId, right: ValueId, value: Float32
) -> ValueId:
    var output = len(tape.values)
    var needs_grad = tape.requires_grad[left] or tape.requires_grad[right]
    tape.values.append(value)
    tape.requires_grad.append(needs_grad)
    if needs_grad:
        tape.nodes.append(Node(op, left, right, output))
    return output

def mul(mut tape: Tape, left: ValueId, right: ValueId) -> ValueId:
    return record_binary(
        tape, OpKind.MUL, left, right, tape.values[left] * tape.values[right]
    )

def add(mut tape: Tape, left: ValueId, right: ValueId) -> ValueId:
    return record_binary(
        tape, OpKind.ADD, left, right, tape.values[left] + tape.values[right]
    )
```

**Manual worked example.** With IDs 0 and 1 holding 3 and 4, `mul(tape,0,1)` computes `3×4=12`, appends 12 at ID 2, and records `(MUL,0,1,2)`. Then `add(tape,2,0)` computes `12+3=15`, appends 15 at ID 3, and records `(ADD,2,0,3)`. The final tape is `values=[3,4,12,15]` with two nodes.

## 6.4 Reverse insertion order is a topological order

Every operation can only consume values that already exist. Therefore every producer node appears before every consumer node in this eager, append-only tape. Walking the node list from the last index to zero is a valid reverse topological traversal; no separate sorting algorithm is needed unless the framework later admits out-of-order graph construction.

```mojo
def reverse_node_indices(tape: Tape) -> List[Int]:
    var order = List[Int]()
    var i = len(tape.nodes)
    while i > 0:
        i -= 1
        order.append(i)
    return order
```

**Manual worked example.** Two nodes were appended in forward order `[MUL, ADD]`. The loop starts at `i=2`, decrements before indexing, and appends 1 then 0. Backward therefore visits `[ADD, MUL]`, exactly the order required to send the loss gradient into value 2 before the multiplication node reads it.

The graph now preserves identity and dataflow without relying on raw pointers, copied tensors, undeclared `grad_fn_index` fields, or a second gradient stored on each node. Chapter 7 derives the local rules; Chapter 8 adds one gradient slot per value and executes this reverse order.
