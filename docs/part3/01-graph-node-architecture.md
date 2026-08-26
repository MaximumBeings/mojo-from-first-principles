# Chapter 6: Graph Node Architecture

Parts 1 and 2 built tensors and the operations on them, but every operation so far has been a dead end — call `elementwise_mul_kernel`, get a result, and the framework immediately forgets how that result was produced. Reverse-mode automatic differentiation needs the opposite: it needs to replay every operation *backward*, so each `Tensor` that requires gradients has to grow a record of what created it. That record is the computational graph, and this chapter is where the tensor library from Parts 1–2 becomes an autograd framework.

## 6.1 Function Registration System

Every differentiable operation registers a forward implementation and the corresponding backward (gradient) implementation together, as a pair, so the graph can never end up with a forward op it doesn't know how to differentiate:

```mojo
trait Differentiable:
    """Every op the graph can record implements both directions."""
    fn forward(self, inputs: List[Tensor]) -> Tensor: ...
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]: ...

struct OpRegistry:
    """Maps an op name to its registered Differentiable implementation."""
    var _ops: Dict[String, Differentiable]

    fn register(mut self, name: String, op: Differentiable):
        self._ops[name] = op

    fn get(self, name: String) -> Differentiable:
        debug_assert(name in self._ops, "Unregistered op: " + name)
        return self._ops[name]
```

`elementwise_mul_kernel` from [Chapter 3](../part2/01-element-wise-operations.md#32-multiplication-and-division) is the `forward` half of a `MulOp`; its `backward` returns `[grad_output * b, grad_output * a]` — the two local-derivative rules derived in that chapter, applied to the tensor now flowing backward through the graph instead of a hand-computed scalar.

## 6.2 Gradient Function Traits

Each node in the graph stores exactly enough to run its op's `backward` later: the op itself, references to its input tensors (kept alive by the `RefCountedBuffer` sharing from [Chapter 2](../part1/06-memory-management-system.md#21-reference-counting-implementation)), and the output it produced.

```mojo
struct GraphNode:
    var op_name: String
    var inputs: List[Tensor]        # shares buffers, doesn't copy
    var output: Tensor
    var grad: Tensor                # accumulated upstream gradient, filled during backward
    var requires_grad: Bool

    fn __init__(out self, op_name: String, inputs: List[Tensor], output: Tensor):
        self.op_name = op_name
        self.inputs = inputs
        self.output = output
        self.requires_grad = True
        self.grad = Tensor.zeros(output.shape)
```

The one field worth pausing on is `requires_grad`. A tensor loaded from a data file (Section 1.3.3) or a fixed hyperparameter never needs a graph node at all — building one would waste memory and, worse, keep the whole upstream subgraph alive for no reason. The framework checks `requires_grad` before recording, so constant inputs are invisible to the graph exactly the way `torch.no_grad()` or JAX's `stop_gradient` behave.

## 6.3 Graph Construction During Forward Pass

The graph is built implicitly, one node per operation, as the forward pass executes — there's no separate "trace" step:

```mojo
struct ComputationGraph:
    var nodes: List[GraphNode]

    fn record(mut self, op_name: String, inputs: List[Tensor], output: Tensor) -> Tensor:
        var needs_grad = False
        for t in inputs:
            if t.requires_grad:
                needs_grad = True
        if not needs_grad:
            return output          # constant-folded: no node created
        var node = GraphNode(op_name, inputs, output)
        output.requires_grad = True
        output.grad_fn_index = len(self.nodes)   # output remembers who made it
        self.nodes.append(node)
        return output

fn mul(mut graph: ComputationGraph, a: Tensor, b: Tensor) -> Tensor:
    var result = elementwise_mul(a, b)              # Chapter 3's kernel
    return graph.record("mul", List[Tensor](a, b), result)
```

Writing `c = mul(graph, a, b)` therefore does two things in one call: it computes `c` numerically (exactly as Chapter 3 defined), and it appends a `GraphNode` recording "`c` came from multiplying `a` and `b`" — which is all the information backward needs, applied one node at a time.

## 6.4 Topological Sorting Implementation

Backward must visit nodes in the reverse of the order their outputs were needed as inputs elsewhere — a node can't run backward until every node that consumed its output already has. Because `nodes` is appended to in forward-execution order, and every node's inputs were necessarily computed (and therefore appended) before it, the list is *already* topologically sorted forward-to-back. Running the list in reverse is a valid reverse-topological order with no separate sort pass required:

```mojo
fn topological_backward_order(graph: ComputationGraph) -> List[Int]:
    """Forward execution order is already a valid topo-sort;
    backward just walks it in reverse."""
    var order = List[Int]()
    var i = len(graph.nodes) - 1
    while i >= 0:
        order.append(i)
        i -= 1
    return order
```

This is the one place where recording the graph *as* the forward pass runs (rather than building an abstract graph up front, the way a static-graph framework does) pays for itself directly: order is free, because Mojo's own sequential execution already produced it.

With graph construction and traversal order settled, Part 4 fills in what actually happens at each node during that reverse walk: the chain rule, gradient accumulation, and the reverse-mode AD engine itself.
