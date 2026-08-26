# Chapter 6: Graph Node Architecture

## Why a graph, and not just a function call?

Suppose you write the expression `w = x*y + x` in Mojo, with `x = 3.0` and `y = 4.0`. Running it forward is trivial:

```
z = x * y = 3.0 * 4.0 = 12.0
w = z + x = 12.0 + 3.0 = 15.0
```

Now ask a different question: *if `x` nudges up by a tiny amount, how much does `w` move?* You could answer this by hand, with ordinary calculus: `w = xy + x`, so `∂w/∂x = y + 1 = 4 + 1 = 5`. But notice what you needed to answer that question — you needed to remember that `w` was built from `z`, that `z` was built from `x` and `y` by multiplication, and that `x` also feeds into `w` a *second* time, directly, through the addition. The moment Mojo finishes computing `15.0`, none of that history is still around — `w` is just a `Float32`, indistinguishable from a `Float32` that arrived from anywhere else.

This is the entire problem Part 3 exists to solve. **A computational graph is nothing more than that missing history, kept around on purpose.** Every time an operation runs, instead of only producing a value, it also records a small note: "I am a multiply, my inputs were `x` and `y`, and I produced `12.0`." Chapter 7 will show that once every operation leaves a note like this, the chain rule can be applied *mechanically* to the whole chain of notes — no human has to sit down and work out `∂w/∂x = y + 1` symbolically ever again. This chapter builds the note itself (`GraphNode`), the mechanism that writes one during every forward computation, and the ordering rule that says which notes to read first when the time comes to walk backward.

Keep the numbers above in your head — `x=3, y=4, z=12, w=15` — because every section below builds the graph these two operations actually produce, one field at a time.

## 6.1 Function Registration System

An operation can only be part of a graph if the framework knows two things about it: how to run it forward, and how to turn an "upstream sensitivity" into sensitivities for each of its inputs. Neither is optional — an op with only a forward implementation can compute `w`, but can never explain how `w` depends on `x`. The framework enforces this by requiring every operation to implement both halves of one trait, registered together, so a graph can never end up holding a forward computation it doesn't know how to differentiate:

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

For the running example, two ops need to exist in the registry before `w = x*y + x` can even be recorded: a `MulOp` and an `AddOp`. Their `forward` methods are exactly the arithmetic already shown (`z = x*y`, `w = z+x`); their `backward` methods are what Chapter 7 derives in detail, but the *shape* of the answer is worth previewing now, because it explains why `backward` needs the arguments it has. For `MulOp`, knowing only the upstream sensitivity isn't enough — you also need to know what the *other* input was: the sensitivity of `z=x*y` to `x` is literally the value of `y` (`4.0`), and the sensitivity to `y` is the value of `x` (`3.0`). That's why `backward` receives `inputs`, not just `grad_output` — the local derivative of a multiply is the *other operand*, and there is no way to know that without seeing both inputs.

## 6.2 Gradient Function Traits

Once an op is registered, each time it actually *runs* inside a graph-tracked computation, the framework needs somewhere to keep the specific inputs and output from that one call — not the op's code (there's only one `MulOp`), but this particular invocation of it, on these particular tensors. That's a `GraphNode`:

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

Running the example through this struct by hand, the multiply produces exactly one `GraphNode`:

```
GraphNode #0:
  op_name = "mul"
  inputs  = [x = 3.0, y = 4.0]
  output  = z = 12.0
  grad    = 0.0   (not yet known -- filled in during backward, Chapter 8)
```

and the addition produces a second one, whose `inputs` list contains `z` — the *tensor* that node #0 produced, not a copy of the number `12.0`:

```
GraphNode #1:
  op_name = "add"
  inputs  = [z = 12.0, x = 3.0]
  output  = w = 15.0
  grad    = 0.0
```

The field worth pausing on is `requires_grad`. If `y` had instead been a fixed hyperparameter — read from a config file, never meant to be trained — building a `GraphNode` for it would be pure waste: memory spent, and a chain of history kept alive for a gradient nobody will ever ask for. The framework checks `requires_grad` on every input before recording anything, so a constant is invisible to the graph the same way `torch.no_grad()` or JAX's `stop_gradient` make a value invisible in those frameworks.

## 6.3 Graph Construction During Forward Pass

There is no separate "build the graph" step in this framework — the graph is a side effect of running the forward pass once, normally:

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

fn add(mut graph: ComputationGraph, a: Tensor, b: Tensor) -> Tensor:
    var result = elementwise_add(a, b)
    return graph.record("add", List[Tensor](a, b), result)
```

Evaluating `w = add(graph, mul(graph, x, y), x)` with `x=3.0, y=4.0` now does two things simultaneously, exactly the way running the arithmetic by hand did: it produces the number `15.0`, and it leaves `graph.nodes` holding precisely the two `GraphNode`s written out in Section 6.2, in the order they were computed:

```
graph.nodes[0] = GraphNode("mul", inputs=[x,y], output=z=12.0)
graph.nodes[1] = GraphNode("add", inputs=[z,x], output=w=15.0)
```

That list is the entire "history" this chapter opened by saying was normally thrown away. Nothing about it is symbolic or abstract — it is a literal, numeric trace of the computation that just ran.

## 6.4 Topological Sorting Implementation

Backward must undo the computation in the opposite order it was built: you cannot ask "how sensitive is `w` to `z`" until you know `w` exists, and `w`'s node was necessarily appended to `graph.nodes` *after* `z`'s node, because `z` had to be computed first to be used as an input. That single fact — every node's inputs were computed, and therefore appended, strictly before it — means the list is already sorted in a valid forward order, and reversing it is a valid order for backward with no separate sorting algorithm required:

```
Forward order (how graph.nodes was built):   [ mul(x,y)→z,  add(z,x)→w ]
Reverse order (the order backward will use): [ add(z,x)→w,  mul(x,y)→z ]
```

Read that reversed list as a to-do list for Chapter 8: first, figure out how sensitive the final `w` is to `z` and to `x` through the addition; only afterward, once `z`'s sensitivity is known, ask how sensitive `z` is to `x` and `y` through the multiplication. Trying to do it in the other order would mean asking "how does `z` affect `x` and `y`" before you even know how much `z` itself matters to the answer — which is not yet a meaningful question.

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

For the running example, `topological_backward_order` returns `[1, 0]` — node 1 (`add`) first, node 0 (`mul`) second. Chapter 7 derives what each node does with the sensitivity it's handed; Chapter 8 walks this exact list, `[1, 0]`, and produces the numbers `x.grad = 5.0` and `y.grad = 3.0` that opened this chapter as something you'd otherwise have to work out by hand.
