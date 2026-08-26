# Chapter 8: Gradient Computation Engine

## 8.1 Reverse-mode AD Implementation

This is the function everything in Parts 1 through 4 has been building toward — the one a user actually calls, conventionally named `backward()`, that turns a scalar loss into gradients on every parameter that fed into it:

```mojo
fn backward(mut graph: ComputationGraph, loss: Tensor):
    """Reverse-mode automatic differentiation from a scalar loss."""
    debug_assert(loss.shape.size() == 1, "backward() requires a scalar")

    # Seed: dL/dL = 1
    loss.grad = Tensor.ones(loss.shape)

    var order = topological_backward_order(graph)   # Chapter 6.4
    for node_idx in order:
        var node = graph.nodes[node_idx]
        if node.grad.is_zero():
            continue    # this output was never used downstream -- nothing flows back

        var input_grads = chain_rule_step(
            node.op_name, node.grad, node.inputs, node.output
        )

        for i in range(len(node.inputs)):
            accumulate_gradient(node.inputs[i], input_grads[i])   # Section 8.2
```

Two details make this correct rather than merely plausible. First, the seed gradient: `dL/dL = 1` is not a convention, it's the base case the chain rule needs to bottom out on — every other gradient in the graph is ultimately `1` multiplied by a chain of local derivatives. Second, the `is_zero()` skip: in a graph where one branch feeds the loss and a sibling branch feeds only a diagnostic printout, the diagnostic branch's node never receives a gradient and correctly contributes nothing — this is what makes it safe to compute expensive-but-unused forward values without breaking gradient correctness elsewhere.

## 8.2 Gradient Accumulation Strategies

A tensor used more than once in a forward pass — the classic case is a shared embedding or a residual connection where `y = f(x) + x` — receives *two* incoming gradients during backward, one from each use, and they must be **added**, not overwritten. This is one of the two or three most common autograd bugs in every framework that implements it:

```mojo
fn accumulate_gradient(mut tensor: Tensor, incoming_grad: Tensor):
    if tensor.grad.is_none():
        tensor.grad = incoming_grad
    else:
        tensor.grad = elementwise_add(tensor.grad, incoming_grad)   # Chapter 3 -- accumulate, don't replace
```

The residual case makes the necessity concrete: in `y = f(x) + x`, `x` is an input to both `AddOp` and whatever produced `f(x)`. `AddOp.backward` passes `grad_output` straight through to *both* of its inputs (Section 7.2), so `x` receives one gradient contribution directly from the addition and a second, separately-computed contribution routed back through `f`. `accumulate_gradient` summing them is precisely `dL/dx = dL/dy · (∂y/∂x)_direct + dL/dy · (∂y/∂x)_through_f` — the multivariate chain rule's sum-over-paths rule, implemented as a running total instead of a symbolic sum.

Before starting a new backward pass, every parameter's accumulated gradient from the *previous* step must be reset to zero — otherwise gradients silently accumulate across training steps, a bug that manifests as loss curves that look almost-but-not-quite right:

```mojo
fn zero_grad(mut params: List[Tensor]):
    for p in params:
        p.grad = Tensor.zeros(p.shape)
```

## 8.3 Graph Traversal and Execution

Section 8.1's `backward()` already showed the traversal itself; what's worth calling out separately is what happens to the graph *after* backward finishes. In this framework, as in eager-mode PyTorch, the graph is single-use: it was built implicitly during forward (Chapter 6.3), consumed once during backward, and then discarded — the next forward pass builds a fresh one. This is a deliberate simplicity trade-off: a persistent, reusable graph (the kind a compiled/static-graph framework builds) enables graph-level optimization passes, but a rebuild-every-step graph is dramatically easier to reason about and, combined with the [Arena allocator](../part1/06-memory-management-system.md#22-arena-based-memory-allocation) from Chapter 2, just as cheap in practice: `arena.reset()` at the top of the next `forward()` call reclaims every node from the discarded graph in O(1).

## 8.4 Memory Optimization for Gradients

Two optimizations matter once this engine runs on real workloads rather than toy examples. First, **gradient-only-where-needed**: a node whose entire input subtree has `requires_grad = False` was already excluded from the graph in Chapter 6.3, so no gradient memory is ever allocated for it — the cheapest optimization is the one where nothing happens at all. Second, **saved-tensor pruning**: a node only needs to keep an input alive for backward if that specific op's backward rule reads it. `AddOp.backward` (Section 7.2) never touches `inputs` at all, so a node recording an addition can drop its references to both inputs immediately after forward and rely purely on `grad_output` passed down later — freeing that memory before backward even starts, rather than holding it for the graph's entire lifetime:

```mojo
struct GraphNode:
    var op_name: String
    var saved_inputs: List[Tensor]   # only populated if this op's backward needs them
    var output: Tensor
    var grad: Tensor

fn needs_input_for_backward(op_name: String) -> Bool:
    # AddOp: grad(a) = grad(b) = grad_output, no input needed.
    # MulOp, MatMulOp: backward reads the *other* input (Section 7.2, 7.3).
    return op_name != "add"
```

This is the same shape of trade-off PyTorch calls "checkpointing" taken to its logical per-op conclusion: don't pay memory for a value nothing will read again.

Parts 1 through 4 now form a complete, working autograd engine: tensors with proper memory management, the arithmetic operations on them, a graph that records how they combine, and a backward pass that turns that graph into gradients. Part 5 makes all of it fast by moving the hot paths onto the GPU.
