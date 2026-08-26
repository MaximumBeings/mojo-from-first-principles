# Chapter 8: Gradient Computation Engine

## The full backward pass, worked by hand, start to finish

Everything is now in place to run the running example — `w = x*y + x`, `x=3, y=4, z=12, w=15` — backward completely, one node at a time, using nothing but the rules Chapter 7 already derived. Walk it as a table, exactly the steps the code in Section 8.1 automates:

| Step | Node visited | Incoming gradient | Local rule (Chapter 7) | Result | Running totals |
|---|---|---|---|---|---|
| 0 | — (seed) | — | `∂w/∂w = 1` by definition | `w.grad = 1.0` | `w: 1.0` |
| 1 | `add(z, x) → w` | `w.grad = 1.0` | `AddOp.backward` passes the gradient through unchanged to both inputs | `z` gets `1.0`; `x` gets `1.0` | `z: 1.0`, `x: 1.0` |
| 2 | `mul(x, y) → z` | `z.grad = 1.0` | `MulOp.backward`: `x` gets `grad_z × y`, `y` gets `grad_z × x` | `x` gets `1.0 × 4 = 4.0`; `y` gets `1.0 × 3 = 3.0` | `x: 1.0 + 4.0 = 5.0`, `y: 3.0` |

Final answer: **`x.grad = 5.0`, `y.grad = 3.0`** — matching the calculus in Chapter 6 (`∂w/∂x = y+1 = 5`, `∂w/∂y = x = 3`) exactly, and arrived at without a single symbolic derivative — only local multiplications and one addition, applied mechanically in the reverse order Chapter 6.4 established. The one place a "sum" happened rather than a plain pass-through is `x` in step 2, because `x` was used twice in the forward pass (once directly in the addition, once inside the multiply) — this is precisely Chapter 7.1's "sum over paths" chain rule, now happening inside the traversal instead of on paper.

## 8.1 Reverse-mode AD implementation

This is the function the worked table above is a hand-simulation of — the one a user actually calls, conventionally named `backward()`, that turns a scalar loss into gradients on every parameter that fed into it:

```mojo
fn backward(mut graph: ComputationGraph, loss: Tensor):
    """Reverse-mode automatic differentiation from a scalar loss."""
    debug_assert(loss.shape.size() == 1, "backward() requires a scalar")

    # Seed: dL/dL = 1 -- Step 0 of the worked table above
    loss.grad = Tensor.ones(loss.shape)

    var order = topological_backward_order(graph)   # Chapter 6.4: [1, 0] for the running example
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

Trace the running example through this code literally: `order = [1, 0]`. Iteration 1 visits `graph.nodes[1]` (the `add` node), calls `chain_rule_step("add", 1.0, [z, x], w)`, gets back `[1.0, 1.0]`, and accumulates `1.0` into both `z.grad` and `x.grad` — exactly Step 1 of the table. Iteration 2 visits `graph.nodes[0]` (the `mul` node), calls `chain_rule_step("mul", z.grad=1.0, [x, y], z)`, gets back `[4.0, 3.0]`, and accumulates them into `x.grad` and `y.grad` — exactly Step 2. Two details make this correct rather than merely plausible. First, the seed: `loss.grad = 1` is not a convention, it's the base case the chain rule needs to bottom out on — every other gradient in the graph is ultimately `1` multiplied by a chain of local derivatives, which is why Step 0 of the table reads `∂w/∂w = 1` rather than an arbitrary starting number. Second, the `is_zero()` skip: in a graph where one branch feeds the loss and a sibling branch feeds only a diagnostic printout, the diagnostic branch's node never receives a gradient and correctly contributes nothing.

## 8.2 Gradient accumulation strategies

Step 2 of the worked table is where `x.grad` became `1.0 + 4.0 = 5.0` rather than being overwritten to `4.0`. That single addition is the entire content of this section, and it is one of the two or three most common autograd bugs in every framework that implements it — get it wrong, and any input used more than once (a shared weight, a residual connection `y = f(x) + x`) silently receives only its *last* gradient contribution instead of the sum of all of them.

```mojo
fn accumulate_gradient(mut tensor: Tensor, incoming_grad: Tensor):
    if tensor.grad.is_none():
        tensor.grad = incoming_grad
    else:
        tensor.grad = elementwise_add(tensor.grad, incoming_grad)   # Chapter 3 -- accumulate, don't replace
```

Walked against the running example: at Step 1, `x.grad` is `.is_none()`, so it's simply set to `1.0`. At Step 2, `x.grad` already holds `1.0`, so the new contribution `4.0` is *added*: `elementwise_add(1.0, 4.0) = 5.0`. Replacing instead of adding at Step 2 would have silently produced `x.grad = 4.0` — plausible-looking, still a number, and wrong. This is exactly why the residual-connection case in a real network is dangerous: `x` feeding both the shortcut and the transformed branch is structurally identical to `x` feeding both `AddOp` and `MulOp` above.

Before starting a new backward pass, every parameter's accumulated gradient from the *previous* step must be reset to zero — otherwise gradients silently accumulate across training steps:

```mojo
fn zero_grad(mut params: List[Tensor]):
    for p in params:
        p.grad = Tensor.zeros(p.shape)
```

**Verifying `x.grad = 5.0` without any calculus at all.** The whole point of a gradient is that it predicts how much the output moves for a tiny nudge to the input — so test that prediction directly, by nudging `x` by `±0.001` and reading `w` both times, the same finite-difference check [Chapter 12.4](../part6/02-advanced-features.md#124-debugging-and-profiling-tools) automates as `gradient_check`:

```
w(x=3.001, y=4) = 3.001×4 + 3.001 = 12.004 + 3.001 = 15.005
w(x=2.999, y=4) = 2.999×4 + 2.999 = 11.996 + 2.999 = 14.995

slope ≈ (15.005 - 14.995) / (3.001 - 2.999) = 0.010 / 0.002 = 5.0   ✓ matches x.grad
```

And the same check for `y`:

```
w(x=3, y=4.001) = 3×4.001 + 3 = 12.003 + 3 = 15.003
w(x=3, y=3.999) = 3×3.999 + 3 = 11.997 + 3 = 14.997

slope ≈ (15.003 - 14.997) / (4.001 - 3.999) = 0.006 / 0.002 = 3.0   ✓ matches y.grad
```

Both slopes land exactly on the `backward()`-computed gradients, which is not a coincidence for a linear-in-each-variable expression like this one — it's the same agreement `gradient_check` looks for on every backward rule in the registry before that rule is trusted.

## 8.3 Graph traversal and execution

Section 8.1's `backward()` already showed the traversal; what's worth calling out separately is what happens to the graph *afterward*. In this framework, as in eager-mode PyTorch, the graph is single-use: `graph.nodes` for the running example held exactly two entries, was consumed once by the loop in Section 8.1, and would be discarded before the next forward pass builds a fresh one from scratch. This is a deliberate simplicity trade-off — a persistent, reusable graph enables graph-level optimization passes, but a rebuild-every-step graph is dramatically easier to reason about, and combined with the [Arena allocator](../part1/06-memory-management-system.md#22-arena-based-memory-allocation) from Chapter 2, just as cheap in practice: `arena.reset()` at the top of the next `forward()` call reclaims every node from the discarded graph in O(1), no matter how many nodes it held.

## 8.4 Memory optimization for gradients

Two optimizations matter once this engine runs on real workloads rather than a two-node example. First, **gradient-only-where-needed**: a node whose entire input subtree has `requires_grad = False` was already excluded from the graph in Chapter 6.3, so no gradient memory is ever allocated for it. Second, **saved-tensor pruning** — and the running example already demonstrates exactly which inputs are safe to drop. `AddOp.backward` (Section 7.2) never reads `inputs` at all — its two return values are just copies of `grad_output`. That means a node recording an addition can drop its references to both inputs immediately after forward runs, relying purely on the `grad_output` handed to it later, while `MulOp` and `MatMulOp` cannot — their backward rules read the *other* input, so both must stay alive until backward visits that node.

```mojo
fn needs_input_for_backward(op_name: String) -> Bool:
    # AddOp: grad(a) = grad(b) = grad_output, no input needed.
    # MulOp, MatMulOp: backward reads the *other* input (Chapter 7.2, 7.3).
    return op_name != "add"
```

Put a number on what this saves: a `[500, 8]` `Float32` activation tensor is `500 × 8 × 4 bytes = 16,000 bytes`. An `AddOp` node in the middle of the training loop from [Chapter 11](../part6/01-neural-network-layers.md) that drops both of its saved inputs the instant forward passes it frees `32,000` bytes it would otherwise have to keep alive for the entire duration of the backward pass. Multiply that by every `add_bias` call in a network with several layers, and by thousands of training steps, and "don't pay memory for a value nothing will read again" stops being a micro-optimization and starts being the difference between a model that fits in GPU memory and one that doesn't.

Parts 1 through 4 now form a complete, working autograd engine, and the running example proves it end to end: a graph was built (Chapter 6), each node's local derivative was derived by hand and matched against code (Chapter 7), and a full reverse pass produced `x.grad=5.0, y.grad=3.0` — verified twice over, once against ordinary calculus and once against finite differences (this chapter). Part 5 makes all of it fast by moving the hot paths onto the GPU.
