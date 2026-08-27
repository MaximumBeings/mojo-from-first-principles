# Chapter 17: Gradient Computation Engine — Running Every Backward Rule to Completion

> "Every rule Chapter 16 derived by hand — `AddOp` passes a gradient through, `MulOp` scales it by the other operand, `MatMulOp` transposes and re-multiplies — is inert until something actually walks the graph and calls each one in the right order, on the right numbers, adding up whatever needs adding. This chapter is that something."

**What you will understand by the end of this chapter:**

- The full backward pass for `w = x*y + x`, traced start to finish as one table — and a precise, confirmed gap between `GraphNode.grad` (Chapter 15.2) and the tensor-level `.grad` fields `accumulate_gradient` actually writes to, plus the exact field, already built in Chapter 15.3, that closes it
- Why `accumulate_gradient` branches on `is_none()` rather than always adding — and the definitive answer, finally, to the tensor-aliasing question Chapter 16 raised about `AddOp.backward` returning one Tensor value to two different callers
- A gap this chapter's own `accumulate_gradient` doesn't handle at all: what has to happen when an operand was broadcast during forward (Chapter 12.4), traced with the exact numbers that section already established
- Why this framework's computation graph is single-use — discarded after every backward pass and rebuilt from scratch on the next forward call — and how Chapter 11.2's arena allocator makes that a genuinely free design choice rather than a wasteful one
- Which saved inputs a node can safely drop the instant forward finishes, and which it must keep alive until backward actually visits it — a distinction that turns into real, countable memory on anything larger than a two-node example

**What you need to know first:**

- Chapter 16 (every backward rule this chapter's traversal actually calls: `AddOp`, `MulOp`, `ExpOp`, `MatMulOp` — and the open aliasing question this chapter resolves)
- Chapter 15.2 and 15.3 (`GraphNode`'s `grad` field and `output.grad_fn_index` — both turn out to be exactly the machinery this chapter's central finding depends on)
- Chapter 11.2 (the arena allocator's `O(1)` reset — the cost model behind "just discard the graph and rebuild it")

## The full backward pass, worked by hand, start to finish

Everything is now in place to run the running example — `w = x*y + x`, `x=3, y=4, z=12, w=15` — backward completely, one node at a time, using nothing but the rules Chapter 16 already derived. Walk it as a table, exactly the steps Section 17.1's code automates:

| Step | Node visited | Incoming gradient | Local rule (Chapter 16) | Result | Running totals |
|---|---|---|---|---|---|
| 0 | — (seed) | — | `∂w/∂w = 1` by definition | `w.grad = 1.0` | `w: 1.0` |
| 1 | `add(z, x) → w` | `w.grad = 1.0` | `AddOp.backward` passes the gradient through unchanged to both inputs | `z` gets `1.0`; `x` gets `1.0` | `z: 1.0`, `x: 1.0` |
| 2 | `mul(x, y) → z` | `z.grad = 1.0` | `MulOp.backward`: `x` gets `grad_z × y`, `y` gets `grad_z × x` | `x` gets `1.0 × 4 = 4.0`; `y` gets `1.0 × 3 = 3.0` | `x: 1.0 + 4.0 = 5.0`, `y: 3.0` |

Final answer: **`x.grad = 5.0`, `y.grad = 3.0`** — matching the calculus in Chapter 15 (`∂w/∂x = y+1 = 5`, `∂w/∂y = x = 3`) exactly, and arrived at without a single symbolic derivative, only local multiplications and one addition, applied mechanically in the reverse order Chapter 15.4 established. The one place a "sum" happened rather than a plain pass-through is `x` in Step 2, because `x` was used twice in the forward pass — once directly in the addition, once inside the multiply — precisely Section 16.1's "sum over paths" chain rule, now happening inside a traversal instead of on paper.

## 17.1 Reverse-mode AD Implementation `[FOUNDATIONAL]`

### Intuition

The worked table above is a hand simulation of one function, conventionally called `backward()`, that turns a scalar loss into gradients on every parameter that fed into it. Writing it correctly means getting three things right at once: seeding the very first gradient, visiting nodes in an order where every dependency is already resolved, and skipping nodes that genuinely have nothing to contribute.

<a id="81-reverse-mode-ad-implementation"></a>
### Background

```mojo
fn backward(mut graph: ComputationGraph, loss: Tensor):
    """Reverse-mode automatic differentiation from a scalar loss."""
    debug_assert(loss.shape.size() == 1, "backward() requires a scalar")

    # Seed: dL/dL = 1 -- Step 0 of the worked table above
    loss.grad = Tensor.ones(loss.shape)

    var order = topological_backward_order(graph)   # Chapter 15.4: [1, 0] for the running example
    for node_idx in order:
        var node = graph.nodes[node_idx]
        if node.grad.is_zero():
            continue    # this output was never used downstream -- nothing flows back

        var input_grads = chain_rule_step(
            node.op_name, node.grad, node.inputs, node.output
        )

        for i in range(len(node.inputs)):
            accumulate_gradient(node.inputs[i], input_grads[i])   # Section 17.2
```

### Worked Example 17.1.1 — Tracing the loop literally

`order = [1, 0]`. Iteration 1 visits `graph.nodes[1]` (the `add` node), calls `chain_rule_step("add", 1.0, [z, x], w)`, gets back `[1.0, 1.0]`, and accumulates `1.0` into both `z.grad` and `x.grad` — Step 1 of the table. Iteration 2 visits `graph.nodes[0]` (the `mul` node), calls `chain_rule_step("mul", z.grad=1.0, [x, y], z)`, gets back `[4.0, 3.0]`, and accumulates them into `x.grad` and `y.grad` — Step 2.

```
[COMMON TRAP]  node.grad is read every iteration, but nothing here ever writes it

Look at what this loop actually touches. It reads `node.grad` (line 4
of the loop body, feeding `chain_rule_step`) and it calls
`accumulate_gradient(node.inputs[i], input_grads[i])` -- writing into
each INPUT tensor's own `.grad` field. `loss.grad = Tensor.ones(...)`
sets `loss`'s own `.grad` field too. Not one of these three writes
ever touches `graph.nodes[node_idx].grad` -- the separate field
GraphNode itself carries, zero-initialized in Chapter 15.2's
`__init__` and never assigned anywhere in this function.

Trace what that means literally for the running example. Before the
loop starts, `graph.nodes[1].grad` (the add node's OWN grad field) is
still `0.0` from construction -- `loss.grad = Tensor.ones(...)` set
`w`'s grad field, a completely different piece of storage from
`graph.nodes[1].grad`, since Tensor and GraphNode are separate structs
each with their own `grad: Tensor` field. The very first thing the
loop does is check `node.grad.is_zero()` for `graph.nodes[1]` -- which
is `True` -- and `continue`s, skipping the add node's backward
entirely. `chain_rule_step` for "add" never runs. `accumulate_gradient`
never gets called for `z` or `x` from this step. Read completely
literally, this function skips every node on its very first
iteration and returns with `x.grad` and `y.grad` never set at all --
not the `5.0` and `3.0` the worked table above claims.

The fix is a piece of machinery this book already built, two chapters
ago, and simply hasn't wired up yet: Chapter 15.3's
`output.grad_fn_index = len(self.nodes)`, which exists specifically so
"the output remembers who made it." Seeding needs to set
`graph.nodes[loss.grad_fn_index].grad = loss.grad`, not just
`loss.grad` itself; and `accumulate_gradient` needs to mirror every
update into `graph.nodes[tensor.grad_fn_index].grad` as well as
`tensor.grad`, whenever `tensor` is itself the output of some earlier
node. With that mirroring in place, `graph.nodes[1].grad` genuinely
does hold `1.0` by the time the loop reaches it, and the trace in
Worked Example 17.1.1 runs exactly as described. Without it, `grad_fn_index`
is a field this book built and never actually used.
```

## 17.2 Gradient Accumulation Strategies `[FOUNDATIONAL]`

### Intuition

Step 2 of the worked table is where `x.grad` became `1.0 + 4.0 = 5.0` rather than being overwritten to `4.0`. That single addition is the entire content of this section, and it is one of the two or three most common autograd bugs in every framework that implements it — get it wrong, and any input used more than once (a shared weight, a residual connection `y = f(x) + x`) silently receives only its *last* gradient contribution instead of the sum of all of them.

<a id="82-gradient-accumulation-strategies"></a>
### Background

```mojo
fn accumulate_gradient(mut tensor: Tensor, incoming_grad: Tensor):
    if tensor.grad.is_none():
        tensor.grad = incoming_grad
    else:
        tensor.grad = elementwise_add(tensor.grad, incoming_grad)   # Chapter 12 -- accumulate, don't replace

fn zero_grad(mut params: List[Tensor]):
    for p in params:
        p.grad = Tensor.zeros(p.shape)
```

### Worked Example 17.2.1 — The two branches, walked against the running example

At Step 1, `x.grad` is `.is_none()`, so it's simply set to `1.0` — the `if` branch, a plain assignment. At Step 2, `x.grad` already holds `1.0`, so the new contribution `4.0` is *added* via the `else` branch: `elementwise_add(1.0, 4.0) = 5.0`. Replacing instead of adding at Step 2 would have silently produced `x.grad = 4.0` — plausible-looking, still a number, and wrong. This is exactly why the residual-connection case in a real network is dangerous: `x` feeding both the shortcut and the transformed branch is structurally identical to `x` feeding both `AddOp` and `MulOp` above.

### Worked Example 17.2.2 — Resolving Chapter 16's open aliasing question

Chapter 16.2 flagged something left unresolved: `AddOp.backward` returns `List[Tensor](grad_output, grad_output)` — the *same* Tensor struct value, aliasing the same underlying buffer, handed to both `z`'s incoming gradient and `x`'s first contribution. Now trace what actually happens to that aliasing, step by step. At Step 1, `accumulate_gradient(z, grad_output)` and `accumulate_gradient(x, grad_output)` both hit the `is_none()` branch — both `z.grad` and `x.grad` are assigned directly from the same aliased value, so immediately after Step 1, `z.grad` and `x.grad` genuinely do point at identical memory. At Step 2, `x` receives its second contribution, `4.0`, and `accumulate_gradient` now takes the *other* branch: `x.grad = elementwise_add(x.grad, 4.0)`. `elementwise_add` builds a brand-new output buffer — it does not mutate either input in place, the same non-mutating contract every kernel since Chapter 12 has followed — and `x.grad` is reassigned to point at that new buffer. `z.grad` is untouched by this reassignment; it still points at the original buffer from Step 1. The aliasing that existed between `z.grad` and `x.grad` for one brief moment was real, but it was always harmless, because accumulation is implemented as "compute a new buffer and reassign," never as "mutate whatever buffer is already there" — exactly the same first-assign-then-add-a-fresh-buffer discipline that the same problem is solved with in production autograd engines generally.

### Worked Example 17.2.3 — Verifying `x.grad = 5.0` without any calculus at all

The whole point of a gradient is that it predicts how much the output moves for a tiny nudge to the input — so test that prediction directly, by nudging `x` by `±0.001` and reading `w` both times, the same finite-difference check the neural-network-layers chapter's `gradient_check` automates:

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

```
[COMMON TRAP]  accumulate_gradient assumes every operand already has the output's shape

Neither `AddOp.backward` (Chapter 16.2) nor `accumulate_gradient`
above ever compares shapes. That's fine for the running example --
every tensor involved is a scalar -- but Chapter 12.4 already built a
kernel, `broadcast_add_kernel`, specifically for the case where one
operand is smaller and gets silently repeated. Reuse that section's
own numbers: A is 2x3, B is a single row of 3 values, and
`broadcast_add_kernel` produces C = [[11,22,33],[14,25,36]], a 2x3
result, from A (2x3) and B (1x3).

Suppose this addition is graph-tracked and the upstream gradient
arriving at C is grad_C = [[1,1,1],[1,1,1]] (a 2x3 matrix of ones, the
same all-ones convention Chapter 16.3 used for grad_output). `A`
already has the full 2x3 output shape, so `grad_A = grad_C` unchanged
is correct. But `B`'s ORIGINAL shape was 1x3, not 2x3 -- it was
repeated down both rows by the broadcast, not actually duplicated in
memory. Handing `B` the full 2x3 `grad_C` and calling
`accumulate_gradient(B, grad_C)` would either fail outright (if
`elementwise_add` requires matching shapes) or -- if it doesn't check
at all -- silently store a 2x3 gradient on a tensor whose own data is
1x3, a shape mismatch that corrupts every later use of `B.grad`.

What actually has to happen is a reduction: every row of `grad_C` that
came from the SAME repeated row of `B` needs to be summed back into
one row before it can be `B`'s gradient. Column by column: `grad_B[0,
0] = grad_C[0,0] + grad_C[1,0] = 1 + 1 = 2`; the same sum applies to
columns 1 and 2, giving `grad_B = [2, 2, 2]` -- a 1x3 result, matching
B's real, pre-broadcast shape. Neither `AddOp.backward` nor
`accumulate_gradient` as shown performs this reduction anywhere; a
broadcasting-aware version would need an explicit step -- illustrated
below, not book source code -- that sums a gradient back down to an
operand's original shape before `accumulate_gradient` ever sees it:

    fn unbroadcast_gradient(grad: Tensor, target_shape: TensorShape) -> Tensor:
        """Sum a gradient back down to an operand's pre-broadcast
        shape -- the reverse of Chapter 7.4's BroadcastSpec, which
        decided which axes were allowed to repeat in the first place."""
        var result = grad
        while result.shape.ndim() > target_shape.ndim():
            result = reduce_sum_axis(result, axis=0)
        for axis in range(target_shape.ndim()):
            if target_shape[axis] == 1 and result.shape[axis] != 1:
                result = reduce_sum_axis(result, axis=axis, keepdims=True)
        return result

Every axis the forward pass was allowed to repeat a value across,
backward has to sum back into one slot -- the exact mirror image of
what made the forward broadcast cheap in the first place.
```

## 17.3 Graph Traversal and Execution `[FOUNDATIONAL]`

### Intuition

Section 17.1's `backward()` already showed the traversal; what's worth calling out separately is what happens to the graph *afterward*.

### Background

In this framework, as in eager-mode PyTorch, the graph is single-use: `graph.nodes` for the running example held exactly two entries, was consumed once by the loop in Section 17.1, and would be discarded before the next forward pass builds a fresh one from scratch. This is a deliberate simplicity trade-off — a persistent, reusable graph enables graph-level optimization passes, but a rebuild-every-step graph is dramatically easier to reason about, and combined with Chapter 11.2's arena allocator, just as cheap in practice: `arena.reset()` at the top of the next `forward()` call reclaims every node from the discarded graph in `O(1)`, no matter how many nodes it held.

### Worked Example 17.3.1 — What "discard and rebuild" actually costs

For the running example's two-node graph, discarding and rebuilding costs exactly the same as discarding and rebuilding a two-thousand-node graph, precisely because `arena.reset()`'s cost doesn't depend on how many allocations it's reclaiming — it's the same constant-time bump-pointer reset Chapter 11.2 traced on much larger examples. The alternative — carefully freeing each `GraphNode` and each tensor it references individually — would cost time proportional to the graph's size every single training step; the arena trades a small amount of memory (the arena's peak size has to accommodate the largest graph any single forward pass ever builds) for making that per-step cost disappear entirely.

## 17.4 Memory Optimization for Gradients `[FOUNDATIONAL]`

### Intuition

Two optimizations matter once this engine runs on real workloads rather than a two-node example.

### Background

First, **gradient-only-where-needed**: a node whose entire input subtree has `requires_grad = False` was already excluded from the graph in Chapter 15.3, so no gradient memory is ever allocated for it. Second, **saved-tensor pruning** — and the running example already demonstrates exactly which inputs are safe to drop. `AddOp.backward` (Chapter 16.2) never reads `inputs` at all — its two return values are just copies of `grad_output`. That means a node recording an addition can drop its references to both inputs immediately after forward runs, relying purely on the `grad_output` handed to it later, while `MulOp` and `MatMulOp` cannot — their backward rules read the *other* input, so both must stay alive until backward visits that node.

```mojo
fn needs_input_for_backward(op_name: String) -> Bool:
    # AddOp: grad(a) = grad(b) = grad_output, no input needed.
    # MulOp, MatMulOp: backward reads the *other* input (Chapter 16.2, 16.3).
    return op_name != "add"
```

### Worked Example 17.4.1 — Putting a number on the saving

A `[500, 8]` `Float32` activation tensor is `500 × 8 × 4 bytes = 16,000 bytes`. An `AddOp` node in the middle of the training loop from the neural-network-layers chapter that drops both of its saved inputs the instant forward passes it frees `32,000` bytes it would otherwise have to keep alive for the entire duration of the backward pass. Multiply that by every `add_bias` call in a network with several layers, and by thousands of training steps, and "don't pay memory for a value nothing will read again" stops being a micro-optimization and starts being the difference between a model that fits in GPU memory and one that doesn't.

## Chapter Summary

`backward()` seeds the final output's gradient to `1.0`, walks `topological_backward_order`'s reversed node list, and calls each node's registered backward rule — but this chapter's closest reading found that `GraphNode.grad`, the field that loop actually reads, is never written to anywhere shown: `accumulate_gradient` updates a *tensor's* `.grad` field, not a `GraphNode`'s, and the missing link is `output.grad_fn_index`, built back in Chapter 15.3 specifically so an output could find its way back to the node that produced it, and never actually used until this chapter needed it. `accumulate_gradient`'s `is_none()` branch versus its `elementwise_add` branch resolved Chapter 16's open question definitively: `AddOp.backward` handing out one aliased Tensor to two callers is genuinely safe, because the *second* time either one needs an additional contribution, accumulation allocates a fresh buffer and reassigns rather than mutating shared memory in place. A gap this chapter's own code doesn't cover at all is broadcasting: an operand smaller than the output, as Chapter 12.4 already demonstrated forward, needs its incoming gradient summed back down to its original shape before accumulation, not handed the full output-shaped gradient directly. Finally, this framework's graph is single-use by design — discarded and rebuilt every forward pass, a decision that Chapter 11.2's arena allocator makes essentially free — and a node's `backward` rule determines whether its saved inputs can be dropped the instant forward finishes (`AddOp`) or must survive until backward actually visits that node (`MulOp`, `MatMulOp`), a distinction worth tens of thousands of bytes per node on real tensors.

## Self-Check Questions

1. Using the `grad_fn_index` fix this chapter derived, trace `backward()` for `w = x*y + x` with `x=5.0, y=2.0` (the same numbers used in Chapter 16's Self-Check Question 1) — what value does `graph.nodes[1].grad` need to hold by the time the loop reaches it, and where does that value come from?
2. `accumulate_gradient` is called three times in sequence for the same tensor `p`, with incoming gradients `2.0`, `3.0`, and `5.0`, in that order. Trace each call: which branch fires each time, and what is `p.grad` after each one?
3. A residual block computes `y = f(x) + x` for some function `f`. Using the same reasoning as Worked Example 17.2.1, explain concretely what would go wrong for `x.grad` if `accumulate_gradient`'s `else` branch used `tensor.grad = incoming_grad` (an overwrite) instead of `elementwise_add`.
4. Extend Worked Example 17.2.2's aliasing trace: if a THIRD node also used `z` (in addition to the `add` node), and that third node's backward also produced a contribution to `z.grad` via the `elementwise_add` branch, would that operation risk mutating `x.grad`'s buffer from Step 1? Why or why not?
5. Using `unbroadcast_gradient`'s two-part algorithm (reduce leading dimensions, then reduce dimensions where the target shape is `1`), trace what it computes for a gradient of shape `[4, 3, 5]` being reduced down to a target shape of `[3, 1]`.

## Where We Go Next

Parts 1 through 4 now form a complete, working autograd engine, and the running example proves it end to end: a graph was built (Chapter 15), each node's local derivative was derived by hand and matched against code (Chapter 16), and a full reverse pass produced `x.grad=5.0, y.grad=3.0` — verified twice over, once against ordinary calculus and once against finite differences (this chapter), with the one remaining gap in the traversal itself (`GraphNode.grad` never being written) traced to its precise, fixable cause. Part 5 makes all of it fast by moving the hot paths onto the GPU.

## Worked Solutions

**1.** `graph.nodes[1]` is the `add` node, whose `output` is `w`. With the `grad_fn_index` fix in place, seeding `loss.grad = Tensor.ones(loss.shape)` also writes `graph.nodes[loss.grad_fn_index].grad = loss.grad` — and since `loss` *is* `w`, and `w.grad_fn_index` was set to `1` when the `add` node was recorded (Chapter 15.3), that value lands in `graph.nodes[1].grad`, giving it `1.0` before the loop ever reaches it. Without that mirroring step, as this chapter's `[COMMON TRAP]` showed, it would still be `0.0`.

**2.** Call 1: `p.grad` is `.is_none()` (assuming it starts unset) → the `if` branch fires → `p.grad = 2.0`. Call 2: `p.grad` is no longer none → the `else` branch fires → `p.grad = elementwise_add(2.0, 3.0) = 5.0`. Call 3: `else` branch again → `p.grad = elementwise_add(5.0, 5.0) = 10.0`. Final: `p.grad = 10.0`, the sum of all three contributions, matching the "accumulate, don't replace" contract for every contribution after the first.

**3.** `x` feeds `y = f(x) + x` through two routes, exactly like the running example's `x` feeding both `AddOp` and `MulOp`. If the second contribution to `x.grad` overwrote instead of added, only whichever contribution arrived *last* in the traversal order would survive — either the direct `+x` route's contribution (`1.0`, structurally) or `f`'s contribution, but never their sum. The gradient reported for `x` would be smaller than the true `∂y/∂x`, silently, with no error raised — precisely the "one of the two or three most common autograd bugs" this section opened by naming, and precisely why residual connections are a natural place for it to surface in a real network.

**4.** No genuine risk, for the same reason Worked Example 17.2.2 gave: the third node's contribution to `z.grad` goes through `accumulate_gradient`'s `else` branch (since `z.grad` is no longer none after Step 1), which computes `elementwise_add(z.grad, new_contribution)` and produces a *brand-new* buffer, reassigning `z.grad` to point at it. `x.grad`'s own buffer — separately reassigned back in Step 2 of the original trace — is never touched by this operation on `z`, since `elementwise_add` never mutates either of its input buffers in place, only ever allocates and returns a new one.

**5.** First, reduce leading dimensions: the gradient has `3` dimensions (`[4,3,5]`) and the target has `2` (`[3,1]`), so one leading-dimension sum fires: `result = sum(grad, axis=0)`, producing shape `[3,5]`. Second, check each remaining axis against the target shape: axis `0` of the target is `3`, matching `result`'s axis `0` (`3`) — no reduction needed there. Axis `1` of the target is `1`, but `result`'s axis `1` is `5` — a mismatch, so `result = sum(result, axis=1, keepdims=True)`, producing final shape `[3,1]`, matching the target exactly.
