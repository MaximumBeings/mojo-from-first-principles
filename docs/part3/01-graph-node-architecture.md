# Chapter 15: Graph Node Architecture — Recording What the Forward Pass Throws Away

> "The moment Mojo finishes computing `15.0`, none of the history that produced it is still around — `w` is just a `Float32`, indistinguishable from a `Float32` that arrived from anywhere else. A computational graph is nothing more than that missing history, kept around on purpose."

**What you will understand by the end of this chapter:**

- Why an operation needs to implement *both* a forward and a backward method, bundled into one `Differentiable` trait, before a graph can ever record it — and why `MulOp.backward` specifically cannot be written from `grad_output` alone, without also seeing what the *other* input was
- `GraphNode`'s exact fields, traced field-by-field into the two nodes this chapter's running `w = x*y + x` example actually produces, down to the literal numbers `3.0`, `4.0`, `12.0`, and `15.0`
- The precise condition under which `ComputationGraph.record` skips creating a node at all — which is narrower than "this particular input is frozen," and worth stating exactly rather than loosely
- Why this book's `topological_backward_order` can get away with simply reversing the order nodes were appended, instead of running a real topological-sort algorithm from a specific output — and the one architectural assumption that safety depends on, checked against how reverse-mode automatic differentiation is implemented in production frameworks generally

**What you need to know first:**

- Chapter 12 (`elementwise_add` and `elementwise_mul` — the kernels `AddOp` and `MulOp` wrap; this chapter is about the bookkeeping layer built *on top of* those kernels, not a replacement for them)
- Chapter 11.1 (shared, non-copying buffer references — `GraphNode.inputs` holds the same tensors an operation was called with, not copies, the same sharing discipline `RefCountedBuffer` established)
- Chapter 6 (the `Tensor` struct this chapter extends with a `requires_grad` flag and, implicitly, a way for an output to remember which node produced it)

## Why a graph, and not just a function call?

Suppose you write the expression `w = x*y + x` in Mojo, with `x = 3.0` and `y = 4.0`. Running it forward is trivial:

```
z = x * y = 3.0 * 4.0 = 12.0
w = z + x = 12.0 + 3.0 = 15.0
```

Now ask a different question: *if `x` nudges up by a tiny amount, how much does `w` move?* Ordinary calculus answers this in one line — `w = xy + x`, so `∂w/∂x = y + 1 = 4 + 1 = 5` — but notice everything that answer silently depended on: that `w` was built from `z`, that `z` was built from `x` and `y` by multiplication, and that `x` feeds into `w` a *second* time, directly, through the addition. The instant Mojo finishes computing `15.0`, none of that structure survives; `w` is one `Float32`, with no record of how it got there.

**A computational graph is nothing more than that missing history, kept around on purpose.** Every time an operation runs, instead of only producing a value, it also records a note: "I am a multiply, my inputs were `x` and `y`, and I produced `12.0`." Chapter 16 shows that once every operation leaves a note like this, the chain rule applies *mechanically* to the whole chain of notes — no human derives `∂w/∂x = y + 1` symbolically ever again. This chapter builds the note itself (`GraphNode`), the mechanism that writes one during every forward computation, and the ordering rule for which notes to read first when walking backward.

Keep `x=3, y=4, z=12, w=15` in mind — every section below builds the graph these two operations actually produce, one field at a time.

## 15.1 Function Registration System `[FOUNDATIONAL]`

### Intuition

An operation can only be part of a graph if the framework knows two things about it: how to run it forward, and how to turn an "upstream sensitivity" into sensitivities for each of its inputs. Neither is optional on its own — an op with only a forward implementation can compute `w`, but can never explain how `w` depends on `x`, which makes it useless to a graph that exists specifically to answer that question. Bundling both directions into one trait, registered together, is what guarantees a graph can never end up holding a forward computation it has no idea how to differentiate.

### Background

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

`backward`'s signature is worth reading carefully: it takes `grad_output`, but also `inputs` and `output` — not just the upstream sensitivity by itself. That is not incidental plumbing. For `MulOp`, knowing only the upstream sensitivity isn't enough to answer anything: the sensitivity of `z = x*y` to `x` is literally the *value of `y`*, and the sensitivity to `y` is the value of `x`. There is no way to produce either number without seeing both operands, which is exactly why `backward` receives `inputs` at all — a design choice every reverse-mode autodiff implementation makes the same way, because the local derivative of a product is fundamentally "the other operand," not something derivable from the output alone.

### Worked Example 15.1.1 — What the registry needs before recording anything

For the running example, two ops must already exist in the registry before `w = x*y + x` can be recorded at all: a `MulOp` and an `AddOp`. Their `forward` methods are exactly the arithmetic already shown (`z = x*y`, `w = z+x`); their `backward` methods are what Chapter 16 derives in full, but the *shape* of the answer previews cleanly here. Calling `registry.get("mul")` before `MulOp` has ever been registered would trip the `debug_assert` in `OpRegistry.get` — but only in a build where `debug_assert` hasn't been compiled out. Chapter 11.2 already established that `debug_assert` is a debug-build-only check in Mojo; in a release build, an unregistered op name here falls straight through to `self._ops[name]`, a dictionary lookup on a key that was never inserted, with whatever failure mode Mojo's own `Dict` gives an absent key — not the friendly `"Unregistered op: mul"` message the debug build would have shown.

## 15.2 Gradient Function Traits `[FOUNDATIONAL]`

### Intuition

Once an op is registered, each time it actually *runs* inside a graph-tracked computation, the framework needs somewhere to keep the specific inputs and output from that one call — not the op's code (there's only one `MulOp` struct in existence), but this particular invocation of it, on these particular tensors. That per-call record is a `GraphNode`.

### Background

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

`grad` is initialized to zero at construction, not left unset — a small design choice with a large consequence Section 15.4 depends on directly: a node whose output is never actually consulted during a real backward pass still holds a well-defined, harmless `0.0` rather than uninitialized memory, so accidentally running its `backward` contributes exactly nothing to anyone's accumulated gradient.

### Worked Example 15.2.1 — The two nodes `w = x*y + x` actually produces

The multiply produces exactly one `GraphNode`:

```
GraphNode #0:
  op_name = "mul"
  inputs  = [x = 3.0, y = 4.0]
  output  = z = 12.0
  grad    = 0.0   (not yet known -- filled in during backward, Chapter 16)
```

and the addition produces a second one, whose `inputs` list contains `z` — the *tensor* node #0 produced, not a copy of the number `12.0`:

```
GraphNode #1:
  op_name = "add"
  inputs  = [z = 12.0, x = 3.0]
  output  = w = 15.0
  grad    = 0.0
```

### Worked Example 15.2.2 — What `requires_grad` actually gates

Suppose `y` had instead been a fixed hyperparameter, `requires_grad = False`, while `x` still needs gradients. It's tempting to read this as "the graph now skips building anything for `y`" — but `GraphNode` is recorded once *per operation*, not once per input, and Section 15.3's `record` method checks whether *any* input needs a gradient, not whether *every* input does. Since `x.requires_grad` is still `True`, the `mul` node above is still created in full, `y` included in its `inputs` list — the framework still needs it to compute `∂z/∂x = y`, even though nobody will ever ask for `∂z/∂y`. The condition that actually skips a node entirely is stricter: *every single input* to that operation must have `requires_grad = False` simultaneously. One frozen operand next to one trainable operand still gets a full node; the constant-folding-out only happens when a whole operation is built entirely from values nobody will ever differentiate — the case Section 15.3 makes precise.

## 15.3 Graph Construction During Forward Pass `[FOUNDATIONAL]`

### Intuition

There is no separate "build the graph" step in this framework — the graph is a side effect of running the forward pass once, normally. Every operation call either appends exactly one node or, if nothing about the call could ever need a gradient, appends nothing at all and returns a plain, untracked tensor.

### Background

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
    var result = elementwise_mul(a, b)              # Chapter 12's kernel
    return graph.record("mul", List[Tensor](a, b), result)

fn add(mut graph: ComputationGraph, a: Tensor, b: Tensor) -> Tensor:
    var result = elementwise_add(a, b)
    return graph.record("add", List[Tensor](a, b), result)
```

`record`'s loop is exactly the condition Worked Example 15.2.2 worked through: `needs_grad` becomes `True` the moment *any* input in the list has `requires_grad = True`, so skipping a node requires the loop to finish without ever setting it — every input frozen, simultaneously. `output.grad_fn_index = len(self.nodes)` is the other half of the bookkeeping: rather than the output tensor holding a pointer or reference back to the node that made it, it holds a plain integer — the position that node is about to occupy in `graph.nodes` — a flat-array-of-nodes design, as opposed to each tensor directly owning a reference to its own producing computation. The trade-off is real: a flat list threaded through every call (`mut graph: ComputationGraph` appears in every op's signature) makes the entire trace trivially inspectable — printing `graph.nodes` shows the whole computation — at the cost of every operation needing that graph object passed in explicitly, everywhere, rather than gradient machinery living entirely inside the tensor type itself.

### Worked Example 15.3.1 — Building the graph, one call at a time

Evaluating `w = add(graph, mul(graph, x, y), x)` with `x=3.0, y=4.0` does two things at once, exactly the way running the arithmetic by hand did: it produces `15.0`, and it leaves `graph.nodes` holding precisely the two `GraphNode`s from Worked Example 15.2.1, in the order they were computed:

```
graph.nodes[0] = GraphNode("mul", inputs=[x,y], output=z=12.0)
graph.nodes[1] = GraphNode("add", inputs=[z,x], output=w=15.0)
```

That list is the entire "history" this chapter opened by saying was normally thrown away — a literal, numeric trace of the computation that just ran, not anything symbolic or abstract.

## 15.4 Topological Sorting Implementation `[FOUNDATIONAL]`

### Intuition

Backward must undo the computation in the opposite order it was built: you cannot ask "how sensitive is `w` to `z`" until `w` exists, and `w`'s node was necessarily appended to `graph.nodes` *after* `z`'s node, because `z` had to be computed first to be usable as an input to the addition. That single fact — every node's inputs were computed, and therefore appended, strictly before it — means `graph.nodes` is already sorted in a valid forward order, and reversing it gives a valid order for backward with no separate sorting algorithm required, for this book's specific construction.

### Background

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

Notice what this function's signature does *not* take: a specific output tensor. It walks the entire `graph.nodes` list, unconditionally, from last-appended to first — there is no way for it to distinguish "give me the ancestors of `w`" from "give me the ancestors of some other tensor entirely." That is a real simplification, and it is safe *specifically* because every example in this book calls `backward()` from the single true final output of the whole graph that was ever built, with nothing else recorded alongside it. Reverse-mode automatic differentiation, implemented generally, doesn't get to assume that: a production engine builds its backward order with a depth-first search that starts *from the specific tensor `.backward()` was called on* and walks only that tensor's recorded parents, so nodes that were computed but never actually feed into the differentiated output are excluded from the walk entirely, by construction — not merely rendered harmless after the fact.

### Worked Example 15.4.1 — The running example's backward order

```
Forward order (how graph.nodes was built):   [ mul(x,y)→z,  add(z,x)→w ]
Reverse order (the order backward will use): [ add(z,x)→w,  mul(x,y)→z ]
```

`topological_backward_order` returns `[1, 0]` — node `1` (`add`) first, node `0` (`mul`) second. Read that as a to-do list: first figure out how sensitive the final `w` is to `z` and to `x` through the addition; only afterward, once `z`'s sensitivity is known, ask how sensitive `z` is to `x` and `y` through the multiplication. Doing it the other way around would mean asking "how does `z` affect `x` and `y`" before knowing how much `z` itself matters to the answer — not yet a meaningful question.

### Worked Example 15.4.2 — A node that is recorded but isn't an ancestor of `w`

Extend the example with one more line, computed but never used: `q = subtract(graph, x, one)`, run *between* the `mul` call and the `add` call. `graph.nodes` now holds three entries in append order: `[mul(x,y)→z, sub(x,1)→q, add(z,x)→w]`. `topological_backward_order` still just reverses the whole list: `[2, 1, 0]` — `add` first, then `sub`, then `mul`. Node `1` (`sub`, producing `q`) is not an ancestor of `w` at all; `w` was built from `z` and `x`, never from `q`. Running `sub`'s `backward` anyway isn't wrong here, only wasted: `GraphNode.__init__` zero-initializes every node's `grad`, and nothing in this trace ever assigns `q`'s node a nonzero gradient, so `sub`'s backward computes contributions of exactly `0.0` into `x` — harmless, but genuinely unnecessary work, and a concrete illustration of exactly the gap Section 15.4's Intuition described: reversing *every recorded node* is not the same operation as walking the *ancestors of one specific output*, even though the two happen to agree whenever nothing dead ever gets recorded alongside the computation you actually care about.

## Chapter Summary

A computational graph exists to keep, on purpose, the history a forward pass would otherwise throw away the moment it finishes. `Differentiable` bundles a `forward` and `backward` method into one trait so a graph can never record an operation it has no idea how to differentiate, and `backward`'s signature takes `inputs` (not just `grad_output`) because a local derivative like multiplication's — the *other* operand — genuinely cannot be produced any other way. `GraphNode` captures one specific invocation of an op: its inputs (shared, not copied), its output, and a `grad` field zero-initialized up front rather than left unset. `ComputationGraph.record` appends a node only when *at least one* input requires a gradient — a looser condition than "this particular input is frozen," since one trainable operand next to one frozen one still produces a full node. Because every node's inputs are computed, and therefore appended, strictly before it, the append order is already a valid forward topological order, and `topological_backward_order` gets away with simply reversing it — a simplification that holds exactly because this book never calls `backward()` on anything but the graph's true final output, and never leaves an unrelated, unused node recorded alongside the one that matters; production reverse-mode engines instead build their backward order from a depth-first search rooted at the specific output being differentiated, for exactly the cases where that assumption wouldn't hold.

## Self-Check Questions

1. `w = x*y + x` is built as `add(graph, mul(graph, x, y), x)` with `x=5.0, y=2.0`. Trace both `GraphNode`s exactly as Worked Example 15.2.1 did — report every field of `GraphNode #0` and `GraphNode #1`, including the final values of `z` and `w`.
2. Suppose *both* `x` and `y` have `requires_grad = False` for a call to `mul(graph, x, y)`. Walk through `record`'s loop step by step. Is a `GraphNode` created? What does the function return instead?
3. Extend Worked Example 15.4.2's three-node graph with a fourth call, `r = mul(graph, q, x)`, run after `add`. Does `r`'s node make `q`'s `sub` node an ancestor of `w`? Does it matter to `w.backward()`'s correctness that `sub`'s node exists in `graph.nodes` either way?
4. `OpRegistry.get("relu")` is called before any `ReluOp` has ever been registered, in a release build where `debug_assert` has been compiled out. What actually happens, in contrast to what would happen in a debug build?
5. Why does `GraphNode.__init__` initialize `grad` to `Tensor.zeros(output.shape)` rather than leaving it in whatever state a freshly allocated `Tensor` would otherwise be in? Connect your answer directly to what Worked Example 15.4.2 showed about running an unrelated node's `backward` by mistake.

## Where We Go Next

Chapter 16 (`part4/01-backward-function-implementation.md`) derives what each `backward` method in this chapter's `Differentiable` trait actually computes — starting with the exact numbers this chapter has been building toward, `x.grad = 5.0` and `y.grad = 3.0` for the running `w = x*y + x` example — by walking `topological_backward_order`'s list and applying the chain rule at each node in turn.

## Worked Solutions

**1.** `z = x*y = 5.0 × 2.0 = 10.0`; `w = z + x = 10.0 + 5.0 = 15.0`. `GraphNode #0`: `op_name="mul"`, `inputs=[x=5.0, y=2.0]`, `output=z=10.0`, `grad=0.0`. `GraphNode #1`: `op_name="add"`, `inputs=[z=10.0, x=5.0]`, `output=w=15.0`, `grad=0.0`.

**2.** The loop checks each of `x` and `y` in turn: `x.requires_grad` is `False`, so the `if` body never runs for it; `y.requires_grad` is also `False`, same result. After the loop, `needs_grad` is still `False` (its initial value), so the function hits `return output` immediately — no `GraphNode` is created, and the returned tensor is the plain, untracked multiplication result, indistinguishable from a value computed with no graph involved at all.

**3.** No — `r = mul(graph, q, x)` makes `q` (and therefore `sub`'s node) an ancestor of `r`, not of `w`. `w` was already fully computed by the earlier `add(graph, z, x)` call, using only `z` and `x`; nothing about `w`'s own definition changes because a later, unrelated computation happens to reuse `q` and `x` afterward. It does not matter to `w.backward()`'s correctness whether `sub`'s node exists in `graph.nodes` — per Worked Example 15.4.2, an unrelated node's `backward` always contributes exactly `0.0` to whatever it touches, since its own `grad` field is never set to anything nonzero by a walk that never reaches it through `w`'s actual dependency chain.

**4.** In the debug build, `debug_assert(name in self._ops, "Unregistered op: relu")` fires before the lookup, halting with that specific message. In a release build, the assertion is compiled out entirely — execution falls straight through to `return self._ops[name]`, a dictionary lookup on a key, `"relu"`, that was never inserted, triggering whatever Mojo's own `Dict` type does for a missing key (a runtime fault with a generic message, not the descriptive one the debug build would have shown) — the exact same debug-assert-only-checks-anything gap Chapter 11.2 identified for `Arena`'s bounds checking.

**5.** Leaving `grad` unset (or relying on whatever a fresh `Tensor` defaults to) would make a node's gradient state ambiguous the moment nothing writes to it — is `0.0` from initialization, or is `0.0` because a caller genuinely computed a zero contribution, or is it uninitialized memory that happens to print as something else entirely? Zero-initializing at construction removes that ambiguity outright, and it's precisely what makes Worked Example 15.4.2 safe: `sub`'s node never has its `grad` touched by anything downstream, so its `backward` reads a well-defined `0.0` and produces a well-defined, harmless zero contribution — not a gamble on whatever bytes happened to be sitting in memory when the node was constructed.
