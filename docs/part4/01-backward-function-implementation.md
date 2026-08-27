# Chapter 16: Backward Function Implementation — The Chain Rule, One Node at a Time

> "Chapter 15 ended with a graph for `w = x*y + x` and a to-do list for backward: visit `add(z,x)→w` first, then `mul(x,y)→z`. This chapter works out, by hand, exactly what happens at each stop on that list — and only afterward writes the Mojo that automates it."

**What you will understand by the end of this chapter:**

- The multivariable chain rule as literally "sum the contribution from every path" — traced on `x`'s two separate routes into `w`, reaching the same `∂w/∂x = 5` Chapter 15 got from plain calculus
- `AddOp` and `MulOp`'s exact backward rules, why `MulOp`'s local derivative is fundamentally *the other operand's value*, and a genuine open question about tensor aliasing that this chapter's own `AddOp.backward` raises but can't fully answer until Chapter 17
- Why `ExpOp` reads the cached forward `output` instead of recomputing `e^x` — and why `GraphNode` had to store `output` at all, back in Chapter 15, for that shortcut to even be possible
- `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, derived from index-summation first principles rather than only asserted, and verified with real numbers on *both* gradients this chapter's running matrix example produces
- The implicit function theorem as an escape hatch for differentiating through an iterative numerical solver — a bisection search — without ever unrolling or unrolling-and-differentiating a single one of its steps

**What you need to know first:**

- Chapter 15 (the `Differentiable` trait's `backward` signature, `GraphNode`'s `inputs`/`output` fields, and `topological_backward_order`'s `[1, 0]` to-do list — this chapter is entirely about what happens at each stop on that list)
- Chapter 13.1 (matrix multiplication and the `X (2×3) @ M (3×2)` running example this chapter's `MatMulOp` backward reuses directly)
- Chapter 12 (`elementwise_add`, `elementwise_mul`, and `elementwise_exp` — the kernels `AddOp`, `MulOp`, and `ExpOp` wrap in their `forward` methods)

## 16.1 Chain Rule Implementation `[FOUNDATIONAL]`

### Intuition

Chapter 15 posed the question by pure calculus: if `x` moves by a tiny amount, how much does `w` move? `w = xy + x`, so `∂w/∂x = y + 1 = 5`. What that one-line answer hides is that `x` doesn't reach `w` by one route — it reaches it by two, and the multivariable chain rule's actual content is nothing more sophisticated than "trace every route separately, then add up what each one contributes."

```
        ┌── (Route 1: through the multiply) ──┐
        │                                      ▼
   x ───┼────────────────────────────────▶ [ mul ] ──▶ z ──▶ [ add ] ──▶ w
        │                                                       ▲
        └── (Route 2: directly) ───────────────────────────────┘
```

### Background

**Route 1 — through the multiply.** `x` feeds into `z = x*y`. If `x` increases by a small `Δx`, `z` increases by approximately `Δx · y`, since `∂z/∂x = y = 4`. That change in `z` then feeds into `w = z + x`, where `∂w/∂z = 1`, so it passes straight through unchanged: `w` moves by `Δx · y · 1 = 4·Δx`.

**Route 2 — directly.** `x` *also* feeds straight into the addition `w = z + x`, independent of `z`. That contributes an additional `∂w/∂x|_{direct} = 1`, so `w` moves by another `1·Δx`.

**Total.** Both routes act on `w` simultaneously, so their contributions add: `w` moves by `(4 + 1)·Δx = 5·Δx` — exactly `∂w/∂x = 5`, now arrived at by summing contributions along every path from `x` to `w` rather than by symbolic differentiation of the whole expression at once. This *sum-over-paths* rule is the entire content of the multivariable chain rule, and it is also, not coincidentally, exactly what the reverse graph traversal in Chapter 17 computes — one path's contribution per visit to a node that uses `x`, summed as they arrive.

In code, "the sensitivity flowing into a node, converted into a sensitivity for each of its inputs" is one dispatch call:

```mojo
fn chain_rule_step(
    op_name: String,
    grad_output: Tensor,
    inputs: List[Tensor],
    output: Tensor,
) -> List[Tensor]:
    """Dispatch to the registered backward for this op. The caller
    (Chapter 17) adds each result into the corresponding input's
    accumulated .grad."""
    var op = registry.get(op_name)
    return op.backward(grad_output, inputs, output)
```

### Worked Example 16.1.1 — Reconciling two routes into one number

Section 15.4's `topological_backward_order` says: visit `add(z,x)→w` first, then `mul(x,y)→z`. Visiting `add` first is exactly what makes Route 2's contribution (the *direct* `1·Δx`) available immediately, and visiting `mul` second is what makes Route 1's contribution (the `4·Δx` that had to flow *through* `z` first) available only after `z`'s own sensitivity is already known. Neither route can be skipped, and neither can run before the node that produces it — which is precisely why the visiting order from Chapter 15 isn't optional bookkeeping, it's a dependency the arithmetic itself imposes.

## 16.2 Element-wise Operation Gradients `[FOUNDATIONAL]`

### Intuition

`AddOp` and `MulOp` are Sections 16.1's two routes made concrete. `Add`'s local derivative is `1` for both inputs — a sensitivity that passes straight through unchanged, which is exactly what "direct" meant in Route 2. `Mul`'s local derivative is *the other operand's value* — which is exactly why `Differentiable.backward` was written, back in Chapter 15, to receive `inputs` and not merely `grad_output`.

<a id="72-element-wise-operation-gradients"></a>
### Background

```mojo
struct AddOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_add(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a+b)/da = 1, d(a+b)/db = 1 -- gradient passes through unchanged
        return List[Tensor](grad_output, grad_output)

struct MulOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_mul(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a*b)/da = b, d(a*b)/db = a
        var grad_a = elementwise_mul(grad_output, inputs[1])
        var grad_b = elementwise_mul(grad_output, inputs[0])
        return List[Tensor](grad_a, grad_b)
```

### Worked Example 16.2.1 — `AddOp`, by hand

`w = z + x`. Backward starts by being handed the sensitivity of the *final* output with respect to `w` itself — the *seed* — which is `1.0` by definition (`w` is exactly as sensitive to itself as it is to itself). `∂w/∂z = 1` and `∂w/∂x = 1`, so both of `AddOp`'s inputs simply receive a copy of whatever sensitivity arrived: with a seed of `1.0`, `AddOp.backward` returns `[1.0, 1.0]` — the first `1.0` is `z`'s incoming gradient, the second is `x`'s *first* contribution, Route 2 from Section 16.1. Hold onto that `x: 1.0` — Chapter 17 adds a second contribution to it.

```
[COMMON TRAP]  AddOp.backward hands out the same gradient tensor twice

`return List[Tensor](grad_output, grad_output)` places two copies of
the SAME Tensor struct value into the returned list -- not two
independently allocated gradient tensors. Tensor, established in
Chapter 6.3, wraps a raw UnsafePointer for its own data buffer;
copying a Tensor struct value copies that pointer, not the memory it
points to. So "z's incoming gradient" and "x's first contribution"
from this single AddOp.backward call are, at the byte level, the same
allocation viewed through two separate Tensor values.

Whether this is actually safe depends entirely on what Chapter 17's
accumulate_gradient does with each contribution next. If it
accumulates by producing a NEW tensor (elementwise_add building a
fresh buffer) and reassigning .grad to that new value, there is no
problem: z.grad and x.grad simply start out pointing at the same
memory and diverge the moment either one is reassigned to something
new. If it instead accumulates by mutating a gradient buffer in
place -- x.grad's underlying memory updated directly, not replaced --
then updating x's gradient with MulOp's second contribution would
silently corrupt z's gradient too, since both currently point at
identical memory. This chapter's own numbers don't reveal which case
applies; it's worth watching for directly in Chapter 17's
accumulate_gradient implementation.
```

### Worked Example 16.2.2 — `MulOp`, by hand

`z = x*y`. `∂z/∂x = y = 4` and `∂z/∂y = x = 3` — each input's local derivative is *the other* input's value, which is exactly why `backward` is passed `inputs` and not just `grad_output`. `MulOp` receives `z`'s gradient from the step above (`1.0`, since `∂w/∂z=1`), and multiplies it by each local derivative: `x` receives `1.0 × 4 = 4.0`, `y` receives `1.0 × 3 = 3.0`.

Add `x`'s two contributions from the two nodes — `1.0` from `AddOp` and `4.0` from `MulOp` — and the total is `5.0`. That is `x.grad`, and it is exactly `∂w/∂x = 5`, computed three different ways now: by plain calculus (Chapter 15), by tracing paths (Section 16.1), and now by literally running the two registered backward functions in order and adding what they hand back. `y` only appears in one node, so `y.grad = 3.0` directly, with no accumulation needed at all.

### Worked Example 16.2.3 — `ExpOp`, and the case that needs `output`

Neither `AddOp` nor `MulOp` needs the node's *output*, only its inputs — but some ops do, and it's worth seeing one before Chapter 17 explains why that matters for memory. Take `u = exp(x)` at `x = 1.0`. Forward: `u = e¹ ≈ 2.71828`. The derivative of `exp` is famously itself: `du/dx = eˣ = u ≈ 2.71828`. With an upstream seed of `1.0`, `ExpOp.backward` returns `1.0 × 2.71828 = 2.71828` — and it gets that `2.71828` by reading `output`, the value forward already computed, rather than recomputing `exp(1.0)` a second time:

```mojo
struct ExpOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_exp(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(e^x)/dx = e^x = output -- reuse the cached forward result
        return List[Tensor](elementwise_mul(grad_output, output))
```

This is exactly why `GraphNode` in Chapter 15 stores `output` alongside `inputs` — `ExpOp` is proof that a backward rule can legitimately need the forward answer, not just the forward arguments, and a `GraphNode` that only kept `inputs` would make `ExpOp`'s shortcut impossible to write at all.

## 16.3 Matrix Operation Gradients `[FOUNDATIONAL]`

### Intuition

Matrix multiplication's backward rule isn't a restatement of a forward formula the way `Add` and `Mul`'s were — every output entry of `Y = X @ M` depends on an entire row of `X` and an entire column of `M` (Chapter 13.1), so a single output's sensitivity has to be redistributed back across many input entries at once, not just one. The rule turns out to have a clean closed form, and it's worth deriving *why* before simply trusting it.

### Background

Write the forward pass in index form, the same way Chapter 13.1 did: `Y[i,j] = Σ_k X[i,k]·M[k,j]`. The chain rule says `∂L/∂X[i,k]` sums the contribution of `X[i,k]` through *every* output entry it participates in — and `X[i,k]` appears in `Y[i,j]` for every `j`, since it's row `i` of `X` that feeds every column of the output:

```
∂L/∂X[i,k] = Σ_j (∂L/∂Y[i,j] · ∂Y[i,j]/∂X[i,k])
           = Σ_j (∂L/∂Y[i,j] · M[k,j])
           = (∂L/∂Y @ Mᵀ)[i,k]
```

The middle step uses `∂Y[i,j]/∂X[i,k] = M[k,j]` directly from the forward formula — `X[i,k]` only ever multiplies `M[k,j]` inside the sum that produces `Y[i,j]`. The final step is just recognizing that "sum over `j` of `∂L/∂Y[i,j]` times `M[k,j]`" is precisely what a matrix product against `Mᵀ` computes, entry by entry. The symmetric derivation for `M` gives `∂L/∂M[k,j] = Σ_i (∂L/∂Y[i,j] · X[i,k]) = (Xᵀ @ ∂L/∂Y)[k,j]`. In code:

```mojo
struct MatMulOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return matrix_multiply(inputs[0], inputs[1])   # Chapter 13
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        var a = inputs[0]
        var b = inputs[1]
        var grad_a = matrix_multiply(grad_output, transpose(b))
        var grad_b = matrix_multiply(transpose(a), grad_output)
        return List[Tensor](grad_a, grad_b)
```

### Worked Example 16.3.1 — `dL/dX`, with real numbers

Reuse Chapter 13.1's exact running example: `X (2×3) @ M (3×2) = Y`, where

```
Y = X @ M =  22  28
             49  64
```

Suppose `Y` feeds into a loss whose gradient with respect to every entry of `Y` happens to be `1` — `grad_output` is a `2×2` matrix of ones, the simplest possible upstream signal. `Mᵀ` (transposing the `3×2` `M`) is:

```
Mᵀ = 1  3  5
     2  4  6
```

`grad_output @ Mᵀ` — a `2×2` matrix of ones times the `2×3` `Mᵀ` above — gives a `2×3` result where every entry is just the *column sum* of `Mᵀ`, because multiplying by a row of ones sums the column it's dotted against:

```
dL/dX = (1·1+1·2)  (1·3+1·4)  (1·5+1·6)     3   7  11
        (1·1+1·2)  (1·3+1·4)  (1·5+1·6)  =  3   7  11
```

Check the shape before trusting the arithmetic: `X` was `[2,3]`, and `dL/dX` is also `[2,3]` — a gradient with the wrong shape is a wrong gradient before a single number is even checked, and this shape-matching test is the cheapest sanity check for any new backward rule added to the registry.

### Worked Example 16.3.2 — `dL/dM`, completing the pair

The same `grad_output` also has to produce a gradient for `M`, using `∂L/∂M = Xᵀ @ ∂L/∂Y`. `Xᵀ` (transposing the `2×3` `X`) is:

```
Xᵀ = 1  4
     2  5
     3  6
```

`Xᵀ @ grad_output` — the `3×2` `Xᵀ` above times a `2×2` matrix of ones — gives a `3×2` result where every entry is the *row sum* of `Xᵀ`, since dotting any row of `Xᵀ` against a column of all ones just adds that row's two entries together:

```
dL/dM = (1+4)  (1+4)     5  5
        (2+5)  (2+5)  =  7  7
        (3+6)  (3+6)     9  9
```

Shape check: `M` was `[3,2]`, and `dL/dM` is also `[3,2]` — both gradients pass the cheap sanity check `MatMulOp.backward` needs to satisfy for *any* shapes, not just this chapter's specific `2×3` and `3×2`.

## 16.4 Custom Function Framework `[FOUNDATIONAL]`

### Intuition

Not every operation belongs in the core registry as a hand-differentiated forward/backward pair over a closed-form expression. Some values come from an *iterative* numerical procedure — a solver that runs a loop and converges toward an answer rather than computing one in a single formula — and differentiating through every step of that loop would be both wasteful and numerically noisy. The **implicit function theorem** is the escape hatch: treat the solver's output as *defined implicitly* by an equation it satisfies at convergence, and differentiate that equation instead of the solver's control flow.

<a id="74-custom-function-framework"></a>
### Background

Consider solving `x² = 2` for `x` with a numerical bisection search rather than `sqrt` — a stand-in, at small scale, for the bond-pricing bisection Part 7 actually differentiates through. Bisection between `a=1` and `b=2`: midpoint `1.5² = 2.25` (too big, move `b` to `1.5`); midpoint `1.25² = 1.5625` (too small, move `a` to `1.25`); midpoint `1.375² = 1.890625`; continuing this halves the bracket every step and converges toward `x ≈ 1.41421` (`√2`). The solver defines `x` implicitly by `f(x, c) = x² - c = 0`, where `c` is the constant being solved against (`c=2` here). The implicit function theorem says:

```
dx/dc = -(∂f/∂c) / (∂f/∂x) = -(-1) / (2x) = 1 / (2x)
```

The framework captures this as one opaque node with a closed-form backward, instead of an unrolled, differentiated bisection loop:

```mojo
struct CustomFunction(Differentiable):
    var forward_fn: fn(List[Tensor]) -> Tensor
    var backward_fn: fn(Tensor, List[Tensor], Tensor) -> List[Tensor]

    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return self.forward_fn(inputs)

    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        return self.backward_fn(grad_output, inputs, output)

fn sqrt_via_bisection_backward(grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
    # output holds the converged x; dx/dc = 1 / (2x) from the implicit function theorem
    var local_grad = Float32(1.0) / (Float32(2.0) * output.item())
    return List[Tensor](elementwise_mul(grad_output, local_grad))
```

### Worked Example 16.4.1 — Checking the implicit-function gradient against finite differences

At the converged solution `x ≈ 1.41421`, `dx/dc = 1 / (2 × 1.41421) = 1 / 2.82842 ≈ 0.35355`. Check this directly against a finite-difference nudge of `c`, the same way earlier chapters checked local derivatives: bisecting `x² = 2.001` instead of `x² = 2` converges to `√2.001 ≈ 1.414568`. The finite-difference slope is `(1.414568 - 1.41421) / 0.001 ≈ 0.358` — close to the implicit-function answer of `0.35355`, with the small remaining gap being ordinary finite-difference approximation error, the same kind the neural-network-layers chapter's `gradient_check` tolerates, not a discrepancy in the calculus. The entire multi-step bisection loop collapses, for gradient purposes, into one multiplication by `1/(2x)` — exactly the pattern the z-spread bisection in Part 7 reuses: the forward pass runs the numerical solver as ordinary control flow, and one hand-derived `backward_fn` plugs the whole iterative procedure into the graph as a single differentiable op.

## Chapter Summary

The multivariable chain rule is nothing more than summing a value's contribution along every path it takes to reach the output — traced concretely on `x`'s two routes into `w`, matching `∂w/∂x = 5` three separate ways. `AddOp` passes its incoming gradient through unchanged to both inputs (though, as this chapter flagged and left open for Chapter 17, it does so by handing out the *same* underlying Tensor value twice, not two independent ones); `MulOp` scales the incoming gradient by whichever input it *isn't* computing the gradient for, which is exactly why `backward` needs `inputs` at all. `ExpOp` demonstrated the other half of `GraphNode`'s design: some backward rules need the forward `output`, not just the forward arguments, to avoid recomputing an identical value a second time. `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, isn't just asserted in this chapter — it's derived from the same index-summation form Chapter 13.1 used for the forward pass, then verified with real numbers on both `dL/dX` (`[[3,7,11],[3,7,11]]`) and `dL/dM` (`[[5,5],[7,7],[9,9]]`), each checked against its own operand's shape. Finally, the implicit function theorem showed that a value produced by an iterative solver — bisection, standing in for the bond-pricing solver Part 7 differentiates through — doesn't need its loop unrolled and differentiated step by step; treating the converged answer as implicitly defined by the equation it satisfies collapses the entire gradient into one closed-form expression, verified here against a finite-difference check the same way every local derivative earlier in this book was checked.

## Self-Check Questions

1. For `w = x*y + x` with `x=5.0, y=2.0` (the numbers from Chapter 15's Self-Check Question 1), trace both backward steps: what does `AddOp.backward` return, what does `MulOp.backward` return, and what is the final `x.grad`?
2. `MulOp.backward` computes `grad_a = elementwise_mul(grad_output, inputs[1])`. If `inputs[1]` (i.e. `y`) were `0.0` instead of `4.0`, what would `x`'s contribution from this node be, and does that match what `∂z/∂x = y` predicts when `y = 0`?
3. Using the same index-summation derivation Section 16.3 used for `∂L/∂X`, and given `grad_output` is *not* a matrix of all ones but instead `[[1, 0], [0, 1]]` (the 2×2 identity), compute `dL/dX = grad_output @ Mᵀ` for this chapter's running `M`. (Recall `Mᵀ = [[1,3,5],[2,4,6]]`.)
4. `ExpOp.backward` reads `output` rather than recomputing `elementwise_exp(inputs[0])`. Suppose `GraphNode` had never stored `output` at all — only `inputs`. Could `ExpOp.backward` still be written correctly? At what cost?
5. In the bisection example, if the target were `x² = 9` instead of `x² = 2` (converging to `x = 3`), what would `dx/dc` be at convergence, using the same implicit-function formula `dx/dc = 1/(2x)`?

## Where We Go Next

Chapter 17 (`part4/02-gradient-computation-engine.md`) is where every backward rule this chapter derived actually gets *run*: seeding `w.grad = 1.0`, walking `[add, mul]` — the reverse order Chapter 15 established — calling `chain_rule_step` at each stop, and accumulating `x`'s two contributions (`1.0` from `AddOp`, `4.0` from `MulOp`) into a final `x.grad = 5.0`. It's also where this chapter's open question about `AddOp.backward` returning one aliased Tensor twice finally gets an answer, in whichever direction `accumulate_gradient` actually implements it.

## Worked Solutions

**1.** `AddOp.backward` (seed `1.0`) returns `[1.0, 1.0]` — `z`'s gradient and `x`'s first contribution. `MulOp.backward`, receiving `z`'s gradient of `1.0` and `inputs = [x=5.0, y=2.0]`, computes `grad_x = 1.0 × y = 1.0 × 2.0 = 2.0` and `grad_y = 1.0 × x = 1.0 × 5.0 = 5.0`. Final `x.grad = 1.0 (from AddOp) + 2.0 (from MulOp) = 3.0`; `y.grad = 5.0` directly. Cross-check with calculus: `∂w/∂x = y+1 = 2+1 = 3` and `∂w/∂y = x = 5` — both match.

**2.** With `y = 0.0`, `grad_a = elementwise_mul(grad_output, 0.0) = 0.0` — `x`'s contribution from the `mul` node would be exactly `0.0`. This matches `∂z/∂x = y` directly: when `y = 0`, nudging `x` doesn't move `z = x·y` at all, since anything times `0` is `0`, so a local derivative of `0` is exactly correct, not a sign of anything broken.

**3.** `grad_output @ Mᵀ` with `grad_output = [[1,0],[0,1]]` and `Mᵀ = [[1,3,5],[2,4,6]]`: row `0` of the identity picks out row `0` of `Mᵀ` unchanged: `[1,3,5]`. Row `1` of the identity picks out row `1` of `Mᵀ` unchanged: `[2,4,6]`. So `dL/dX = [[1,3,5],[2,4,6]]` — multiplying by the identity matrix, unsurprisingly, just returns the other operand exactly, the same `A @ I = A` fact Chapter 13.4 verified for forward matrix multiplication, now shown to hold for this backward computation too.

**4.** No — not correctly, not without extra cost. `d(eˣ)/dx = eˣ`, and the *only* two values available would be `grad_output` and `inputs[0]` (the original `x`, not `eˣ`). `ExpOp.backward` would have to recompute `elementwise_exp(inputs[0])` from scratch inside `backward` — a second, redundant evaluation of `exp` for every single backward pass, doubling the exponential evaluations this op ever needs to perform, purely because the forward result wasn't kept around. `GraphNode` storing `output` is precisely what avoids paying that cost.

**5.** `dx/dc = 1/(2x)` at `x=3`: `1/(2×3) = 1/6 ≈ 0.16667`. This is a smaller sensitivity than the `x=√2` case (`≈0.35355`) because the derivative shrinks as `x` grows — the same `1/(2x)` shape that makes `sqrt`'s own derivative, `d(√c)/dc = 1/(2√c)`, flatten out for larger inputs.
