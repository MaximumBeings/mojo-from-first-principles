# Chapter 7: Backward Function Implementation

Chapter 6 ended with a graph for `w = x*y + x` (`x=3, y=4, z=12, w=15`) and a to-do list for backward: visit `add(z,x)→w` first, then `mul(x,y)→z`. This chapter works out, by hand, exactly what happens at each stop on that list — and only afterward writes the Mojo that automates it.

## 7.1 Chain Rule Implementation

Start with the question Chapter 6 posed: if `x` moves by a tiny amount, how much does `w` move? Trace the ripple in two separate steps, because `x` reaches `w` by two different routes.

**Route 1 — through the multiply.** `x` feeds into `z = x*y`. If `x` increases by a small amount `Δx`, then `z` increases by approximately `Δx · y` — because `∂z/∂x = y = 4`. That change in `z` then feeds into `w = z + x`, where `∂w/∂z = 1`, so it passes straight through unchanged: `w` moves by `Δx · y · 1 = 4·Δx`.

**Route 2 — directly.** `x` *also* feeds straight into the addition `w = z + x`, independent of `z`. That contributes an additional `∂w/∂x|_{direct} = 1`, so `w` moves by another `1·Δx`.

**Total.** Since both routes act on `w` simultaneously, their contributions add: `w` moves by `(4 + 1)·Δx = 5·Δx`, which is exactly `∂w/∂x = 5` — the answer Chapter 6 got from plain calculus, now arrived at by adding up contributions along every path from `x` to `w`. This *sum-over-paths* rule is the entire content of the multivariable chain rule, and it is also, not coincidentally, exactly what the reverse graph traversal in Chapter 8 computes — one path's contribution per visit to a node that uses `x`, summed as they arrive.

In code, "the sensitivity flowing into a node, converted into a sensitivity for each of its inputs" is one function call:

```mojo
fn chain_rule_step(
    op_name: String,
    grad_output: Tensor,
    inputs: List[Tensor],
    output: Tensor,
) -> List[Tensor]:
    """Dispatch to the registered backward for this op. The caller
    (Chapter 8) adds each result into the corresponding input's
    accumulated .grad."""
    var op = registry.get(op_name)
    return op.backward(grad_output, inputs, output)
```

Sections 7.2–7.4 are the catalog of what `op.backward` actually computes for each kind of operation — starting with the two the running example needs.

## 7.2 Element-wise operation gradients

**`AddOp`, worked by hand.** `w = z + x`. Backward starts by being handed the sensitivity of the *final* output with respect to `w` itself — call it the *seed* — which is `1.0` by definition (`w` is exactly as sensitive to itself as it is to itself). `∂w/∂z = 1` and `∂w/∂x = 1`, so both of `AddOp`'s inputs simply receive a copy of whatever sensitivity arrived: with a seed of `1.0`, `z` receives `1.0` and `x` receives `1.0`.

```mojo
struct AddOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_add(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a+b)/da = 1, d(a+b)/db = 1 -- gradient passes through unchanged
        return List[Tensor](grad_output, grad_output)
```

Applied to the running example with `grad_output = 1.0`: `AddOp.backward` returns `[1.0, 1.0]` — the first `1.0` is `z`'s incoming gradient, the second is `x`'s *first* contribution (Route 2 from Section 7.1). Hold onto that `x: 1.0` — Section 7.3 in Chapter 8 will add a second contribution to it.

**`MulOp`, worked by hand.** `z = x*y`. `∂z/∂x = y = 4` and `∂z/∂y = x = 3` — each input's local derivative is *the other* input's value, which is exactly why `backward` is passed `inputs` and not just `grad_output`. `MulOp` receives `z`'s gradient from the step above (`1.0`, since `∂w/∂z=1`), and multiplies it by each local derivative: `x` receives `1.0 × 4 = 4.0`, `y` receives `1.0 × 3 = 3.0`.

```mojo
struct MulOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_mul(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a*b)/da = b, d(a*b)/db = a
        var grad_a = elementwise_mul(grad_output, inputs[1])
        var grad_b = elementwise_mul(grad_output, inputs[0])
        return List[Tensor](grad_a, grad_b)
```

Add `x`'s two contributions from the two nodes — `1.0` from `AddOp` and `4.0` from `MulOp` — and the total is `5.0`. That is `x.grad`, and it is exactly the `∂w/∂x = 5` computed two different ways already, in Chapter 6 by calculus and in Section 7.1 by tracing paths. `y` only appears in one node, so `y.grad = 3.0` directly. Chapter 8 formalizes "add the two contributions" as `accumulate_gradient` — everything numeric about it is already sitting right here.

**A second worked example, for `ExpOp`.** Neither `add` nor `mul` needs the node's *output*, only its inputs — but some ops do, and it's worth seeing one before Chapter 8 explains why that matters for memory. Take `u = exp(x)` at `x = 1.0`. Forward: `u = e^1 ≈ 2.71828`. The derivative of `exp` is famously itself: `du/dx = e^x = u ≈ 2.71828`. With an upstream seed of `1.0`, `ExpOp.backward` returns `1.0 × 2.71828 = 2.71828` — and it gets that `2.71828` by reading `output`, the value forward already computed, rather than recomputing `exp(1.0)` a second time:

```mojo
struct ExpOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_exp(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(e^x)/dx = e^x = output -- reuse the cached forward result
        return List[Tensor](elementwise_mul(grad_output, output))
```

This is exactly why `GraphNode` in Chapter 6 stores `output` alongside `inputs` — `ExpOp` is proof that a backward rule can legitimately need the forward answer, not just the forward arguments.

## 7.3 Matrix Operation Gradients

Matrix multiplication's backward rule isn't a restatement of a forward formula the way `Add` and `Mul` were — it's derived from the contraction rule, and it's worth deriving on the exact numbers [Chapter 4.1](../part2/02-matrix-operations.md#41-matrix-multiplication) already worked by hand: `X` (2×3) times `M` (3×2) gave

```
Y = X @ M =  22  28
             49  64
```

Suppose `Y` feeds into a loss whose gradient with respect to every entry of `Y` happens to be `1` (i.e. `grad_output` is a 2×2 matrix of ones — the simplest possible upstream signal). The rule for matmul's backward is:

```
dL/dX = dL/dY @ Mᵀ
dL/dY_that_is_actually_dL/dM = Xᵀ @ dL/dY
```

Work `dL/dX` out with real numbers. `Mᵀ` (transposing the 3×2 `M` from Chapter 4.1) is:

```
Mᵀ = 1  3  5
     2  4  6
```

`grad_output @ Mᵀ` — a 2×2 matrix of ones, times the 2×3 `Mᵀ` above — gives a 2×3 result where every entry is just the *column sum* of `Mᵀ`, because multiplying by a row of ones sums the column it's dotted against:

```
dL/dX = (1·1+1·2)  (1·3+1·4)  (1·5+1·6)     3   7  11
        (1·1+1·2)  (1·3+1·4)  (1·5+1·6)  =  3   7  11
```

Check the shape before trusting the arithmetic: `X` was `[2,3]`, and `dL/dX` above is also `[2,3]` — a gradient with the wrong shape is a wrong gradient before a single number is even checked, and this shape-matching test is the cheapest sanity check for any new backward rule added to the registry. Now the code that produces this mechanically, for any `X`, `M`, and any upstream gradient — not just the all-ones example above:

```mojo
struct MatMulOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return matrix_multiply(inputs[0], inputs[1])   # Chapter 4
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        var a = inputs[0]
        var b = inputs[1]
        var grad_a = matrix_multiply(grad_output, transpose(b))
        var grad_b = matrix_multiply(transpose(a), grad_output)
        return List[Tensor](grad_a, grad_b)
```

## 7.4 Custom Function Framework

Not every operation belongs in the core registry. Consider solving `x² = 2` for `x` with a numerical bisection search rather than `sqrt` — a stand-in, at small scale, for the bond-pricing bisection Part 7 actually differentiates through. Bisection between `a=1` and `b=2`: midpoint `1.5² = 2.25` (too big, move `b` to `1.5`); midpoint `1.25² = 1.5625` (too small, move `a` to `1.25`); midpoint `1.375² = 1.890625`; continuing this halves the bracket every step and converges toward `x ≈ 1.41421` (`√2`). Differentiating *through* dozens of these halving steps would be wasteful and numerically noisy — but there's a shortcut. Treat the solver as defining `x` implicitly by `f(x, c) = x² - c = 0`, where `c` is the constant being solved against (`c=2` here). The **implicit function theorem** says:

```
dx/dc = -(∂f/∂c) / (∂f/∂x) = -(-1) / (2x) = 1 / (2x)
```

At the converged solution `x ≈ 1.41421`, that's `1 / 2.82842 ≈ 0.35355` — which you can check directly: `√2.001 ≈ 1.414568`, and `(1.414568 - 1.41421) / 0.001 ≈ 0.358`, close to `0.35355` (the small gap is finite-difference approximation error, the same kind [Chapter 12.4](../part6/02-advanced-features.md#124-debugging-and-profiling-tools)'s `gradient_check` tolerates). The framework captures this as one opaque node with a closed-form backward, instead of an unrolled, differentiated bisection loop:

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

This is the exact pattern [Chapter 12.1](../part6/02-advanced-features.md#121-custom-autograd-functions) reuses for the z-spread bisection in Part 7: the forward pass runs the numerical solver as ordinary control flow, and one hand-derived `backward_fn` plugs the whole iterative procedure into the graph as a single differentiable op.

Chapter 8 now has everything it needs to finish the running example: seed `w.grad = 1.0`, walk the reverse order `[add, mul]` from Chapter 6.4, call `chain_rule_step` at each stop using the exact rules derived above, and accumulate `x`'s two contributions into a final `x.grad = 5.0`.
