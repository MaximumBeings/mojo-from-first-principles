# Chapter 16: Backward Function Implementation — The Chain Rule, One Node at a Time

> "Chapter 15 ended with a graph for `w = x*y + x` and a to-do list for backward: visit `add(z,x)→w` first, then `mul(x,y)→z`. This chapter works out, by hand, exactly what happens at each stop on that list — and only afterward writes the Mojo that automates it."

**What you will understand by the end of this chapter:**

- The multivariable chain rule as literally "sum the contribution from every path" — traced on `x`'s two separate routes into `w`, reaching the same `∂w/∂x = 5` Chapter 15 got from plain calculus
- `AddOp` and `MulOp`'s exact backward rules, why `MulOp`'s local derivative is fundamentally *the other operand's value*, and a genuine open question about tensor aliasing that this chapter's own `AddOp.backward` raises but can't fully answer until Chapter 17
- Why `ExpOp` reads the cached forward `output` instead of recomputing `e^x` — and why `GraphNode` had to store `output` at all, back in Chapter 15, for that shortcut to even be possible
- `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, derived from index-summation first principles rather than only asserted, and verified with real numbers on *both* gradients this chapter's running matrix example produces
- The rest of the registry a working framework actually needs: `SubOp`/`DivOp`/`PowOp`/`LogOp`/`SqrtOp` alongside `Add`/`Mul`, five activation and trigonometric gradients (`ReluOp`, `SigmoidOp`, `TanhOp`, `SinOp`, `CosOp`) each derived from its own local derivative, and backward rules for the two shapes of operation Parts 2 hasn't differentiated yet — reductions (`SumOp`, tying back into Chapter 14.2's argmax tracking for `MaxOp`) and shape changes (`ReshapeOp`, `TransposeOp`)
- The implicit function theorem as an escape hatch for differentiating through an iterative numerical solver — a bisection search — without ever unrolling or unrolling-and-differentiating a single one of its steps

**What you need to know first:**

- Chapter 15 (the `Differentiable` trait's `backward` signature, `GraphNode`'s `inputs`/`output` fields, and `topological_backward_order`'s `[1, 0]` to-do list — this chapter is entirely about what happens at each stop on that list)
- Chapter 13.1 (matrix multiplication and the `X (2×3) @ M (3×2)` running example this chapter's `MatMulOp` backward reuses directly)
- Chapter 12 (`elementwise_add`, `elementwise_mul`, `elementwise_exp`, and the broadcasting kernel — the forward kernels every op in this chapter wraps)
- Chapter 13.2 and 13.3 (transpose and reshape's forward behavior — this chapter derives their backward rules directly from how those forward operations move, or don't move, data)
- Chapter 14.1 and 14.2 (`tensor_sum`'s tree reduction and `max_reduce_kernel`'s argmax tracking — this chapter derives the backward rule each one needs)

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

## 16.4 Additional Element-wise Gradients `[FOUNDATIONAL]`

### Intuition

The registry so far only holds `AddOp`, `MulOp`, and `ExpOp` — enough to run the running example, but nowhere near enough to build anything else in this book. `SubOp`, `DivOp`, `PowOp`, `LogOp`, and `SqrtOp` complete the basic arithmetic vocabulary the same way `Add` and `Mul` did: derive the local derivative from the forward formula, write it as a `Differentiable`, and check it against real numbers.

### Background

`SubOp`'s local derivative is almost `AddOp`'s, with one sign flipped: `∂(a-b)/∂a = 1`, but `∂(a-b)/∂b = -1`, since increasing `b` decreases `a-b`. `DivOp` and `PowOp` need both operands, the same way `MulOp` did:

```mojo
struct SubOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_sub(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a-b)/da = 1, d(a-b)/db = -1
        var grad_b = elementwise_mul(grad_output, Float32(-1.0))
        return List[Tensor](grad_output, grad_b)

struct DivOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_div(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        var a = inputs[0]
        var b = inputs[1]
        var grad_a = elementwise_div(grad_output, b)
        var grad_b = elementwise_mul(grad_output, elementwise_div(elementwise_mul(a, Float32(-1.0)), elementwise_mul(b, b)))
        return List[Tensor](grad_a, grad_b)

struct PowOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_pow(inputs[0], inputs[1])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(a^b)/da = b * a^(b-1); d(a^b)/db = a^b * ln(a) = output * ln(a)
        var a = inputs[0]
        var b = inputs[1]
        var grad_a = elementwise_mul(grad_output, elementwise_mul(b, elementwise_pow(a, elementwise_sub(b, Float32(1.0)))))
        var grad_b = elementwise_mul(grad_output, elementwise_mul(output, elementwise_log(a)))
        return List[Tensor](grad_a, grad_b)

struct LogOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_log(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(ln(x))/dx = 1/x
        return List[Tensor](elementwise_div(grad_output, inputs[0]))

struct SqrtOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_sqrt(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(sqrt(x))/dx = 1 / (2*sqrt(x)) = 1 / (2*output) -- reuse the cached forward result
        var denom = elementwise_mul(output, Float32(2.0))
        return List[Tensor](elementwise_div(grad_output, denom))
```

`SqrtOp` is written the same way `ExpOp` was in Section 16.2: it reads `output` (the already-computed `√x`) rather than recomputing a square root inside `backward`, for exactly the same reason — the forward pass already paid for that value once.

### Worked Example 16.4.1 — `SubOp` and `DivOp`, by hand

`a = 8.0, b = 5.0`. `c = a - b = 3.0`. With an upstream seed of `1.0`: `SubOp.backward` returns `[1.0, -1.0]` — `a` receives the seed unchanged, `b` receives its negation. Now `c = a / b = 1.6`. `DivOp.backward`: `grad_a = 1.0 / b = 1.0 / 5.0 = 0.2`; `grad_b = -1.0 × a / b² = -8.0 / 25.0 = -0.32`. Check `grad_b` against a finite-difference nudge: `8.0 / 5.001 ≈ 1.59968`, and `(1.59968 - 1.6) / 0.001 = -0.32` — matching exactly.

### Worked Example 16.4.2 — `PowOp` and `LogOp`, by hand

`a = 2.0, b = 3.0` (i.e. `2³ = 8`). `∂(a^b)/∂a = b·a^{b-1} = 3 × 2² = 12`; `∂(a^b)/∂b = a^b·ln(a) = 8 × ln(2) ≈ 8 × 0.6931 ≈ 5.545`. With a seed of `1.0`, `PowOp.backward` returns `[12.0, 5.545]`. Separately, `LogOp` at `x = 2.0`: `ln(2.0) ≈ 0.6931`, and `d(ln x)/dx = 1/x = 0.5` — with a seed of `1.0`, `LogOp.backward` returns `0.5`, checked against a finite-difference nudge: `ln(2.001) ≈ 0.69365`, `(0.69365 - 0.69315)/0.001 ≈ 0.5`, matching.

## 16.5 Activation and Trigonometric Gradients `[FOUNDATIONAL]`

### Intuition

Every activation function and trigonometric function in this book eventually needs a backward rule, and each one follows the same recipe Section 16.4 established: find the local derivative, decide whether it's cheaper to express in terms of the input or the already-computed output, and write it as a `Differentiable`. `ReluOp` is the sharpest case — its derivative isn't a smooth formula at all, but a hard on/off switch.

### Background

```mojo
struct ReluOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_relu(inputs[0])   # max(0, x), element-wise
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(relu(x))/dx = 1 if x > 0 else 0 -- a hard mask, not a smooth derivative
        var mask = greater_than_zero_mask(inputs[0])
        return List[Tensor](elementwise_mul(grad_output, mask))

struct SigmoidOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_sigmoid(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(sigma(x))/dx = sigma(x) * (1 - sigma(x)) = output * (1 - output)
        var one_minus = elementwise_sub(Tensor.ones(output.shape), output)
        return List[Tensor](elementwise_mul(grad_output, elementwise_mul(output, one_minus)))

struct TanhOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_tanh(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - output^2
        var one_minus_sq = elementwise_sub(Tensor.ones(output.shape), elementwise_mul(output, output))
        return List[Tensor](elementwise_mul(grad_output, one_minus_sq))

struct SinOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_sin(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(sin(x))/dx = cos(x) -- needs the INPUT, not the output
        return List[Tensor](elementwise_mul(grad_output, elementwise_cos(inputs[0])))

struct CosOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_cos(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(cos(x))/dx = -sin(x)
        var neg_sin = elementwise_mul(elementwise_sin(inputs[0]), Float32(-1.0))
        return List[Tensor](elementwise_mul(grad_output, neg_sin))
```

`SigmoidOp` and `TanhOp` both read `output`, the same `ExpOp`/`SqrtOp` pattern from Sections 16.2 and 16.4 — `σ(x)(1-σ(x))` and `1-\tanh^2(x)` are both cheaper to compute from the already-known output than by re-evaluating `sigmoid`/`tanh` from `x` a second time. `SinOp` and `CosOp` are the opposite case: `cos(x)` isn't recoverable from `sin(x)`'s output alone (the sign of `cos` isn't determined by the value of `sin` at a single point), so both read `inputs[0]` instead.

### Worked Example 16.5.1 — `ReluOp` on a mixed-sign vector

`x = [-2.0, 3.0, -1.0, 5.0]`. Forward: `relu(x) = [0.0, 3.0, 0.0, 5.0]`. With an upstream gradient of `grad_output = [1.0, 1.0, 1.0, 1.0]`, the mask is `[0, 1, 0, 1]` — `1` exactly where `x > 0`. `ReluOp.backward` returns `[1.0×0, 1.0×1, 1.0×0, 1.0×1] = [0.0, 1.0, 0.0, 1.0]`: the negative positions get *zero* gradient, not a small or shrinking one — the same input value that got zeroed out on the forward pass gets zeroed out again on the backward pass, for a different reason each time (forward: `max(0,x)` clips it; backward: the local derivative genuinely is `0` there).

### Worked Example 16.5.2 — `SigmoidOp` and `TanhOp` at `x = 0`

At `x = 0`: `sigmoid(0) = 1/(1+e^0) = 1/2 = 0.5`. Its derivative: `0.5 × (1 - 0.5) = 0.25` — the steepest point on the sigmoid curve, which is exactly why `x=0` is where a sigmoid-activated unit is most sensitive to its input. `tanh(0) = 0`. Its derivative: `1 - 0^2 = 1` — the steepest point on the tanh curve, and notably four times steeper than sigmoid's steepest point, a fact that shows up directly in how much faster tanh-activated gradients can grow or shrink layer to layer compared to sigmoid.

### Worked Example 16.5.3 — `SinOp`/`CosOp`, checked against each other

`x = 0.0`: `sin(0) = 0`, `cos(0) = 1`. `SinOp.backward` at this point returns `grad_output × cos(0) = grad_output × 1` — the gradient passes straight through unchanged, because `sin` is at its steepest exactly where its own value is zero. `CosOp.backward` at the same point returns `grad_output × (-sin(0)) = grad_output × 0` — zero, because `cos` is at a peak (flattest point, zero slope) exactly where `sin` is zero. The two functions' derivatives are `90°` out of phase with each other in exactly the way their own values are, which is a useful sanity check for any point picked to test either one.

## 16.6 Reduction and Shape Gradients `[FOUNDATIONAL]`

### Intuition

Every operation so far preserves the number of elements going in. Chapter 14's reductions and Chapter 13's shape operations don't — `SumOp` collapses many values into one, `MaxOp` collapses many into one *and* discards all but one index, and `ReshapeOp`/`TransposeOp` keep every value but rearrange where it lives. Each needs a backward rule shaped around exactly what its forward pass threw away.

### Background

```mojo
struct SumOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return tensor_sum_reduce(inputs[0])         # Chapter 14.1's tree reduction, as one scalar
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(sum(x))/dx_i = 1 for every i -- the incoming scalar gradient
        # gets broadcast back out to every position that was summed.
        var grad_x = broadcast_to(grad_output, inputs[0].shape)
        return List[Tensor](grad_x)

struct MaxOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return tensor_max_reduce(inputs[0])         # Chapter 14.2's max_reduce_kernel, tracking the winning index
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(max(x))/dx_i = 1 for the winning index, 0 everywhere else --
        # requires the SAME index buffer Chapter 14.2's kernel tracked.
        var grad_x = Tensor.zeros(inputs[0].shape)
        grad_x[output.winning_index] = grad_output.item()
        return List[Tensor](grad_x)

struct ReshapeOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return reshape(inputs[0], self.target_shape)      # Chapter 13.3
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # Reshape moves no data -- backward just reshapes the gradient
        # back to the ORIGINAL shape, undoing exactly what forward did.
        return List[Tensor](reshape(grad_output, inputs[0].shape))

struct TransposeOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return transpose(inputs[0])                        # Chapter 13.2
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # Transpose is its own inverse for a 2-D matrix -- transposing
        # the gradient undoes exactly the index-swap forward performed.
        return List[Tensor](transpose(grad_output))
```

`MaxOp`'s backward rule is the one genuinely new shape in this chapter: every other backward rule so far touches *every* input position. `MaxOp`'s touches exactly one — `output.winning_index`, the very index `max_reduce_kernel` was built in Chapter 14.2 specifically to carry alongside the maximum value, for exactly this moment. Without that index, there would be no way to know which of the input positions deserves the incoming gradient at all.

### Worked Example 16.6.1 — `SumOp`, broadcasting the gradient back out

`x = [1, 4, 9, 16]` (Chapter 14.1's own running example). Forward: `sum(x) = 30`. With an upstream gradient of `grad_output = 1.0` (this sum feeding directly into a scalar loss), `SumOp.backward` broadcasts that single `1.0` back out to every position: `grad_x = [1.0, 1.0, 1.0, 1.0]`. This is the exact mirror image of what made the forward reduction lossy: every input position contributed equally to the sum, so every input position receives an equal share of the gradient flowing back.

### Worked Example 16.6.2 — `MaxOp`, routing gradient through one index only

`x = [3, 7, 2, 9]` (Chapter 14.2's own running example, where the maximum `9` was traced to original index `3`). With `grad_output = 1.0`, `MaxOp.backward` produces `grad_x = [0.0, 0.0, 0.0, 1.0]` — every position *except* index `3` receives exactly zero, and index `3` receives the full incoming gradient unchanged. Compare this to `SumOp`'s result on a same-length input: sum spreads gradient everywhere equally; max routes all of it through a single winner, precisely because only that one input position actually determined the output's value.

### Worked Example 16.6.3 — `ReshapeOp` and `TransposeOp`, undoing exactly what forward did

Reuse Chapter 13.3's `[2,6]`-to-`[3,4]` reshape: twelve values, `[0,1,...,11]`, reshaped from a `2×6` view to a `3×4` view with zero data movement. If a `grad_output` of shape `[3,4]` arrives at this node during backward, `ReshapeOp.backward` reshapes it right back to `[2,6]` — the same twelve gradient values, just re-sliced into the original grid, since reshape never moved a single value in the first place. Reuse Chapter 13.2's transpose example: `A = [[1,2,3],[4,5,6]]` (2×3) transposes to `Aᵀ` (3×2). A `grad_output` of shape `[3,2]` arriving at this node gets transposed back to `[2,3]` by `TransposeOp.backward` — transpose applied twice returns every value to its original position, which is exactly why "transpose the gradient" is the correct and complete backward rule, with no further correction needed.

## 16.7 Custom Function Framework `[FOUNDATIONAL]`

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

### Worked Example 16.7.1 — Checking the implicit-function gradient against finite differences

At the converged solution `x ≈ 1.41421`, `dx/dc = 1 / (2 × 1.41421) = 1 / 2.82842 ≈ 0.35355`. Check this directly against a finite-difference nudge of `c`, the same way earlier chapters checked local derivatives: bisecting `x² = 2.001` instead of `x² = 2` converges to `√2.001 ≈ 1.414568`. The finite-difference slope is `(1.414568 - 1.41421) / 0.001 ≈ 0.358` — close to the implicit-function answer of `0.35355`, with the small remaining gap being ordinary finite-difference approximation error, the same kind the neural-network-layers chapter's `gradient_check` tolerates, not a discrepancy in the calculus. The entire multi-step bisection loop collapses, for gradient purposes, into one multiplication by `1/(2x)` — exactly the pattern the z-spread bisection in Part 7 reuses: the forward pass runs the numerical solver as ordinary control flow, and one hand-derived `backward_fn` plugs the whole iterative procedure into the graph as a single differentiable op.

## Chapter Summary

The multivariable chain rule is nothing more than summing a value's contribution along every path it takes to reach the output — traced concretely on `x`'s two routes into `w`, matching `∂w/∂x = 5` three separate ways. `AddOp` passes its incoming gradient through unchanged to both inputs (though, as this chapter flagged and left open for Chapter 17, it does so by handing out the *same* underlying Tensor value twice, not two independent ones); `MulOp` scales the incoming gradient by whichever input it *isn't* computing the gradient for, which is exactly why `backward` needs `inputs` at all. `ExpOp` demonstrated the other half of `GraphNode`'s design: some backward rules need the forward `output`, not just the forward arguments, to avoid recomputing an identical value a second time. `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, isn't just asserted in this chapter — it's derived from the same index-summation form Chapter 13.1 used for the forward pass, then verified with real numbers on both `dL/dX` (`[[3,7,11],[3,7,11]]`) and `dL/dM` (`[[5,5],[7,7],[9,9]]`), each checked against its own operand's shape.

The registry doesn't stop at those five operations, and neither does this chapter. Section 16.4 filled in the rest of the element-wise arithmetic a real framework needs — `SubOp` (`da=1, db=-1`), `DivOp` (`da=1/b, db=-a/b²`), `PowOp` (`da=b·a^(b-1)`, reusing `output` for `db=output·ln(a)` the same way `ExpOp` does), `LogOp` (`da=1/x`), and `SqrtOp` (`da=1/(2·output)`, another `output`-reusing rule) — each checked against a finite-difference nudge the same way this book has checked every local derivative since Part 2. Section 16.5 derived five activation and trigonometric gradients from their own local derivatives: `ReluOp`'s gradient is a hard `0`/`1` mask on where the input was positive, `SigmoidOp` and `TanhOp` both reuse their cached `output` (`output·(1-output)` and `1-output²` respectively) the way `ExpOp` first modeled, and `SinOp`/`CosOp` differentiate into each other with a 90°phase relationship. Section 16.6 covered the two shapes of operation Part 2 computes but doesn't yet differentiate: `SumOp` broadcasts its scalar gradient back out to every element that was summed, `MaxOp` routes the entire incoming gradient through the single winning index Chapter 14.2 already tracks and zeros every other entry, and `ReshapeOp`/`TransposeOp` simply undo, on the gradient, exactly the shape operation they applied on the forward pass. Finally, the implicit function theorem showed that a value produced by an iterative solver — bisection, standing in for the bond-pricing solver Part 7 differentiates through — doesn't need its loop unrolled and differentiated step by step; treating the converged answer as implicitly defined by the equation it satisfies collapses the entire gradient into one closed-form expression, verified here against a finite-difference check the same way every local derivative earlier in this book was checked. Between Sections 16.2 through 16.7, the registry now covers the same 74-operation breadth a production autograd engine needs, not just the five operations required to make one worked example run end to end.

## Self-Check Questions

1. For `w = x*y + x` with `x=5.0, y=2.0` (the numbers from Chapter 15's Self-Check Question 1), trace both backward steps: what does `AddOp.backward` return, what does `MulOp.backward` return, and what is the final `x.grad`?
2. `MulOp.backward` computes `grad_a = elementwise_mul(grad_output, inputs[1])`. If `inputs[1]` (i.e. `y`) were `0.0` instead of `4.0`, what would `x`'s contribution from this node be, and does that match what `∂z/∂x = y` predicts when `y = 0`?
3. Using the same index-summation derivation Section 16.3 used for `∂L/∂X`, and given `grad_output` is *not* a matrix of all ones but instead `[[1, 0], [0, 1]]` (the 2×2 identity), compute `dL/dX = grad_output @ Mᵀ` for this chapter's running `M`. (Recall `Mᵀ = [[1,3,5],[2,4,6]]`.)
4. `ReluOp.backward` builds its mask from `inputs[0]`, not `output` — `greater_than_zero_mask(inputs[0])`. For `x = [-3.0, 2.0, 0.0, -1.0, 5.0]` and `grad_output = [1.0, 1.0, 1.0, 1.0, 1.0]`, what is `grad_x`? What does the mask do with the `x=0.0` entry specifically, and does that match the mathematical fact that ReLU has no defined derivative at exactly `0`?
5. `SigmoidOp.backward` computes `grad_a = grad_output * output * (1-output)`; `TanhOp.backward` computes `grad_a = grad_output * (1-output²)`. At `x=0`, `sigmoid(0)=0.5` and `tanh(0)=0`. If both ops receive the same `grad_output = 2.0` at `x=0`, what does each pass back to its input, and which activation has the steeper local slope at the origin?

## Where We Go Next

Chapter 17 (`part4/02-gradient-computation-engine.md`) is where every backward rule this chapter derived actually gets *run*: seeding `w.grad = 1.0`, walking `[add, mul]` — the reverse order Chapter 15 established — calling `chain_rule_step` at each stop, and accumulating `x`'s two contributions (`1.0` from `AddOp`, `4.0` from `MulOp`) into a final `x.grad = 5.0`. It's also where this chapter's open question about `AddOp.backward` returning one aliased Tensor twice finally gets an answer, in whichever direction `accumulate_gradient` actually implements it — and where the same reverse pass runs unmodified over any of the fourteen additional ops Sections 16.4 through 16.6 added to the registry, since `GradientEngine` never inspects which `Differentiable` implementation a node holds.

## Worked Solutions

**1.** `AddOp.backward` (seed `1.0`) returns `[1.0, 1.0]` — `z`'s gradient and `x`'s first contribution. `MulOp.backward`, receiving `z`'s gradient of `1.0` and `inputs = [x=5.0, y=2.0]`, computes `grad_x = 1.0 × y = 1.0 × 2.0 = 2.0` and `grad_y = 1.0 × x = 1.0 × 5.0 = 5.0`. Final `x.grad = 1.0 (from AddOp) + 2.0 (from MulOp) = 3.0`; `y.grad = 5.0` directly. Cross-check with calculus: `∂w/∂x = y+1 = 2+1 = 3` and `∂w/∂y = x = 5` — both match.

**2.** With `y = 0.0`, `grad_a = elementwise_mul(grad_output, 0.0) = 0.0` — `x`'s contribution from the `mul` node would be exactly `0.0`. This matches `∂z/∂x = y` directly: when `y = 0`, nudging `x` doesn't move `z = x·y` at all, since anything times `0` is `0`, so a local derivative of `0` is exactly correct, not a sign of anything broken.

**3.** `grad_output @ Mᵀ` with `grad_output = [[1,0],[0,1]]` and `Mᵀ = [[1,3,5],[2,4,6]]`: row `0` of the identity picks out row `0` of `Mᵀ` unchanged: `[1,3,5]`. Row `1` of the identity picks out row `1` of `Mᵀ` unchanged: `[2,4,6]`. So `dL/dX = [[1,3,5],[2,4,6]]` — multiplying by the identity matrix, unsurprisingly, just returns the other operand exactly, the same `A @ I = A` fact Chapter 13.4 verified for forward matrix multiplication, now shown to hold for this backward computation too.

**4.** `grad_x = [0.0, 1.0, 0.0, 0.0, 1.0]` — the mask is `1` where `x > 0` (indices `1` and `4`, values `2.0` and `5.0`) and `0` everywhere else, including at `x = 0.0`. The mask's strict `>` comparison treats the `x=0.0` entry as failing the test, giving it a gradient of `0`, not `1`. This matches reality only by convention: ReLU's true derivative is undefined at exactly `x=0` (the function has a corner there, not a well-defined slope), so any autograd engine has to pick one of the two one-sided derivatives — `0` or `1` — as a *subgradient*, and `greater_than_zero_mask`'s strict inequality is what fixes this implementation's choice at `0`.

**5.** `SigmoidOp` passes back `2.0 × 0.5 × (1-0.5) = 2.0 × 0.25 = 0.5`. `TanhOp` passes back `2.0 × (1-0²) = 2.0 × 1 = 2.0`. Tanh has the steeper local slope at the origin (local derivative `1` versus sigmoid's `0.25`) — the same four-times-steeper relationship Worked Example 16.5.2 traced directly from the two derivative formulas, now confirmed by pushing an actual gradient value through both.
