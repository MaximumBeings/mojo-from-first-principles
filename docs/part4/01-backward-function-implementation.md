# Chapter 7: Backward Function Implementation

## 7.1 Chain Rule Implementation

Every backward function in this framework is one application of the chain rule: given the gradient of the loss with respect to a node's *output* (`grad_output`, handed down from whatever consumed it), compute the gradient with respect to each of the node's *inputs*, and hand those further down the graph.

```mojo
fn chain_rule_step(
    op_name: String,
    grad_output: Tensor,
    inputs: List[Tensor],
    output: Tensor,
) -> List[Tensor]:
    """Dispatch to the registered backward for this op, then
    the caller adds each result into the corresponding input's
    accumulated .grad (Section 8.2)."""
    var op = registry.get(op_name)
    return op.backward(grad_output, inputs, output)
```

Nothing here is specific to any one operation — the chain rule is just "look up how this op's output changes with respect to its inputs, multiply by how much the final loss changes with respect to this output." Sections 7.2–7.4 are the catalog of those per-op local derivatives.

## 7.2 Element-wise Operation Gradients

Straight from the local-derivative rules already stated informally in [Chapter 3](../part2/01-element-wise-operations.md):

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

struct ExpOp(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return elementwise_exp(inputs[0])
    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # d(e^x)/dx = e^x = output -- reuse the cached forward result
        return List[Tensor](elementwise_mul(grad_output, output))
```

`ExpOp` is worth lingering on: its backward needs `output`, not `inputs[0]`, which is exactly why `GraphNode` in Chapter 6 stores the forward output alongside the inputs — recomputing `exp(x)` during backward would work but wastes a kernel launch for a value the forward pass already has sitting in memory.

## 7.3 Matrix Operation Gradients

Matrix multiplication's backward is the one gradient rule in this book that isn't a direct restatement of a forward formula — it's derived from the contraction rule in [Chapter 4](../part2/02-matrix-operations.md#41-matrix-multiplication). For `C = A @ B`:

```
dL/dA = dL/dC @ Bᵀ
dL/dB = Aᵀ @ dL/dC
```

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

The shapes are worth checking by hand once: if `A` is `[m, k]` and `B` is `[k, n]`, then `C` and `grad_output` are `[m, n]`. `grad_output @ Bᵀ` is `[m, n] @ [n, k] = [m, k]` — matching `A`'s shape, as any correct gradient must. This shape-matching property is the cheapest sanity check for any new backward rule added to the registry: if the gradient's shape doesn't match its input's shape, the rule is wrong before a single numerical test runs.

## 7.4 Custom Function Framework

Not every operation belongs in the core registry — Part 7's bond-pricing bisection solver, for instance, is not differentiable in the naive sense (it's an iterative root-find). The framework exposes a `CustomFunction` escape hatch for exactly this: the author supplies forward and backward directly, and the graph treats it as an opaque node like any other.

```mojo
struct CustomFunction(Differentiable):
    var forward_fn: fn(List[Tensor]) -> Tensor
    var backward_fn: fn(Tensor, List[Tensor], Tensor) -> List[Tensor]

    fn forward(self, inputs: List[Tensor]) -> Tensor:
        return self.forward_fn(inputs)

    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        return self.backward_fn(grad_output, inputs, output)
```

This is the pattern [Chapter 12](../part6/02-advanced-features.md#121-custom-autograd-functions) uses for the z-spread bisection in Part 7: the forward pass runs the numerical solver as ordinary Mojo control flow (no graph node needed inside the loop), and one hand-derived `backward_fn` — the implicit-function-theorem gradient of the solved spread with respect to the market price — plugs the whole iterative procedure into the graph as a single differentiable op.

Chapter 8 assembles these per-op backward rules into the actual reverse-mode traversal: walking the topologically-sorted graph from Chapter 6, calling `chain_rule_step` at each node, and accumulating the results into every leaf tensor's `.grad`.
