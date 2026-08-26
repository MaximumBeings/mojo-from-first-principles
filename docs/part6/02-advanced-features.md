# Chapter 12: Advanced Features

## 12.1 Custom Autograd Functions

The `CustomFunction` escape hatch introduced in [Chapter 7.4](../part4/01-backward-function-implementation.md#74-custom-function-framework) exists for operations whose backward pass isn't a mechanical composition of registered ops. The z-spread bisection solver used in [Part 7](../part7/01-quantitative-finance-examples.md#132-credit-spread-and-risk-analytics) is the book's running example: the forward pass runs dozens of iterations of bisection, none of which the graph needs to record individually, and the backward pass uses the implicit function theorem rather than differentiating through the loop.

```mojo
struct ZSpreadSolve(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        var market_price = inputs[0]
        var spread = bisection_method(-0.1, 0.1, TOLERANCE)   # ordinary control flow, not graph nodes
        return spread

    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        # Implicit function theorem: if f(price, spread) = 0 defines
        # spread(price), then d(spread)/d(price) = -(df/d_price) / (df/d_spread).
        # df/d_spread is the bond's (negative) price sensitivity -- its DV01 --
        # which is already computed as a byproduct of pricing at the solved spread.
        var df_dspread = bond_price_derivative_wrt_spread(output)
        var df_dprice = Float32(-1.0)   # price enters the objective linearly
        var d_spread_d_price = -df_dprice / df_dspread
        return List[Tensor](elementwise_mul(grad_output, d_spread_d_price))
```

This pattern generalizes beyond finance: any iterative solver — Newton's method for an implied volatility, a fixed-point iteration for an equilibrium — plugs into the same graph the same way, as one opaque node with a closed-form backward instead of an unrolled, differentiated loop.

## 12.2 Higher-order Derivatives

Chapter 8's `backward()` populates `.grad` tensors, but those `.grad` tensors are themselves ordinary `Tensor`s — nothing stops one from calling `backward()` again with a gradient tensor as the new "loss," producing a second derivative. This is exactly how the framework computes a Hessian-vector product without materializing the full Hessian:

```mojo
fn hessian_vector_product(mut graph: ComputationGraph, loss: Tensor, params: List[Tensor], v: Tensor) -> Tensor:
    """Computes H @ v where H = d^2(loss)/d(params)^2, without forming H."""
    backward(graph, loss)                       # first backward: params[0].grad = dL/dparams
    var grad_dot_v = elementwise_mul(params[0].grad, v)
    var scalar = tensor_sum(grad_dot_v, grad_dot_v.shape.size())  # Chapter 5

    zero_grad(params)
    backward(graph, scalar)                      # second backward through the *first* backward's graph
    return params[0].grad                         # this is H @ v
```

The prerequisite this relies on is that `backward()` itself is built entirely from the same differentiable ops (`elementwise_mul`, `matrix_multiply`, `tensor_sum`) as the forward pass — so a graph built by differentiating those ops is itself differentiable. Second-order information like this backs curvature-aware optimizers and the convexity calculations Part 7 uses for bond portfolios (duration is the first derivative of price with respect to yield; convexity is the second).

## 12.3 Model Serialization

A trained network is just a set of `Matrix` weight buffers; serialization reuses the raw memory interface from [Section 1.3.3](../part1/03-tensor-creation-functions.md#part-133-data-importexport) to write each one to disk with a small header describing its shape:

```mojo
fn save_model(path: String, weights: List[Matrix]):
    var f = open(path, "wb")
    f.write_int(len(weights))
    for w in weights:
        f.write_int(w.rows)
        f.write_int(w.cols)
        f.write_bytes(w.data, w.size * sizeof[Float32]())
    f.close()

fn load_model(path: String) -> List[Matrix]:
    var f = open(path, "rb")
    var count = f.read_int()
    var weights = List[Matrix]()
    for _ in range(count):
        var rows = f.read_int()
        var cols = f.read_int()
        var m = Matrix(rows, cols)
        f.read_bytes(m.data, m.size * sizeof[Float32]())
        weights.append(m)
    f.close()
    return weights
```

Loading strictly validates shapes against the architecture it's about to populate before copying a single byte — a shape mismatch here is a configuration bug, and failing loudly at load time beats corrupting a network with misaligned weights.

## 12.4 Debugging and Profiling Tools

Two classes of bug are unique to autograd frameworks and worth dedicated tooling for: a **wrong gradient** (the math is off but the code runs), and a **numerically unstable gradient** (correct in exact arithmetic, `inf`/`nan` in `Float32`).

**Gradient checking** catches the first by comparing the analytic gradient from `backward()` against a finite-difference approximation — expensive, so it's a debugging tool, not something run every training step:

```mojo
fn gradient_check(f: fn(Tensor) -> Tensor, x: Tensor, analytic_grad: Tensor, epsilon: Float32 = 1e-4) -> Float32:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps) ~= f'(x)."""
    var max_relative_error: Float32 = 0.0
    for i in range(x.shape.size()):
        var x_plus = x.clone(); x_plus.data[i] += epsilon
        var x_minus = x.clone(); x_minus.data[i] -= epsilon
        var numeric_grad = (tensor_sum_scalar(f(x_plus)) - tensor_sum_scalar(f(x_minus))) / (2.0 * epsilon)
        var analytic = analytic_grad.data[i]
        var rel_error = abs(numeric_grad - analytic) / max(abs(numeric_grad) + abs(analytic), 1e-8)
        max_relative_error = max(max_relative_error, rel_error)
    return max_relative_error   # should be < ~1e-4 for a correct backward rule
```

Every backward rule added to the registry in Chapter 7 was checked this way before being trusted — `MatMulOp.backward`'s transpose-based formula, `ExpOp.backward`'s output-reuse, and `ZSpreadSolve.backward`'s implicit-function-theorem rule above all passed a `gradient_check` against their forward function before appearing in this book.

**NaN/Inf detection** catches the second class by instrumenting `accumulate_gradient` (Chapter 8.2) to fail fast rather than let a corrupted gradient silently propagate through every remaining parameter:

```mojo
fn check_gradient_health(grad: Tensor, node_name: String):
    for i in range(grad.shape.size()):
        var v = grad.data[i]
        debug_assert(v == v, "NaN gradient at " + node_name)          # NaN != NaN
        debug_assert(abs(v) < 1e10, "Exploding gradient at " + node_name)
```

Run in debug builds during development and compiled out entirely in release builds (Mojo's `debug_assert` is a no-op when assertions are disabled), this pinpoints the *exact* op in the graph where a gradient went bad — the difference between a training run that diverges after 500 silent steps and one that fails immediately with a message naming the offending layer.

Parts 1 through 6 form a complete, general-purpose autograd and neural-network framework. Part 7 is the test that matters most for this book's stated purpose: pointing all of it at quantitative finance, where a silently wrong gradient is not a lower accuracy score but a mispriced position.
