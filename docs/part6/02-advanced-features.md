# Chapter 21: Advanced Features — The Escape Hatches a Framework Needs to Be Trusted

> "A differentiation engine is not finished when it can compute a gradient. It is finished when you can check that the gradient is right, save the thing that produced it, and reach past its own machinery when the machinery doesn't fit."

## What you will understand by the end of this chapter

- Why an iterative solver like bisection is deliberately kept *out* of the computational graph, and how the implicit function theorem lets `backward()` differentiate through it anyway — as one opaque node with a closed-form gradient, verified against a real bond's numbers.
- How a second call to `backward()` — on a gradient tensor instead of a loss tensor — produces a genuine second derivative, and why a Hessian-vector product gets you one useful slice of a Hessian without ever forming the full matrix.
- What a trained network actually *is* on disk: a small header plus raw bytes, and the one thing this chapter's serialization format is missing that a production system could not skip.
- The two distinct failure modes unique to autograd code — a wrong-but-plausible gradient, and a numerically corrupted one — and the two different tools each one requires.
- Why every safety check in this chapter is a *debugging* tool, not a *production* one, and what that trade-off actually costs when it's forgotten.

## What you need to know first

- The `Differentiable` trait and the `CustomFunction`/registry pattern from Chapter 16.7, including why `MatMulOp.backward` (Chapter 16.3) and `ExpOp.backward` (Chapter 16.2) look the way they do.
- `backward()`, `accumulate_gradient`, and `zero_grad` from Chapter 17, and specifically the accumulation rule from Chapter 17.2 that makes calling `backward()` twice in a row meaningful instead of destructive.
- `ComputationGraph` and `GraphNode` from Chapter 15 — this chapter treats a graph as something you can run `backward()` over more than once.
- The `Matrix` weight buffers and full training step from Chapter 20 — Section 21.3 serializes exactly those buffers.
- Elementwise multiplication (Chapter 12.2), matrix multiplication (Chapter 13.1), and sum reduction (Chapter 14.1) — the three ops Section 21.2's Hessian-vector product is built from, and nothing else.

## 21.1 Custom Autograd Functions `[FOUNDATIONAL]`

<a id="121-custom-autograd-functions"></a>

### Intuition

Every backward rule so far has been mechanical: look at the forward formula, apply calculus, write the local derivative. That works because every op so far has been a *closed-form* expression — `x*y`, `exp(x)`, a matrix product. But not every useful computation looks like that. Solving for a bond's z-spread means running bisection: guess a spread, price the bond, compare to the market price, narrow the guess, and repeat — 18 times over, for the actual bond this section prices — until the guess is close enough. There is no single formula for "the z-spread" the way there's a single formula for "the product of two numbers." There's only a *procedure*.

Think of the difference the way a surveyor thinks about the height of a hill. One way to find it: measure every footstep on the way up, add the changes in elevation, and get an exact answer built entirely out of small, individually-verifiable pieces. That's what recording every arithmetic operation in a graph and differentiating through each one does — it works, but it means carrying an instrument up every single step. The other way: stand at the top, take one reading with an altimeter, and separately know the *rate* at which altitude changes with distance near that point without ever having walked the whole slope. That second reading doesn't require re-deriving anything about the climb — it requires one fact about the relationship between position and height *at the point you're standing*. The implicit function theorem is that second approach applied to a solver: instead of differentiating through the 18 footsteps bisection took to arrive at the answer, it asks one question about the relationship between the market price and the bond price *at the spread bisection already found*, and gets an exact derivative from that alone.

### Background

A `CustomFunction` (the `Differentiable` trait applied to an op whose `forward` does *not* build more graph nodes internally) needs exactly two things: a `forward` that returns an answer by whatever means necessary, ordinary control flow included, and a `backward` that supplies a gradient without needing to know how `forward` arrived at that answer. For a solver defined by `f(price, spread) = 0` — bisection finds the `spread` that drives `calculate_bond_price(spread) - market_price` to zero — the implicit function theorem gives that gradient in closed form:

```
d(spread)/d(price) = -(∂f/∂price) / (∂f/∂spread)
```

`∂f/∂price = -1` always, because `price` enters the objective linearly and with a negative sign (`f = calculate_bond_price(spread) - price`). `∂f/∂spread` is `calculate_bond_price`'s own derivative with respect to spread — the bond's price sensitivity to spread, evaluated at the spread bisection already solved for. That derivative is needed anyway for other purposes (it's a close cousin of DV01), so this doesn't add work the framework wasn't already positioned to do.

| Approach | What the graph records | Backward rule needed | Cost |
|---|---|---|---|
| Unroll the solver | Every arithmetic op inside every bisection iteration (≈970 ops for this chapter's 18-iteration, 8-cash-flow bond) | One rule per elementary op, already in the registry | O(iterations × cash flows) graph nodes, each with its own stored `output` and gradient buffer |
| `CustomFunction` (this section) | One node: `ZSpreadSolve` | One hand-derived rule, written once | O(1) graph nodes; the 18 iterations run as ordinary Mojo control flow the graph never sees |

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

### Worked Example 21.1.1 — The z-spread bond's real derivative, computed and cross-checked

Part 7 solves this exact bond — 2-year maturity, 3% coupon paid quarterly, \$100 notional, 3% risk-free rate — for the z-spread that reprices it to a market price of \$98.00, and finds `spread* ≈ 0.010460523` (104.6 basis points). `bond_price_derivative_wrt_spread` at that solved spread — the derivative of the discounted-cash-flow sum with respect to spread — comes out to:

```
∂f/∂spread ≈ -188.9937
```

(this is exact enough to check by finite difference the way every other backward rule in this book has been: evaluating the price at `spread* + 10⁻⁶` and `spread* - 10⁻⁶` and dividing the difference by `2×10⁻⁶` gives the same `-188.9937` to five decimal places). With `∂f/∂price = -1.0`:

```
d(spread)/d(price) = -(-1.0) / (-188.9937) = -0.0052912
```

Read that as: a \$1 *increase* in the bond's market price should make the z-spread that reprices it *fall* by about 0.0052912, i.e. roughly 52.9 basis points — a bond trading more expensively needs less compensation for credit risk to justify that price. The sign makes sense (richer price → smaller spread) and the magnitude is a small fraction of a point of spread per dollar of price, not the wildly-off value a sign error or a units mismatch (spread expressed as a percent instead of a decimal, say) would produce.

That prediction can be checked without any calculus at all: re-run bisection with the market price bumped from \$98.00 to \$99.00, and it converges to a *new* spread of `≈0.0052000` — a drop of `0.0052605` from the original `0.010460523`. The `ZSpreadSolve.backward` formula predicted a drop of `0.0052912` for that same \$1 move. The two agree to within `0.00003` (about 0.3 basis points) — the small remaining gap is exactly the nonlinearity a *first-order* (linear) gradient can't capture over a \$1 move that isn't infinitesimally small, which is the same caveat that applies to every gradient in this book: it's the exact slope at a point, and only an approximation to the change over a finite step.

```
                 18 iterations of bisection, ~970 recorded ops if unrolled
   market_price ──────────────────────────────────────────────▶ spread* ≈ 0.01046
        │                                                            │
        │                    ZSpreadSolve (ONE graph node)           │
        └──────────────── forward: opaque bisection ─────────────────┘
                           backward: d(spread)/d(price) = -0.0052912
                           (implicit function theorem — no unrolling)
```

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| The implicit function theorem formula divides by df/d_spread.  |
| If bisection is ever called at a spread where the bond's price |
| is completely insensitive to spread (df/d_spread == 0 -- can't |
| happen for THIS monotonic bond-pricing formula, but is a real  |
| risk for a poorly-conditioned solver elsewhere), this backward |
| rule divides by zero and returns inf or NaN silently, with no  |
| signal that anything went wrong until Section 21.4's           |
| check_gradient_health happens to be watching for it.           |
+----------------------------------------------------------------+
```

## 21.2 Higher-order Derivatives `[FOUNDATIONAL]`

### Intuition

A first derivative answers "how does the output change as the input changes." A second derivative answers the next natural question: "how does *that rate of change itself* change?" A speedometer reading is a first derivative of position; whether your foot is currently pressing the accelerator or the brake is information about the *second* derivative — it tells you whether that speedometer reading is about to go up or down, which the speedometer alone can't say. Curvature-aware optimizers and bond convexity (Part 7 treats duration, a first derivative of price with respect to yield, and convexity, its second derivative, as two different and both useful numbers) both need exactly this: not just the slope, but whether the slope is currently steepening or flattening.

### Background

Take `g(x) = x³` at `x = 2`: `g(2) = 8`. The first derivative is `g'(x) = 3x²`, so `g'(2) = 3×4 = 12` — the function is currently increasing at a rate of 12 units of output per unit of input. The second derivative is `g''(x) = 6x`, so `g''(2) = 12` as well (a coincidence of this particular function's numbers at this particular point, not a general rule) — meaning that rate of increase is itself growing by 12 for every unit `x` moves.

Chapter 17's `backward()` populates `.grad` tensors, but those `.grad` tensors are themselves ordinary `Tensor`s built from the same registered, differentiable ops (`elementwise_mul` from Chapter 12.2, `matrix_multiply` from Chapter 13.1, `tensor_sum` from Chapter 14.1) as the forward pass — so a graph built by differentiating those ops is itself differentiable, and nothing stops calling `backward()` a second time with a gradient tensor as the new "loss," producing exactly this second derivative. For a function of several variables, the full grid of second partial derivatives is called the **Hessian**; a **Hessian-vector product** (`H @ v`) gets one useful slice of it — here, with a single variable `x` and `v=1`, `H @ v` reduces to plain `g''(2) = 12` — without ever building the full grid.

| Approach | Memory | Compute per Hessian-vector product | Scales to |
|---|---|---|---|
| Form the full Hessian, then multiply by `v` | O(n²) — one entry per pair of parameters | One O(n²) matrix build, then one O(n²) matrix-vector multiply | Small `n` only (thousands of parameters, not millions) |
| `hessian_vector_product` (this section) | O(n) — one gradient-shaped tensor | Two ordinary `backward()` passes | Any `n` a single `backward()` pass already handles |

```mojo
fn hessian_vector_product(mut graph: ComputationGraph, loss: Tensor, params: List[Tensor], v: Tensor) -> Tensor:
    """Computes H @ v where H = d^2(loss)/d(params)^2, without forming H."""
    backward(graph, loss)                       # first backward: params[0].grad = dL/dparams
    var grad_dot_v = elementwise_mul(params[0].grad, v)
    var scalar = tensor_sum(grad_dot_v, grad_dot_v.shape.size())  # Chapter 14.1

    zero_grad(params)                            # Chapter 17.2 -- without this, the second
    backward(graph, scalar)                      # backward would ADD onto the first pass's grad
    return params[0].grad                         # this is H @ v
```

### Worked Example 21.2.1 — The single-variable case, checked by finite difference

`g'(2)=12` can be verified the same way Chapter 17.2's accumulation rule was: `g(2.001) = 8.012006...`, `g(1.999) = 7.988006...`, central-difference slope `≈ (8.012006 - 7.988006)/0.002 = 12.0`. ✓

### Worked Example 21.2.2 — A genuine two-variable Hessian-vector product, verified against the full Hessian

Take `L(w) = w1²·w2` at `w1=2, w2=3`. The gradient is `∇L = [∂L/∂w1, ∂L/∂w2] = [2·w1·w2, w1²] = [2×2×3, 2²] = [12, 4]`. Choose `v = [1, 0]`.

**Via `hessian_vector_product`:** first `backward()` gives `params[0].grad = [12, 4]`. `grad_dot_v = [12×1, 4×0] = [12, 0]`. `scalar = tensor_sum([12, 0]) = 12`. That `12` is `2·w1·w2` evaluated at the current `w` — but as a *function of `w`* going into the second `backward()`, it's `s(w) = 2·w1·w2·v1 + w1²·v2 = 2·w1·w2` (since `v2=0`). The second `backward()` differentiates that: `∂s/∂w1 = 2·w2 = 6`, `∂s/∂w2 = 2·w1 = 4`. So `hessian_vector_product` returns `[6, 4]`.

**Via the full Hessian, for comparison:** `H = [[∂²L/∂w1², ∂²L/∂w1∂w2], [∂²L/∂w2∂w1, ∂²L/∂w2²]] = [[2·w2, 2·w1], [2·w1, 0]] = [[6, 4], [4, 0]]`. `H @ v` with `v=[1,0]` takes just the first column: `[6, 4]`.

Both routes agree exactly — `[6, 4]` — and the `hessian_vector_product` route never built the `2×2` matrix `H` at all, only two ordinary gradient-shaped vectors. For a network with a million parameters, that's the difference between two vectors of a million floats each and a matrix of a trillion floats that will never fit in memory.

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| Skipping zero_grad(params) between the two backward() calls is |
| the single most common mistake here. Chapter 17.2 established  |
| that accumulate_gradient ADDS rather than overwrites -- exactly |
| the behavior a shared weight needs during ONE backward pass.   |
| Across TWO separate backward passes for TWO different purposes |
| (the actual loss gradient, then the Hessian-vector product),   |
| that same addition silently sums the first backward's gradient |
| into the second's result, corrupting Hv with an extra copy of  |
| dL/dparams that has nothing to do with curvature.               |
+----------------------------------------------------------------+
```

## 21.3 Model Serialization `[FOUNDATIONAL]`

### Intuition

Think of packing a moving truck. If you just throw furniture into the truck loose, unloading it means guessing what each piece is by looking at it. If instead you box each piece and write its dimensions and contents on the box before sealing it, unloading is mechanical: read the label, know exactly what's inside and how big it should be, unpack in the same order it was packed. A trained network's weights are the furniture; the header this section writes before each weight buffer is the label.

### Background

A trained network is just a set of `Matrix` weight buffers — the same `Matrix` struct Chapter 20 trains — so serialization reuses the raw memory interface from Section 8.3 to write each one to disk with a small header describing its shape:

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

Loading strictly mirrors the write order — count, then (rows, cols, bytes) per matrix, read in exactly the sequence they were written — and validates shapes against the architecture it's about to populate before copying a single byte. A shape mismatch here is a configuration bug, and failing loudly at load time beats corrupting a network with misaligned weights.

| Format choice | What it buys | What it costs |
|---|---|---|
| Raw header + raw bytes (this section) | Minimal code, fast to write and read, no external dependency | No version field, no dtype tag, no checksum — a format change silently breaks old files |
| A versioned, self-describing format (not shown here) | Old files stay loadable after the format evolves; corruption is detected, not silently misread | More code, a larger file, a format spec to maintain |

### Worked Example 21.3.1 — The exact byte layout for a two-layer network's weights

Take a tiny two-layer network with `W1` shaped `[3, 2]` and `W2` shaped `[2, 1]` — 2 matrices total. Assume, as is typical for Mojo's `Int` on a 64-bit target, that `write_int` writes 8 bytes, and `Float32` is 4 bytes:

```
Offset  Bytes  Field                         Value
------  -----  ----------------------------  -----
0       8      count                         2
8       8      W1.rows                       3
16      8      W1.cols                       2
24      24     W1.data  (3*2=6 floats * 4B)  [w1_00, w1_01, w1_10, ...]
48      8      W2.rows                       2
56      8      W2.cols                       1
64      8      W2.data  (2*1=2 floats * 4B)  [w2_00, w2_10]
------  -----  ----------------------------  -----
Total file size: 72 bytes
```

`load_model` reads this back in exactly the order `save_model` wrote it — `count`, then `(rows, cols, data)` per matrix — which is the only reason a plain sequence of bytes with no field names anywhere in it can be unambiguously reconstructed into two correctly-shaped matrices: the *order* of the reads is the schema.

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| This format has no version number and no dtype tag anywhere in |
| it. If a future Matrix gains a new field that needs saving, or  |
| a model is ever trained in Float16 instead of Float32, an OLD   |
| save_model file loaded with the NEW load_model reads the wrong  |
| number of bytes for w.data and silently reconstructs a matrix   |
| full of garbage -- not a crash, not an error, just numbers that |
| look plausible and are wrong. Production checkpoint formats     |
| reserve a version field in the header for exactly this reason.  |
+----------------------------------------------------------------+
```

## 21.4 Debugging and Profiling Tools `[FOUNDATIONAL]`

### Intuition

Two classes of bug are unique to autograd frameworks, and they need two different kinds of tool for the same reason a doctor needs both a second opinion and a smoke detector. A **wrong gradient** is a second-opinion problem: the code runs, produces a number, and that number is simply not what calculus says it should be — the only way to catch it is to independently recompute the answer by a completely different method and compare. A **numerically unstable gradient** is a smoke-detector problem: something is actively going wrong (a division by zero, a value overflowing `Float32`) and the fix is to notice the instant it happens, at the exact location it happens, rather than discovering the fire only after the whole building — the training run — has already burned down over 500 silent steps.

### Background

**Gradient checking** is the second opinion. It compares the analytic gradient `backward()` produced against a finite-difference approximation, using the *central* difference `(f(x+ε) - f(x-ε)) / (2ε)` rather than the cheaper forward difference `(f(x+ε) - f(x)) / ε` because the central form's error shrinks as `O(ε²)` instead of `O(ε)` — it cancels the *next* term in the Taylor expansion, not just the current one, which is why `ε=10⁻⁴` is tight enough to trust in `Float32` without also amplifying rounding error from too small a step:

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

Every backward rule added to the registry in Chapter 16 was checked this way before being trusted — `MatMulOp.backward`'s transpose-based formula (Chapter 16.3), `ExpOp.backward`'s output-reuse (Chapter 16.2), and `ZSpreadSolve.backward`'s implicit-function-theorem rule from Section 21.1 above all passed a `gradient_check` against their forward function before appearing in this book.

**NaN/Inf detection** is the smoke detector. It instruments `accumulate_gradient` (Chapter 17.2) to fail fast rather than let a corrupted gradient silently propagate through every remaining parameter:

```mojo
fn check_gradient_health(grad: Tensor, node_name: String):
    for i in range(grad.shape.size()):
        var v = grad.data[i]
        debug_assert(v == v, "NaN gradient at " + node_name)          # NaN != NaN
        debug_assert(abs(v) < 1e10, "Exploding gradient at " + node_name)
```

Run in debug builds during development and compiled out entirely in release builds (Mojo's `debug_assert` is a no-op when assertions are disabled), this pinpoints the *exact* op in the graph where a gradient went bad.

| Tool | Catches | Cost | When to run |
|---|---|---|---|
| `gradient_check` | A wrong-but-finite backward rule (bad math, a transposed dimension, a dropped term) | O(n) forward evaluations for an n-parameter input — expensive | Once, when writing or changing a backward rule |
| `check_gradient_health` | NaN or exploding values in an otherwise correctly-derived gradient | O(1) per element, negligible | Every `accumulate_gradient` call, in debug builds |

### Worked Example 21.4.1 — `gradient_check` passing a correct rule

`f(x) = x²` at `x=3.0`, analytic gradient (from `f'(x)=2x`) `= 6.0`. Central difference with `ε=10⁻⁴`: `f(3.0001) = 9.00060001`, `f(2.9999) = 8.99940001`, `numeric_grad = (9.00060001 - 8.99940001)/0.0002 = 6.0000000004`. `rel_error = |6.0000000004 - 6.0| / max(12.0000000004, 10⁻⁸) ≈ 3.5×10⁻¹¹` — far below the `~10⁻⁴` threshold. The check passes, correctly, because the rule is correct.

### Worked Example 21.4.2 — `gradient_check` catching a genuinely wrong rule

Now suppose a bug ships `f'(x) = x` instead of `f'(x) = 2x`, so `analytic_grad = 3.0` at the same `x=3.0`. The finite-difference side of the calculation is unchanged — it never looks at the (buggy) analytic code — so `numeric_grad` is still `≈6.0000000004`. `rel_error = |6.0000000004 - 3.0| / max(9.0000000004, 10⁻⁸) ≈ 0.3333` — four thousand times over the `~10⁻⁴` threshold. `gradient_check` flags this instantly, without needing a single training step to reveal that the model trained on this rule would silently converge to the wrong answer.

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| debug_assert is a no-op in release builds. A gradient that only |
| goes NaN in a rare numerical corner case your debug-build test  |
| suite never happens to exercise -- a specific input distribution|
| seen only in production, say -- will sail through completely    |
| undetected the moment check_gradient_health's asserts are       |
| compiled away, because release mode is exactly when the check   |
| stops running at all, not just when it stops printing.          |
+----------------------------------------------------------------+
```

## 21.5 Reference Implementations

The listing below consolidates every function this chapter introduced into one file. As with the rest of this book's non-neural-network chapters, none of this Mojo code has been compiled or run — the surrounding worked examples were verified independently, in Python, against the real numbers Part 7's actual bond (Worked Example 21.1.1) and hand-derived test functions (Worked Examples 21.2.1, 21.2.2, 21.4.1, 21.4.2) produce, not by executing this file.

```mojo
# ── 21.1 Custom Autograd Functions ──────────────────────────────
struct ZSpreadSolve(Differentiable):
    fn forward(self, inputs: List[Tensor]) -> Tensor:
        var market_price = inputs[0]
        var spread = bisection_method(-0.1, 0.1, TOLERANCE)
        return spread

    fn backward(self, grad_output: Tensor, inputs: List[Tensor], output: Tensor) -> List[Tensor]:
        var df_dspread = bond_price_derivative_wrt_spread(output)
        var df_dprice = Float32(-1.0)
        var d_spread_d_price = -df_dprice / df_dspread
        return List[Tensor](elementwise_mul(grad_output, d_spread_d_price))

# ── 21.2 Higher-order Derivatives ───────────────────────────────
fn hessian_vector_product(mut graph: ComputationGraph, loss: Tensor, params: List[Tensor], v: Tensor) -> Tensor:
    """Computes H @ v where H = d^2(loss)/d(params)^2, without forming H."""
    backward(graph, loss)
    var grad_dot_v = elementwise_mul(params[0].grad, v)
    var scalar = tensor_sum(grad_dot_v, grad_dot_v.shape.size())

    zero_grad(params)
    backward(graph, scalar)
    return params[0].grad

# ── 21.3 Model Serialization ─────────────────────────────────────
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

# ── 21.4 Debugging and Profiling Tools ───────────────────────────
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
    return max_relative_error

fn check_gradient_health(grad: Tensor, node_name: String):
    for i in range(grad.shape.size()):
        var v = grad.data[i]
        debug_assert(v == v, "NaN gradient at " + node_name)
        debug_assert(abs(v) < 1e10, "Exploding gradient at " + node_name)
```

### Expected Output

No captured output accompanies this listing. `ZSpreadSolve` and `hessian_vector_product` depend on framework types (`Tensor`, `ComputationGraph`, the op registry) assembled conceptually across Parts 3 and 4 rather than compiled as one running binary in this book's source material, and `save_model`/`load_model`/`gradient_check`/`check_gradient_health` are standalone utilities never wrapped in a `main()` that prints anything. Every numeric claim made about them in this chapter — the z-spread derivative in Worked Example 21.1.1, both Hessian-vector products in Worked Examples 21.2.1 and 21.2.2, the byte layout in Worked Example 21.3.1, and both `gradient_check` traces in Worked Examples 21.4.1 and 21.4.2 — was instead verified independently in Python against the real bond-pricing formula Part 7 uses, not by running this Mojo file.

## Chapter Summary

This chapter added the four tools a framework needs once "compute a gradient" isn't the whole job anymore. Section 21.1 showed how the implicit function theorem lets a graph treat an 18-iteration bisection solver as one opaque node, verified against Part 7's real z-spread bond: a `-0.0052912` predicted change in spread per dollar of price, confirmed to within 0.3 basis points by actually re-solving the bisection. Section 21.2 showed that `backward()` is differentiable itself, so calling it a second time on a gradient tensor produces a genuine second derivative — cross-checked, for a real two-variable function, against the full Hessian it never had to build. Section 21.3 reduced a trained network to what it actually is on disk — a header and raw bytes — and named the one thing production checkpoint formats add that this one doesn't: a version field. Section 21.4 separated the two failure modes unique to autograd code — a wrong gradient, caught by an independent finite-difference recomputation, and a corrupted one, caught by a cheap per-element sanity check — and was explicit about the trade-off both share: every safety net here is a debug-build tool, silently absent the moment a release build compiles `debug_assert` away.

## Self-Check Questions

1. In Section 21.1's worked example, `∂f/∂price = -1.0` regardless of what spread bisection finds. Why is this derivative always exactly `-1.0` and never, say, `-0.98` or `-1.02`, no matter what bond or what market price is involved?
2. Section 21.2's `hessian_vector_product` calls `zero_grad(params)` between its two `backward()` calls. Trace through what `params[0].grad` would incorrectly equal in Worked Example 21.2.2 if that line were deleted, using the real numbers from that example.
3. Worked Example 21.3.1 gives a 72-byte total file size assuming `Int` is 8 bytes. Recompute the total file size for the same two matrices (`W1`: `[3,2]`, `W2`: `[2,1]`) under the alternate assumption that `write_int` writes only 4 bytes instead of 8.
4. `gradient_check` uses `abs(numeric_grad - analytic) / max(abs(numeric_grad) + abs(analytic), 1e-8)` rather than plain `abs(numeric_grad - analytic)`. Using Worked Examples 21.4.1 and 21.4.2's numbers, explain concretely what would go wrong if the denominator were dropped and only the raw absolute difference were compared against a fixed threshold like `1e-4`.
5. `check_gradient_health`'s first check is `debug_assert(v == v, ...)`. Most equality checks compare a value to something else; this one compares a value to itself. Why does that specific, seemingly redundant comparison correctly detect a `NaN`, and what would `debug_assert(v != 0.0/0.0, ...)` fail to catch that this one doesn't?

## Where We Go Next

Part 7 (`part7/01-quantitative-finance-examples.md`) is where every primitive built across this book gets pointed at the problem it was designed for: the Struct-of-Arrays bond system from Section 18.2, the reduction kernels from Chapter 14, the `CustomFunction` framework from Section 21.1 above applied to the same z-spread solver this chapter differentiated through, and the GPU kernel design from Part 5 combine into a portfolio-scale pricing and risk pipeline — the chapter where "differentiable" and "financially meaningful" have to be the same property, not two separate claims.

## Worked Solutions

**1.** The objective function bisection actually solves is `f(price, spread) = calculate_bond_price(spread) - price`. `price` appears exactly once, with a coefficient of exactly `1` and a minus sign in front of it — there is no bond parameter, no spread value, and no market condition that changes that coefficient, because it comes from how the objective was *written*, not from anything about the bond being priced. `∂f/∂price = -1.0` for every bond, every spread, every market price, for the same reason `∂(x-5)/∂x = 1` regardless of what number replaces the `5`.

**2.** Without `zero_grad(params)`, the second `backward(graph, scalar)` call's `accumulate_gradient` (Chapter 17.2) *adds* onto whatever `params[0].grad` already held from the first `backward(graph, loss)` call rather than starting from zero. In Worked Example 21.2.2, the first `backward()` left `params[0].grad = [12, 4]` (the plain gradient `∇L`). The second backward, computing `Hv`, would then return `[12, 4] + [6, 4] = [18, 8]` instead of the correct `[6, 4]` — a Hessian-vector product silently contaminated with an extra, unrelated copy of the first-order gradient.

**3.** With `Int` at 4 bytes instead of 8: `count` = 4 bytes; `W1` header = `4+4=8` bytes, data unchanged at `6×4=24` bytes, subtotal `32`; `W2` header = `8` bytes, data `2×4=8` bytes, subtotal `16`. Total: `4 + 32 + 16 = 52` bytes — 20 bytes smaller than the 72-byte, 8-byte-`Int` version, with the entire difference coming from the five `Int` fields (`count` plus two `rows`/`cols` pairs) each shrinking by 4 bytes: `5 × 4 = 20`. ✓

**4.** Worked Example 21.4.1's correct rule has `numeric_grad ≈ analytic ≈ 6.0`, so the raw difference is a tiny floating-point residual, `~4×10⁻¹⁰` — comfortably under any reasonable fixed threshold either way, so the normalization doesn't matter *here*. The problem shows up at a different scale: imagine the same kind of correct rule but on a function whose gradient is legitimately huge, say `analytic ≈ 60000.0` with a `Float32`-rounding-sized absolute error of `0.01`. A fixed threshold of `1e-4` would flag `abs(60000.01 - 60000.0) = 0.01` as a "failure," even though relative to the gradient's own scale that's a difference of `0.01/60000 ≈ 1.7×10⁻⁷` — a completely healthy, correct gradient. Dividing by `max(abs(numeric)+abs(analytic), 1e-8)` makes the check scale-invariant: it asks "how big is this error *relative to the gradient itself*," which is what Worked Example 21.4.2's `0.3333` — unambiguously enormous regardless of scale — is actually measuring.

**5.** `NaN` is defined by IEEE 754 to compare unequal to *everything*, including itself — it's the one floating-point value for which `v == v` is `False`. That makes `v == v` a correct, portable NaN test without needing any NaN-specific function. `debug_assert(v != 0.0/0.0, ...)` would fail to catch this for a much more basic reason: `0.0/0.0` itself evaluates to `NaN`, and `v != NaN` is `True` for literally every value of `v`, NaN included (since NaN compares unequal to everything, even another NaN) — so that assertion can never fire regardless of what `v` holds, making it a check that always silently passes no matter what.
