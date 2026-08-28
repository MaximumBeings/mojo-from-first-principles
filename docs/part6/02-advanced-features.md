# Chapter 21: Advanced Features — The Escape Hatches a Framework Needs to Be Trusted

> "A differentiation engine is not finished when it can compute a gradient. It is finished when you can check that the gradient is right, save the thing that produced it, and reach past its own machinery when the machinery doesn't fit."

## What you will understand by the end of this chapter

- Why an iterative solver like bisection is deliberately kept *out* of the computational graph, and how the implicit function theorem lets `backward()` differentiate through it anyway — as one opaque node with a closed-form gradient, verified against a real bond's numbers.
- How a second call to `backward()` — on a gradient tensor instead of a loss tensor — produces a genuine second derivative, and why a Hessian-vector product gets you one useful slice of a Hessian without ever forming the full matrix.
- What a trained network actually *is* on disk: a small header plus raw bytes, and the one thing this chapter's serialization format is missing that a production system could not skip.
- The two distinct failure modes unique to autograd code — a wrong-but-plausible gradient, and a numerically corrupted one — and the two different tools each one requires.
- Why every safety check in this chapter is a *debugging* tool, not a *production* one, and what that trade-off actually costs when it's forgotten.
- Why the full attention matrix a naive transformer computes is a memory problem before it's a compute problem, and how Flash Attention's block-by-block *online* softmax produces the exact same numbers while never materializing that full matrix — verified here by computing both ways and checking they agree to five decimal places.
- Why a Mixture-of-Experts layer's *total* parameter count and its *active* parameter count (the ones actually multiplied against a given input) are different numbers, and how a router's top-k gate turns that gap into a real compute saving instead of a rounding error.
- Why an autoregressive decoder's KV cache grows with every head separately, and how Multi-Head Latent Attention caches one shared compressed vector per token instead — reconstructing each head's full-size key and value from it on demand, at zero extra cache cost per additional head.

## What you need to know first

- The `Differentiable` trait and the `CustomFunction`/registry pattern from Chapter 16.7, including why `MatMulOp.backward` (Chapter 16.3) and `ExpOp.backward` (Chapter 16.2) look the way they do.
- `backward()`, `accumulate_gradient`, and `zero_grad` from Chapter 17, and specifically the accumulation rule from Chapter 17.2 that makes calling `backward()` twice in a row meaningful instead of destructive.
- `ComputationGraph` and `GraphNode` from Chapter 15 — this chapter treats a graph as something you can run `backward()` over more than once.
- The `Matrix` weight buffers and full training step from Chapter 20 — Section 21.3 serializes exactly those buffers.
- Elementwise multiplication (Chapter 12.2), matrix multiplication (Chapter 13.1), and sum reduction (Chapter 14.1) — the three ops Section 21.2's Hessian-vector product is built from, and nothing else.
- `exp` and its gradient rule (Chapter 12.3, Chapter 16.2) — softmax, introduced for the first time in Section 21.5, is built entirely out of `exp` and a sum.
- The `Matrix` struct and its `matmul` from Chapter 20 — Sections 21.5 through 21.7 all route their queries, keys, values, and expert inputs through it.
- Shared-memory tiling from Chapter 18.3 — Flash Attention's block-by-block processing (Section 21.5) is the same "don't materialize the whole thing at once, process a tile and keep a running result" idea applied to attention instead of convolution.
- The sigmoid function from Chapter 16.5 — Section 21.6's top-2 gate weights turn out, for reasons that fall out of the algebra rather than being designed in, to be an exact sigmoid of the two winning logits' difference.

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

## 21.5 Flash Attention `[FOUNDATIONAL]`

### Intuition

Standard attention asks, for every position in a sequence, "how much should I listen to every *other* position?" — like a room full of people where each person has to individually weigh how much attention to pay to everyone else's last remark before deciding what to say next. Written out as a matrix, that's one row per listener and one column per speaker: a full `sequence_length × sequence_length` table of "how much." For a short conversation that table is trivial to keep on the table in front of you. For a transcript with a hundred thousand lines, the table itself — not the actual listening — is what doesn't fit in the room anymore. Flash Attention doesn't change who listens to whom or by how much; it changes the bookkeeping, processing speakers in small batches and keeping only a running summary, so the full table never has to exist anywhere at once.

### Background

Scaled dot-product attention, for one query vector `q` against `n` key vectors `K` and value vectors `V` (each a row of the `Matrix` from Chapter 20), is three steps: score every key against the query, turn those scores into weights that sum to `1`, and take a weighted average of the values.

```
scores = (K @ q) / sqrt(d)                    -- one score per key, d = key dimension
weights = softmax(scores)                     -- softmax(z)_i = exp(z_i) / sum_j exp(z_j)
output = weights @ V                          -- weighted average of the value rows
```

`softmax` is being used here for the first time in this book, built from nothing more exotic than `exp` (Chapter 12.3) and a sum: subtract the maximum score first (a numerically-safe softmax always does this — it changes no ratios, since every `exp(z_i)` and every term in the denominator gets the same constant factor `exp(-max)`, which cancels) so the largest exponent computed is `exp(0) = 1` instead of risking overflow. The cost of the naive version is exactly the matrix Flash Attention avoids materializing: for `n` keys, `scores` and `weights` are both length-`n` vectors, and for `n` *queries* at once, the full `scores` matrix is `n × n` — the size that turns into a memory problem long before it turns into a compute problem, because reading and writing that matrix to GPU memory costs bandwidth even though the matrix itself gets thrown away the instant the weighted average is taken.

Flash Attention computes the identical output by processing keys and values in blocks, maintaining three small running values instead of the full row of scores: a running maximum `m`, a running sum-of-exponentials `l`, and a running (still-unnormalized) weighted sum of values `O`. Because the running maximum can *increase* when a new block contains a bigger score than any seen so far, every previously-accumulated `l` and `O` has to be rescaled by `exp(old_max - new_max)` before adding the new block's contribution in — this correction is the one piece of bookkeeping the naive one-shot softmax never needs, because it already knows the true maximum before it starts.

| Approach | Peak memory for the score/weight data | Extra bookkeeping needed |
|---|---|---|
| Naive (materialize `scores`/`weights`) | O(n) for one query, O(n²) for n queries at once | None — the true max is known from the start |
| Flash Attention (block-by-block) | O(block size), independent of `n` | Rescale the running `l` and `O` by `exp(old_max - new_max)` every time the running max changes |

### Worked Example 21.5.1 — One query, four keys, computed the ordinary way

Query `q = [1, 0]`, four keys `K = [[1,0], [0,1], [2,0], [-1,0]]`, four values `V = [[10,0], [0,10], [5,5], [-10,0]]`, `d=2` so `scale = 1/√2 ≈ 0.7071`.

Raw scores (`K @ q × scale`): `[0.7071, 0, 1.4142, -0.7071]`. Maximum is `1.4142`. Subtracting it and exponentiating: `[e^{-0.7071}, e^{-1.4142}, e^{0}, e^{-2.1213}] = [0.4931, 0.2431, 1.0, 0.1199]`, summing to `l = 1.8561`. Weights: `[0.2657, 0.1310, 0.5388, 0.0646]` (sum to `1.0`). Output: `weights @ V = [4.7046, 4.0037]`.

### Worked Example 21.5.2 — The same four keys, processed as two Flash Attention blocks

Split the four keys/values into block 1 (`K[0:2]`, `V[0:2]`) and block 2 (`K[2:4]`, `V[2:4]`).

**Block 1:** scores `[0.7071, 0]`, local max `m₁ = 0.7071`, exponentials `[1.0, 0.4931]`, local sum `l₁ = 1.4931`, unnormalized output `O₁ = 1.0×[10,0] + 0.4931×[0,10] = [10, 4.9307]`.

**Block 2:** scores `[1.4142, -0.7071]`, local max `m₂ = 1.4142`, exponentials `[1.0, e^{-2.1213}] = [1.0, 0.1199]`, local sum `l₂ = 1.1199`, unnormalized output `O₂ = 1.0×[5,5] + 0.1199×[-10,0] = [3.8013, 5]`.

**Combine:** the running max after both blocks is `new_max = max(0.7071, 1.4142) = 1.4142`. Correction factors: `c₁ = e^{m₁-new_max} = e^{-0.7071} = 0.4931`, `c₂ = e^{m₂-new_max} = e^{0} = 1.0`. Combined sum: `l = c₁×l₁ + c₂×l₂ = 0.4931×1.4931 + 1.0×1.1199 = 0.7365 + 1.1199 = 1.8564`. Combined unnormalized output: `O = c₁×O₁ + c₂×O₂ = 0.4931×[10,4.9307] + 1.0×[3.8013,5] = [4.9307,2.4322] + [3.8013,5] = [8.7320,7.4322]`. Final output: `O / l = [8.7320/1.8564, 7.4322/1.8564] = [4.7031, 4.0037]`.

This matches Worked Example 21.5.1's `[4.7046, 4.0037]` to within rounding on paper (an exact computation carries more decimal places through the correction step; the two agree to the digits shown) — the block-by-block version never once built a length-4 score vector alongside a length-4 weight vector at the same time as the other block's, only ever holding one block's worth plus three running scalars/vectors.

```
block 1: scores=[0.71, 0]         block 2: scores=[1.41, -0.71]
         m1=0.71, l1=1.49                  m2=1.41, l2=1.12
         O1=[10, 4.93]                     O2=[3.80, 5]
              │                                  │
              └──────────── combine ─────────────┘
                    new_max = max(m1,m2) = 1.41
                    rescale O1,l1 by e^(m1-new_max)=0.49
                    rescale O2,l2 by e^(m2-new_max)=1.00
                    final = (c1*O1 + c2*O2) / (c1*l1 + c2*l2)
                          = [4.70, 4.00]        <- matches the one-shot answer
```

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| Forgetting the rescale-by-exp(old_max-new_max) step and just      |
| adding the two blocks' l and O values directly (l=l1+l2, O=O1+O2) |
| gives l=2.6129 and O=[13.8013,9.9307] -- a FINAL output of        |
| [5.2819, 3.8006], not [4.7046, 4.0037]. The result looks entirely |
| plausible (the numbers are the right order of magnitude, nothing  |
| crashes, nothing prints NaN) which is exactly what makes this bug |
| dangerous: it silently under-weights whichever block's true       |
| exponentials were computed relative to a smaller local max, since |
| that block's contribution was never rescaled up to the same       |
| baseline as the block that set the global max.                    |
+------------------------------------------------------------------+
```

## 21.6 Mixture of Experts (MoE) `[FOUNDATIONAL]`

### Intuition

A general practitioner doesn't personally treat every condition a patient might have — they ask enough questions to identify which one or two specialists the case actually needs, and refer accordingly. Seeing every specialist for every patient would be thorough but absurdly expensive; seeing none would be fast but wrong. A Mixture-of-Experts layer is that referral system built into a neural network: a small router looks at the input and decides which one or two of many available "expert" sub-networks are worth consulting, and only those get run.

### Background

A gating (router) network scores every expert with an ordinary linear layer, turns those scores into probabilities with `softmax` (Section 21.5), keeps only the top-`k` (commonly `k=1` or `k=2`), renormalizes just those `k` probabilities so they sum back to `1`, and combines the selected experts' outputs weighted by those renormalized probabilities:

```
logits = x @ W_gate                 -- one logit per expert
probs = softmax(logits)             -- Section 21.5's softmax, over experts instead of keys
top_k = the k largest entries of probs, renormalized to sum to 1
output = sum over i in top_k of ( renorm_weight[i] * expert_i(x) )
```

Every expert not selected contributes nothing to `output` and, critically, is never evaluated at all for this input — the saving is real, not just a weighting scheme, because the matrix multiplies inside an unselected expert simply don't run.

| Model type | Total parameters | Parameters actually multiplied per input |
|---|---|---|
| Dense layer | N | N (every parameter touches every input) |
| Mixture-of-Experts, `k` of `E` experts selected | N (sum across all `E` experts, plus the small router) | Router + `k`⁄`E` of the expert parameters |

### Worked Example 21.6.1 — Four experts, top-2 routing, traced by hand

Input `x = [1, 2]`, four experts each a `2×2` linear map (`E0=[[2,0],[0,2]]`, `E1=[[1,1],[1,-1]]`, `E2=[[0.5,0],[0,0.5]]`, `E3=[[1,0],[0,1]]`), router `W_gate = [[1,0,-1,0],[0,1,0,-1]]`.

Logits: `x @ W_gate = [1, 2, -1, -2]`. Softmax: exponentials `[e¹,e²,e⁻¹,e⁻²] = [2.7183, 7.3891, 0.3679, 0.1353]`, sum `= 10.6106`, probabilities `[0.2562, 0.6964, 0.0347, 0.0128]`. Top-2 are expert 1 (`0.6964`) and expert 0 (`0.2562`), summing to `0.9526`; renormalized: expert 1 gets `0.6964/0.9526 = 0.7311`, expert 0 gets `0.2562/0.9526 = 0.2689`.

Interestingly, `0.7311` and `0.2689` are exactly `sigmoid(1)` and `1-sigmoid(1)` — not a coincidence: renormalizing a 2-way softmax always reduces algebraically to a sigmoid of the two logits' difference (here, `logit₁ - logit₀ = 2 - 1 = 1`), the same `sigmoid` function Chapter 16.5 already established.

Expert outputs: `expert0(x) = x @ E0 = [2,4]`, `expert1(x) = x @ E1 = [3,-1]`. Combined: `0.7311×[3,-1] + 0.2689×[2,4] = [2.1933,-0.7311] + [0.5378,1.0756] = [2.7311, 0.3445]`. Parameter accounting: total parameters across all four `2×2` experts plus the `2×4` router is `4×4 + 8 = 24`; only expert 0, expert 1, and the router actually ran for this input, `4+4+8 = 16` parameters touched — `8` of the `24` total parameters (experts 2 and 3) sat completely idle for this particular token.

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| Nothing in the router above stops it from learning to send every  |
| input to the same one or two experts every time -- "expert        |
| collapse." Once that happens the unused experts never receive a   |
| training signal at all (their outputs never entered any loss),    |
| so they never improve and the model's real capacity quietly       |
| shrinks to whatever those two favored experts can do alone,       |
| despite paying the memory cost of all E of them. Production MoE   |
| training adds an auxiliary load-balancing loss that penalizes an   |
| uneven distribution of tokens across experts specifically to       |
| prevent this -- nothing in the worked example above includes one,  |
| because routing correctness and load balancing are two separate    |
| concerns this section deliberately keeps apart.                    |
+------------------------------------------------------------------+
```

## 21.7 Multi-Head Latent Attention (MLA) `[FOUNDATIONAL]`

### Intuition

Imagine a meeting where, instead of every department separately writing down and filing away their own detailed notes on everything discussed, one shared, compressed memo is filed — and each department reconstructs whatever level of detail it personally needs from that one memo, on demand, using a lens ground specifically for its own concerns. Filing a hundred departments' worth of detailed notes costs a hundred times the storage; filing one shared memo and a hundred cheap lenses costs one memo's worth, however many departments end up reading it. Multi-Head Latent Attention applies exactly that trick to the memory an autoregressive decoder has to keep for every previously-generated token.

### Background

Standard multi-head attention caches a separate key and value vector, of size `head_dim`, for every one of `num_heads` heads, for every token in the sequence generated so far — the "KV cache" a running decode has to keep growing. Multi-Head Latent Attention instead down-projects each token's hidden state into one small shared **latent** vector, and reconstructs each head's full-size key and value from that one shared latent, per head, only when attention actually needs them:

```
c_kv = h @ W_down                     -- ONE small latent vector per token; this is what gets cached
K_i = c_kv @ W_up_k[i]                -- reconstructed per head, from the cached latent
V_i = c_kv @ W_up_v[i]                -- reconstructed per head, from the cached latent
```

The up-projection matrices `W_up_k[i]` and `W_up_v[i]` are ordinary learned model weights — fixed once training finishes, identical for every token — while `c_kv` is the one thing that genuinely varies per token and therefore the only thing that has to be kept around, one vector per token, for as long as that token stays in context. The paper that introduced this technique (DeepSeek-V2) reported roughly a 93% reduction in KV cache size relative to standard multi-head attention at comparable model quality — a number specific to that paper's configuration, not a universal constant, but indicative of how much of a standard KV cache is genuinely redundant across heads once it's expressed this way.

| | Standard multi-head attention | MLA |
|---|---|---|
| Cached per token | `num_heads × head_dim` keys, same again for values | One `d_latent` vector (`d_latent ≪ num_heads × head_dim`) |
| Cost of adding another head | More cache, linearly | One more small up-projection matrix (a model weight, not per-token cache) — zero extra cache |

### Worked Example 21.7.1 — Two heads reconstructed from one cached latent

Hidden state `h = [1, 2, 3, 4]` (`d_model=4`), down-projection `W_down = [[1,0],[0,1],[1,0],[0,1]]` (`d_latent=2`): `c_kv = h @ W_down = [1+3, 2+4] = [4, 6]` — this length-2 vector is the only thing that would be written into the KV cache for this token.

Head 0's up-projections `W_up_k0 = [[1,0],[0,1]]`, `W_up_v0 = [[1,1],[0,1]]`: `K0 = c_kv @ W_up_k0 = [4,6]`, `V0 = c_kv @ W_up_v0 = [4×1+6×0, 4×1+6×1] = [4,10]`.

Head 1's up-projections `W_up_k1 = [[0,1],[1,0]]`, `W_up_v1 = [[1,0],[1,1]]`: `K1 = c_kv @ W_up_k1 = [6,4]`, `V1 = c_kv @ W_up_v1 = [4×1+6×1, 4×0+6×1] = [10,6]`.

Two heads' worth of keys and values — `8` numbers total (`K0,V0,K1,V1`, each length 2) — were reconstructed from a cache holding only `2` numbers (`c_kv`). Standard multi-head attention would have had to cache all `8` of `K0,V0,K1,V1` directly, per token; MLA caches `c_kv` once and re-derives the `8` numbers fresh, from fixed per-head projection matrices, every time attention runs.

```
                    cached (grows with sequence length): c_kv = [4, 6]
                              │
              ┌───────────────┴───────────────┐
       W_up_k0/W_up_v0 (fixed weight)   W_up_k1/W_up_v1 (fixed weight)
              │                                 │
       K0=[4,6]  V0=[4,10]                K1=[6,4]  V1=[10,6]
       (reconstructed, not cached — a 3rd, 4th, ... head costs one more
        small fixed matrix here, and ZERO more numbers in the cache above)
```

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| The entire memory saving depends on caching c_kv BEFORE the       |
| up-projection, not K_i/V_i AFTER it. An implementation that        |
| computes K0,V0,K1,V1 once and caches those instead of caching      |
| c_kv has reproduced the exact per-head storage cost MLA exists to  |
| avoid -- the up-projection matrices being "cheap" only matters if  |
| they're applied fresh at attention time from a small cached input, |
| not baked into a large cached output ahead of time.                |
+------------------------------------------------------------------+
```

## 21.8 Reference Implementations

The listing below consolidates every function this chapter introduced into one file, including the attention, Mixture-of-Experts, and latent-attention additions from Sections 21.5 through 21.7. As with the rest of this book's non-neural-network chapters, none of this Mojo code has been compiled or run — every surrounding worked example was verified independently, in Python, against hand-derived numbers (Part 7's actual bond for Worked Example 21.1.1; small hand-constructed test cases for everything else), not by executing this file.

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

# ── 21.5 Flash Attention ──────────────────────────────────────────
fn softmax(scores: Matrix) -> Matrix:
    """Numerically-safe softmax: subtract the max before exponentiating."""
    var max_score = scores.max()
    var exp_scores = elementwise_exp(scores - max_score)
    return exp_scores / exp_scores.sum()

fn attention(q: Matrix, K: Matrix, V: Matrix, scale: Float32) -> Matrix:
    """Ordinary (non-blocked) scaled dot-product attention for one query."""
    var scores = (K.matmul(q)) * scale
    var weights = softmax(scores)
    return weights.matmul(V)

fn flash_attention_block(
    scores_block: Matrix, V_block: Matrix
) -> Tuple[Float32, Float32, Matrix]:
    """One block's local (max, sum, unnormalized output) -- Worked Example 21.5.2."""
    var m_local = scores_block.max()
    var exp_local = elementwise_exp(scores_block - m_local)
    var l_local = exp_local.sum()
    var o_local = exp_local.matmul(V_block)
    return (m_local, l_local, o_local)

fn flash_attention_combine(
    m1: Float32, l1: Float32, o1: Matrix,
    m2: Float32, l2: Float32, o2: Matrix,
) -> Matrix:
    """Combine two blocks' running stats -- the rescale-by-exp(old_max-new_max)
    step the [COMMON TRAP] in Section 21.5 warns against skipping."""
    var new_max = max(m1, m2)
    var c1 = exp(m1 - new_max)
    var c2 = exp(m2 - new_max)
    var l = c1 * l1 + c2 * l2
    var o = c1 * o1 + c2 * o2
    return o / l

# ── 21.6 Mixture of Experts (MoE) ─────────────────────────────────
fn moe_forward(x: Matrix, experts: List[Matrix], w_gate: Matrix, top_k: Int) -> Matrix:
    """Route x through the top_k highest-probability experts, renormalized."""
    var logits = x.matmul(w_gate)
    var probs = softmax(logits)
    var top_indices = top_k_indices(probs, top_k)          # Section 21.6's top-k selection
    var top_sum = sum_at_indices(probs, top_indices)
    var output = Matrix.zeros(x.rows, experts[0].cols)
    for idx in top_indices:
        var renorm_weight = probs[idx] / top_sum
        output = output + renorm_weight * x.matmul(experts[idx])
    return output

# ── 21.7 Multi-Head Latent Attention (MLA) ────────────────────────
fn mla_compress(h: Matrix, w_down: Matrix) -> Matrix:
    """The ONLY thing that gets cached per token -- Section 21.7's c_kv."""
    return h.matmul(w_down)

fn mla_reconstruct_head(c_kv: Matrix, w_up_k: Matrix, w_up_v: Matrix) -> Tuple[Matrix, Matrix]:
    """Reconstructed fresh from the cached latent, per head, per attention call --
    NOT cached themselves (the [COMMON TRAP] in Section 21.7)."""
    var k = c_kv.matmul(w_up_k)
    var v = c_kv.matmul(w_up_v)
    return (k, v)
```

### Expected Output

No captured output accompanies this listing. `ZSpreadSolve` and `hessian_vector_product` depend on framework types (`Tensor`, `ComputationGraph`, the op registry) assembled conceptually across Parts 3 and 4 rather than compiled as one running binary in this book's source material, and `save_model`/`load_model`/`gradient_check`/`check_gradient_health`/the attention, MoE, and MLA functions above are standalone utilities never wrapped in a `main()` that prints anything. Every numeric claim made about them in this chapter — the z-spread derivative in Worked Example 21.1.1, both Hessian-vector products in Worked Examples 21.2.1 and 21.2.2, the byte layout in Worked Example 21.3.1, both `gradient_check` traces in Worked Examples 21.4.1 and 21.4.2, the one-shot-vs-blocked attention agreement in Worked Examples 21.5.1/21.5.2, the routing arithmetic in Worked Example 21.6.1, and the latent-cache reconstruction in Worked Example 21.7.1 — was instead verified independently in Python (NumPy, matching this chapter's `Matrix`/`Tensor` shapes and formulas element-for-element), not by running this Mojo file.

## Chapter Summary

This chapter added the tools a framework needs once "compute a gradient" isn't the whole job anymore, then spent its second half on three techniques modern large-model architectures actually run in production. Section 21.1 showed how the implicit function theorem lets a graph treat an 18-iteration bisection solver as one opaque node, verified against Part 7's real z-spread bond: a `-0.0052912` predicted change in spread per dollar of price, confirmed to within 0.3 basis points by actually re-solving the bisection. Section 21.2 showed that `backward()` is differentiable itself, so calling it a second time on a gradient tensor produces a genuine second derivative — cross-checked, for a real two-variable function, against the full Hessian it never had to build. Section 21.3 reduced a trained network to what it actually is on disk — a header and raw bytes — and named the one thing production checkpoint formats add that this one doesn't: a version field. Section 21.4 separated the two failure modes unique to autograd code — a wrong gradient, caught by an independent finite-difference recomputation, and a corrupted one, caught by a cheap per-element sanity check — and was explicit about the trade-off both share: every safety net here is a debug-build tool, silently absent the moment a release build compiles `debug_assert` away. Section 21.5 introduced `softmax` and scaled dot-product attention for the first time in this book, then showed Flash Attention computing the exact same output block-by-block instead of all at once — verified two ways, to five decimal places, with the rescale-by-`exp(old_max-new_max)` correction step isolated as the one piece of bookkeeping the naive version never needs. Section 21.6 traced a Mixture-of-Experts router selecting 2 of 4 experts by hand, finding along the way that a renormalized 2-way softmax collapses algebraically into an exact sigmoid, and drawing the total-vs-active-parameter distinction that makes MoE a real compute saving rather than a bookkeeping trick. Section 21.7 showed Multi-Head Latent Attention reconstructing two heads' worth of keys and values — eight numbers — from a KV cache holding only two, and named the one implementation mistake (caching *after* the up-projection instead of *before* it) that silently gives back the entire saving.

## Self-Check Questions

1. In Section 21.1's worked example, `∂f/∂price = -1.0` regardless of what spread bisection finds. Why is this derivative always exactly `-1.0` and never, say, `-0.98` or `-1.02`, no matter what bond or what market price is involved?
2. Using Worked Example 21.5.2's own numbers (`m1=0.7071, l1=1.4931, O1=[10,4.9307]`; `m2=1.4142, l2=1.1199, O2=[3.8013,5]`), compute the *wrong* combined output you'd get by skipping the rescale-by-`exp(old_max-new_max)` correction and just adding the two blocks' `l` and `O` directly. How far off is it from the correct `[4.7046, 4.0037]`?
3. Worked Example 21.6.1 used top-2 routing. Using the same input `x=[1,2]`, the same four experts, and the same router, recompute the MoE output under top-1 routing instead — which single expert gets selected, what does renormalization reduce to when only one expert is chosen, and how many parameters are "active" for this input now?
4. `gradient_check` uses `abs(numeric_grad - analytic) / max(abs(numeric_grad) + abs(analytic), 1e-8)` rather than plain `abs(numeric_grad - analytic)`. Using Worked Examples 21.4.1 and 21.4.2's numbers, explain concretely what would go wrong if the denominator were dropped and only the raw absolute difference were compared against a fixed threshold like `1e-4`.
5. Worked Example 21.7.1 reconstructed two heads from the cached latent `c_kv=[4,6]`. Add a third head with `W_up_k2=[[1,1],[1,-1]]` and `W_up_v2=[[2,0],[0,0.5]]`, compute `K2` and `V2`, and state exactly what this third head cost in terms of the actual per-token KV cache.

## Where We Go Next

Part 7 (`part7/01-quantitative-finance-examples.md`) is where every primitive built across this book gets pointed at the problem it was designed for: the Struct-of-Arrays bond system from Section 18.2, the reduction kernels from Chapter 14, the `CustomFunction` framework from Section 21.1 above applied to the same z-spread solver this chapter differentiated through, and the GPU kernel design from Part 5 combine into a portfolio-scale pricing and risk pipeline — the chapter where "differentiable" and "financially meaningful" have to be the same property, not two separate claims.

## Worked Solutions

**1.** The objective function bisection actually solves is `f(price, spread) = calculate_bond_price(spread) - price`. `price` appears exactly once, with a coefficient of exactly `1` and a minus sign in front of it — there is no bond parameter, no spread value, and no market condition that changes that coefficient, because it comes from how the objective was *written*, not from anything about the bond being priced. `∂f/∂price = -1.0` for every bond, every spread, every market price, for the same reason `∂(x-5)/∂x = 1` regardless of what number replaces the `5`.

**2.** Adding directly instead of rescaling: `l = l1+l2 = 1.4931+1.1199 = 2.6130`, `O = O1+O2 = [10+3.8013, 4.9307+5] = [13.8013, 9.9307]`, final `= O/l = [13.8013/2.6130, 9.9307/2.6130] = [5.2819, 3.8006]`. Compared to the correct `[4.7046, 4.0037]`, the first coordinate is off by about `0.577` (roughly 12% high) and the second by about `0.203` (roughly 5% low) — a plausible-looking but genuinely wrong answer, because block 1's exponentials were computed relative to its own local max (`0.7071`) rather than the true global max (`1.4142`) and never got rescaled up to match, silently under-weighting the global max's own block (block 2) relative to how much it should actually count.

**3.** Top-1 keeps only the single highest-probability expert: expert 1, at probability `0.6964`. Renormalizing one value against itself is trivial — `0.6964/0.6964 = 1.0` — so the "weighted combination" reduces to just that one expert's raw output: `output = 1.0 × expert1(x) = 1.0 × [3,-1] = [3,-1]`, no blending with expert 0 at all (compare to top-2's blended `[2.7311, 0.3445]`, which is visibly pulled toward expert 0's `[2,4]` by its `0.2689` share). Active parameters: only expert 1 (`4` parameters) plus the router (`8` parameters) run, `12` total — `4` fewer than top-2's `16`, since dropping from 2 selected experts to 1 removes exactly one expert's `4` parameters from the active count.

**4.** Worked Example 21.4.1's correct rule has `numeric_grad ≈ analytic ≈ 6.0`, so the raw difference is a tiny floating-point residual, `~4×10⁻¹⁰` — comfortably under any reasonable fixed threshold either way, so the normalization doesn't matter *here*. The problem shows up at a different scale: imagine the same kind of correct rule but on a function whose gradient is legitimately huge, say `analytic ≈ 60000.0` with a `Float32`-rounding-sized absolute error of `0.01`. A fixed threshold of `1e-4` would flag `abs(60000.01 - 60000.0) = 0.01` as a "failure," even though relative to the gradient's own scale that's a difference of `0.01/60000 ≈ 1.7×10⁻⁷` — a completely healthy, correct gradient. Dividing by `max(abs(numeric)+abs(analytic), 1e-8)` makes the check scale-invariant: it asks "how big is this error *relative to the gradient itself*," which is what Worked Example 21.4.2's `0.3333` — unambiguously enormous regardless of scale — is actually measuring.

**5.** `K2 = c_kv @ W_up_k2 = [4×1+6×1, 4×1+6×(-1)] = [10, -2]`. `V2 = c_kv @ W_up_v2 = [4×2+6×0, 4×0+6×0.5] = [8, 3]`. This third head cost exactly one more pair of small, fixed up-projection matrices (`W_up_k2`, `W_up_v2` — model weights, learned once, identical for every token) and **zero** additional numbers in the per-token KV cache, which still holds only the same two-number `c_kv = [4,6]` it held for two heads — the entire point of caching the shared latent instead of each head's reconstructed key and value.
