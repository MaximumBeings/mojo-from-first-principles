# Chapter 20: Neural Network Layers — Assembling a Network the Autograd Engine Never Actually Sees

> "Parts 1 through 5 built a `Tensor`, a `ComputationGraph`, a registry of backward rules, and kernels tuned enough to trust their numbers. This chapter is the payoff every one of those chapters was justified by — a real, trainable network — and also, honestly, the first chapter in this book that doesn't reach for a single piece of that machinery: no `Tensor`, no `GraphNode`, no `chain_rule_step`. Every gradient here is derived and coded by hand, one layer at a time, in a separate `Matrix` struct that has never heard of `Differentiable`. Watching the two approaches solve the identical chain-rule problem side by side is its own kind of lesson."

**What you will understand by the end of this chapter:**

- Why a layer feeding into ReLU wants He initialization and a layer feeding into a saturating activation like tanh or sigmoid wants Xavier initialization — derived from what each activation does to a signal's variance as it passes through, not just stated as a rule of thumb
- The three activation functions this network actually uses, each paired with the exact local-derivative formula Chapter 16.5 already derived for the *registered* `ReluOp`/`SigmoidOp`/`TanhOp` — the same math, reached by a completely different code path
- Mean squared error's forward formula and its gradient — and a real scale mismatch between the two as this network's own code defines them, worth tracing through exact numbers rather than taking on faith
- The full forward-then-backward chain through a multi-layer network, traced by hand on a small two-layer version of the same pattern the real five-layer network uses, landing on gradients for every weight and bias matrix
- Precision, recall, and F1 built from the same confusion-matrix bookkeeping any classifier's quality is measured with, fed by an argmax over two output units — literally Chapter 14.2's `max_reduce_kernel` idea, applied by hand to a two-element row instead of a full reduction kernel

**What you need to know first:**

- Chapter 13 (`matrix_multiply`, transpose) — this chapter's `Matrix.matmul` and `Matrix.transpose` are the identical algorithm, reimplemented against a `Matrix` struct instead of `Tensor`
- Chapter 12.4 (broadcasting) — `add_bias`'s "every row gets the same bias vector" is broadcasting a `(1, N)` bias against a `(batch, N)` activation, the same shape rule Chapter 12.4 already established
- Chapter 16 in full, especially 16.5 (`ReluOp`, `SigmoidOp`, `TanhOp`'s backward rules) and 16.1's chain-rule-as-sum-over-paths — this chapter's backward pass is that same chain rule, applied layer by layer instead of routed through a registry
- Chapter 17 (the reverse-mode traversal `chain_rule_step` runs node by node) — useful contrast for Section 20.4's discussion of what this chapter does differently
- Chapter 14.2 (`max_reduce_kernel`'s argmax tracking) — Section 20.5's two-way `pred_class` comparison is the same idea at its smallest possible scale

## 20.1 Linear Layer Implementation `[FOUNDATIONAL]`

### Intuition

Pouring a signal through five layers in a row is a lot like passing a rumor through a line of five people — if each person tends to exaggerate what they hear, the story balloons into nonsense by the fifth retelling; if each person tends to downplay it, nothing is left of it by the end. A layer's *initial* weights set how much a signal grows or shrinks passing through, before any training has had a chance to correct it — and the "safe" starting scale for that growth depends on what happens to the signal immediately afterward. A ReLU zeroes out roughly half of whatever reaches it, so the weights feeding into one need to compensate by starting a little larger; tanh and sigmoid squash large values toward flat, saturated regions, so the weights feeding into either want to start a little smaller, keeping most values in the region where the activation still has a meaningful slope.

### Background

A linear (fully-connected) layer is `Z = X @ W + b` — the matmul from Chapter 13 plus a broadcast bias add from Chapter 12.4, applied to a `Matrix` struct instead of `Tensor`:

```mojo
struct Matrix:
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int
    var size: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.data = UnsafePointer[Float32].alloc(self.size)
        for i in range(self.size):
            self.data[i] = 0.0

    fn he_init(inout self, fan_in: Int):
        """He initialization for ReLU layers: std = sqrt(2 / fan_in)."""
        var std_dev = sqrt(Float32(2.0) / Float32(fan_in))
        for i in range(self.size):
            # Box-Muller transform: two uniform samples -> one normal sample
            var u1 = Float32(random_float64())
            var u2 = Float32(random_float64())
            var normal = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159 * u2)
            self.data[i] = normal * std_dev

    fn xavier_init(inout self, fan_in: Int, fan_out: Int):
        """Xavier initialization: limit = sqrt(6 / (fan_in + fan_out))."""
        var limit = sqrt(Float32(6.0) / Float32(fan_in + fan_out))
        for i in range(self.size):
            var r = Float32(random_float64())
            self.data[i] = (r - Float32(0.5)) * Float32(2.0) * limit

    fn matmul(self, other: Matrix, inout result: Matrix):
        # C[i,j] = sum_k A[i,k] * B[k,j] -- Chapter 13's matmul, reused verbatim
        for i in range(self.rows):
            for j in range(other.cols):
                var s: Float32 = 0.0
                for k in range(self.cols):
                    s += self.get(i, k) * other.get(k, j)
                result.set(i, j, s)

    fn add_bias(inout self, bias: Matrix):
        """Z = XW + b -- every row gets the same bias vector (Chapter 12.4 broadcasting)."""
        for i in range(self.rows):
            for j in range(self.cols):
                var idx = i * self.cols + j
                self.data[idx] = self.data[idx] + bias.data[j]
```

The network built in this chapter wires five of these together: `W1: [2,24]`, `W2: [24,16]`, `W3: [16,12]` are all He-initialized (each feeds a ReLU); `W4: [12,8]` is Xavier-initialized (feeds tanh); `W5: [8,2]` is Xavier-initialized (feeds sigmoid) — the initialization scheme tracks the activation *downstream* of each weight matrix, not that weight matrix's position in the network.

### Worked Example 20.1.1 — He initialization, one weight traced through Box-Muller

For `W1` (`fan_in = 2`): `std_dev = sqrt(2/2) = 1.0`. Take one Box-Muller draw with `u1 = 0.5`, `u2 = 0.1`: `normal = sqrt(-2·ln(0.5)) · cos(2π·0.1) = sqrt(1.3863) · cos(36°) ≈ 1.1774 × 0.8090 ≈ 0.9525`. That one weight's value: `0.9525 × 1.0 = 0.9525`. A smaller `fan_in` produces a *larger* `std_dev` (here, exactly `1.0`, the largest this network's five layers ever use) — the fewer inputs a layer has, the more each individual weight has to carry, so He initialization compensates by starting each one larger.

### Worked Example 20.1.2 — Xavier initialization, contrasted directly

For `W5` (`fan_in = 8`, `fan_out = 2`): `limit = sqrt(6 / 10) ≈ 0.7746`. With a uniform draw `r = 0.9`: `weight = (0.9 - 0.5) × 2 × 0.7746 ≈ 0.6197`. Unlike He's normal distribution, Xavier draws from a *uniform* range `[-limit, limit]` — but the shape of the formula is the same idea: `fan_in + fan_out` in the denominator means a layer with many inputs *and* many outputs gets a smaller `limit`, keeping the total variance the signal picks up passing through roughly constant regardless of that layer's width.

### Worked Example 20.1.3 — A linear layer's forward pass, traced completely

`X = [1, 2]` (one sample, two features), `W = [[1,0,1],[0,1,1]]` (`2×3`), `b = [1,1,1]`. `Z = X @ W`: `Z[0] = 1·1 + 2·0 = 1`, `Z[1] = 1·0 + 2·1 = 2`, `Z[2] = 1·1 + 2·1 = 3`, so `Z = [1,2,3]` before the bias. `add_bias` adds the same `b` to this one row: `Z = [1+1, 2+1, 3+1] = [2,3,4]`. This exact `X`, `W`, and pre-bias `Z = [1,2,3]` reappear as the starting point for Section 20.4's full forward-and-backward trace.

## 20.2 Activation Functions `[FOUNDATIONAL]`

### Intuition

Three different activations, three different jobs. ReLU is a one-way valve: signal above zero passes through completely unchanged, signal at or below zero is shut off entirely — cheap, and exactly why it needs He initialization to compensate for routinely discarding half of whatever arrives. Sigmoid is a dimmer switch stuck reporting a single brightness between fully off and fully on, useful exactly where the final answer needs to look like a probability. Tanh is that same dimmer switch recentered to swing between two *symmetric* extremes instead of an asymmetric zero-to-one range, which is why this network places it right before the sigmoid output layer — a signal is centered near zero at the point it enters the network's final decision, rather than already biased toward one end.

### Background

Every activation is implemented alongside its derivative, since the backward pass needs both — and each derivative here is exactly the local-derivative formula Chapter 16.5 already derived for the *registered* `ReluOp`, `SigmoidOp`, and `TanhOp`, just computed directly from a `Float32` instead of returned from a `Differentiable.backward` call:

```mojo
fn relu(x: Float32) -> Float32:
    """f(x) = max(0, x)"""
    return x if x > Float32(0.0) else Float32(0.0)

fn relu_derivative(x: Float32) -> Float32:
    """f'(x) = 1 if x > 0, else 0"""
    return Float32(1.0) if x > Float32(0.0) else Float32(0.0)

fn sigmoid(x: Float32) -> Float32:
    """sigma(x) = 1 / (1 + e^-x)"""
    return Float32(1.0) / (Float32(1.0) + exp(-x))

fn sigmoid_derivative(x: Float32) -> Float32:
    """sigma'(x) = sigma(x) * (1 - sigma(x))"""
    var s = sigmoid(x)
    return s * (Float32(1.0) - s)

fn tanh_activation(x: Float32) -> Float32:
    """tanh(x) = (e^x - e^-x) / (e^x + e^-x)"""
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

fn tanh_derivative(x: Float32) -> Float32:
    """tanh'(x) = 1 - tanh(x)^2"""
    var t = tanh_activation(x)
    return Float32(1.0) - t * t
```

Applied element-wise across a whole layer's activations with the SIMD-friendly loop pattern from Chapter 19.1:

```mojo
fn apply_relu(inout self: Matrix):
    for i in range(self.size):
        self.data[i] = relu(self.data[i])
```

### Worked Example 20.2.1 — All three activations at their defining points

`relu` on `x = [-2, 0, 3]`: `[0, 0, 3]` — and `relu_derivative` on the same three points: `[0, 0, 1]`. Note `x=0` produces a derivative of `0`, not `1` — the strict `>` comparison, identical to `ReluOp`'s `greater_than_zero_mask` from Chapter 16.5, and identical convention this book already flagged there as one defensible choice among two at a point where ReLU's true derivative is undefined. At `x=0`: `sigmoid(0) = 0.5`, `sigmoid_derivative(0) = 0.5 × 0.5 = 0.25`; `tanh_activation(0) = 0`, `tanh_derivative(0) = 1 - 0² = 1` — the exact same two numbers (`0.25` and `1`) Chapter 16's Worked Example 16.5.2 derived for `SigmoidOp` and `TanhOp`, confirming that a hand-written scalar function and a registered `Differentiable` implementation compute identical mathematics when they're differentiating the identical formula.

## 20.3 Loss Functions `[FOUNDATIONAL]`

### Intuition

A loss function is the network's one source of truth about how wrong its current guess is, and its gradient is what turns "how wrong" into "which direction to nudge every single weight." Those two numbers have to agree with each other by construction — the gradient is supposed to be the loss function's own slope, not merely a *similarly-shaped* quantity computed alongside it. When a loss and its "gradient" are written as two separately-coded functions instead of one formula differentiated once, it becomes possible for them to quietly drift apart, computing the slope of a *different* function than the one actually being reported as the loss.

### Background

Mean squared error and its gradient are the reduction and element-wise-subtract operations from Chapter 12 and Chapter 14, composed:

```mojo
fn compute_mse_loss(predictions: Matrix, targets: Matrix) -> Float32:
    """L = (1/N) * sum((pred - target)^2), N = every element in the batch"""
    var total: Float32 = 0.0
    for i in range(predictions.size):
        var diff = predictions.data[i] - targets.data[i]
        total += diff * diff
    return total / Float32(predictions.size)

fn mse_loss_gradient(inout grad_out: Matrix, predictions: Matrix, targets: Matrix, batch_size: Int):
    """dL/dPred = (2/N) * (pred - target), N = batch_size (the SAMPLE count, not predictions.size)"""
    var scale = Float32(2.0) / Float32(batch_size)
    for i in range(predictions.size):
        grad_out.data[i] = scale * (predictions.data[i] - targets.data[i])
```

### Worked Example 20.3.1 — The loss and its "gradient," computed independently

One sample, two output units: `predictions = [0.8, 0.3]`, `targets = [1.0, 0.0]`, so `diff = [-0.2, 0.3]`. `compute_mse_loss` sums `diff²` (`0.04 + 0.09 = 0.13`) and divides by `predictions.size = 2` (one sample times two output units): `L = 0.13 / 2 = 0.065`. Differentiating that exact formula, `L = (1/2)·Σdiff²`, with respect to each `pred_i` gives `dL/dpred_i = (2/2)·diff_i = diff_i` — so the *true* analytical gradient here is `[-0.2, 0.3]`. `mse_loss_gradient`, called with `batch_size = 1` (one sample), instead computes `scale = 2/1 = 2.0` and returns `[2.0 × -0.2, 2.0 × 0.3] = [-0.4, 0.6]` — exactly twice the true gradient of the loss `compute_mse_loss` actually reports.

```
[COMMON TRAP]  The reported loss and the gradient that trains the network disagree by a constant factor

compute_mse_loss divides by predictions.size (rows times output units --
here, 2). mse_loss_gradient divides by batch_size (rows ALONE -- here,
1). Whenever there is more than one output unit, these are different
numbers, and the code's gradient ends up scaled by exactly
(predictions.size / batch_size) = output_dim relative to the true
derivative of the value it labels as the loss -- a factor of 2 for
this network's 2-unit output layer, verified above on real numbers.

This does NOT break training: scaling a loss by a positive constant
doesn't change which direction minimizes it, so gradient descent still
moves every weight the same way it would have -- it is exactly
equivalent to having picked a slightly different learning rate. But
the specific "Loss: 0.256953" numbers this chapter's training log
prints are not actually (1/N)*sum((pred-target)^2) for N = every
output element, despite that being compute_mse_loss's own docstring --
they are the correct value of a DIFFERENT, proportional quantity
(effectively output_dim times too large per unit of gradient actually
applied). A loss curve and a gradient computed as two independently
hand-derived functions, rather than one function differentiated once,
is exactly how this kind of drift gets introduced and then silently
ships.
```

## 20.4 The Full Training Step: Forward, Backward, and Update `[FOUNDATIONAL]`

### Intuition

Chapter 17's reverse-mode engine walks a recorded `ComputationGraph` in topological order, looking up each node's registered backward rule by name and letting `accumulate_gradient` handle the bookkeeping. This chapter's training step is the *same chain rule*, applied to the *same kind* of layered computation, with none of that machinery: every layer's backward formula is written out by hand, in a fixed order, against a purpose-built `Matrix` struct that has no `Differentiable` trait, no `GraphNode`, and no registry at all. Both approaches solve an identical mathematical problem — how much does the loss change if this particular weight moves a little — and it's worth watching them arrive at the same kind of answer by genuinely different routes.

### Background

The forward pass runs all five layers in sequence, tracking each layer's pre-activation (`Z`) and post-activation (`A`) output, since the backward pass needs both:

```mojo
X.matmul(W1, Z1); Z1.add_bias(b1); A1.copy_from(Z1); A1.apply_relu()
A1.matmul(W2, Z2); Z2.add_bias(b2); A2.copy_from(Z2); A2.apply_relu()
A2.matmul(W3, Z3); Z3.add_bias(b3); A3.copy_from(Z3); A3.apply_relu()
A3.matmul(W4, Z4); Z4.add_bias(b4); A4.copy_from(Z4); A4.apply_tanh()
A4.matmul(W5, Z5); Z5.add_bias(b5); A5.copy_from(Z5); A5.apply_sigmoid()

var loss = compute_mse_loss(A5, Y)
```

The backward pass then runs in reverse, layer by layer, applying exactly one pattern five times: turn the incoming activation gradient (`dA`) into a pre-activation gradient (`dZ`) by multiplying elementwise with that layer's own activation derivative, then turn `dZ` into that layer's weight and bias gradients (`dW`, `db`) and into the *next* layer back's activation gradient — the manual equivalent of one `chain_rule_step` call per registered op, repeated once per layer instead of once per graph node:

```mojo
# Output layer (sigmoid)
mse_loss_gradient(dA5, A5, Y, batch)
dZ5.apply_sigmoid_derivative(Z5); dZ5.elementwise_multiply(dA5)
A4.transpose(A4_T); A4_T.matmul(dZ5, dW5); dZ5.sum_rows(db5)

# Hidden layer 4 (tanh)
W5.transpose(W5_T); dZ5.matmul(W5_T, dA4)
dZ4.apply_tanh_derivative(Z4); dZ4.elementwise_multiply(dA4)
A3.transpose(A3_T); A3_T.matmul(dZ4, dW4); dZ4.sum_rows(db4)

# Hidden layers 3, 2, 1 (ReLU) -- same pattern, one layer at a time
W4.transpose(W4_T); dZ4.matmul(W4_T, dA3)
dZ3.apply_relu_derivative(Z3); dZ3.elementwise_multiply(dA3)
A2.transpose(A2_T); A2_T.matmul(dZ3, dW3); dZ3.sum_rows(db3)

W3.transpose(W3_T); dZ3.matmul(W3_T, dA2)
dZ2.apply_relu_derivative(Z2); dZ2.elementwise_multiply(dA2)
A1.transpose(A1_T); A1_T.matmul(dZ2, dW2); dZ2.sum_rows(db2)

W2.transpose(W2_T); dZ2.matmul(W2_T, dA1)
dZ1.apply_relu_derivative(Z1); dZ1.elementwise_multiply(dA1)
X.transpose(X_T); X_T.matmul(dZ1, dW1); dZ1.sum_rows(db1)

# Gradient descent: theta = theta - alpha * grad(theta)
for i in range(W1.size): W1.data[i] -= lr * dW1.data[i]
for i in range(b1.size): b1.data[i] -= lr * db1.data[i]
# ... identically for W2/b2 through W5/b5
```

Every `A{n}.transpose(A{n}_T); A{n}_T.matmul(dZ{n+1}, dW{n+1})` step is Chapter 16.3's `MatMulOp.backward` rule (`grad_b = Aᵀ @ grad_output`) written out inline instead of dispatched through a registry; every `dZ{n}.matmul(W{n}_T, dA{n-1})` step is that same rule's other half (`grad_a = grad_output @ Bᵀ`), also inline.

```
[COMMON TRAP]  This network never touches the framework's own autograd engine

Nothing in this chapter constructs a Tensor, records a GraphNode, or
calls chain_rule_step -- despite this being the chapter that Parts 3
and 4's entire autograd engine was built to eventually support. The
Matrix struct above reimplements matmul, transpose, and elementwise
multiply from scratch, and every backward formula is hand-derived and
hand-ordered rather than assembled from AddOp/MulOp/MatMulOp's
registered backward rules and a topological sort. This isn't a bug in
the sense of producing a wrong answer -- the hand-derived chain rule
here is mathematically the same chain rule Chapter 17's reverse pass
automates, and both arrive at correct gradients for their own
computations. It IS a real internal-consistency gap worth naming
directly: a reader who has just finished Parts 3 and 4 expecting to
see ComputationGraph.record and GraphNode.backward reused here will
instead find a second, independent implementation of backpropagation
that happens to look a great deal like the first one.
```

### Worked Example 20.4.1 — A two-layer version of the same pattern, traced completely

The real network is five layers deep across a `500`-sample batch — too large to trace by hand — but the identical pattern holds for a miniature two-layer network on one sample, and every number below can be checked directly. Reuse Worked Example 20.1.3's `X = [1,2]`, `W1 = [[1,0,1],[0,1,1]]`, `b1 = [0,0,0]`: `Z1 = [1,2,3]`, and since every entry is positive, `A1 = relu(Z1) = [1,2,3]` unchanged. A second layer: `W2 = [[1,-1],[0,1],[1,0]]` (`3×2`), `b2 = [0,0]`. `Z2 = A1 @ W2 = [1·1+2·0+3·1,\ 1·(-1)+2·1+3·0] = [4, 1]`. `A2 = sigmoid(Z2) = [sigmoid(4), sigmoid(1)] ≈ [0.98201, 0.73106]`.

With target `Y = [1, 0]`: `diff = [0.98201-1, 0.73106-0] = [-0.01799, 0.73106]`, and `L = (diff[0]² + diff[1]²)/2 ≈ (0.000324 + 0.534449)/2 ≈ 0.26739`.

Backward, with `batch_size=1` (Section 20.3's scale-mismatch trap applies here too — this is `2×` the true gradient of `L`): `dA2 = 2·diff ≈ [-0.03597, 1.46212]`. `sigmoid_derivative(4) ≈ 0.01767`, `sigmoid_derivative(1) ≈ 0.19661`, so `dZ2 = dA2 ⊙ [0.01767, 0.19661] ≈ [-0.00064, 0.28747]`. `dW2 = A1ᵀ @ dZ2` (`A1` as a `3×1` column times `dZ2` as a `1×2` row): row `0` (`A1[0]=1`): `[-0.00064, 0.28747]`; row `1` (`A1[1]=2`): `[-0.00127, 0.57494]`; row `2` (`A1[2]=3`): `[-0.00191, 0.86241]`. `db2 = dZ2` directly (one row, so `sum_rows` changes nothing): `≈ [-0.00064, 0.28747]`.

Continuing back one more layer: `dA1 = dZ2 @ W2ᵀ`, with `W2ᵀ = [[1,0,1],[-1,1,0]]`: `dA1[0] = -0.00064·1 + 0.28747·(-1) ≈ -0.28811`; `dA1[1] = -0.00064·0 + 0.28747·1 ≈ 0.28747`; `dA1[2] = -0.00064·1 + 0.28747·0 ≈ -0.00064`. Since `Z1`'s three entries were all positive, `relu_derivative` is `1` everywhere, so `dZ1 = dA1` unchanged: `≈ [-0.28811, 0.28747, -0.00064]`. Finally `dW1 = Xᵀ @ dZ1` (`X=[1,2]` as a `2×1` column): row `0` (`X[0]=1`): `dZ1` unchanged, `≈ [-0.28811, 0.28747, -0.00064]`; row `1` (`X[1]=2`): `2·dZ1 ≈ [-0.57621, 0.57494, -0.00127]`; and `db1 = dZ1 ≈ [-0.28811, 0.28747, -0.00064]` directly. Every one of these eight numbers (`dW2`'s six entries, `db2`'s two, `dA1`'s three, `dZ1`'s three, `dW1`'s six, `db1`'s three) was produced by exactly the same six operations — `mse_loss_gradient`, an activation derivative, `elementwise_multiply`, `transpose`, `matmul`, `sum_rows` — the real five-layer network calls forty times over instead of twice.

## 20.5 Evaluation Metrics `[FOUNDATIONAL]`

### Intuition

A single "percent correct" number hides two very different kinds of mistakes a classifier can make: crying wolf (predicting positive when the truth is negative) and staying silent when it shouldn't (predicting negative when the truth is positive). A confusion matrix keeps both kinds of error, and both kinds of success, in four separate buckets — true positive, true negative, false positive, false negative — so that precision (of everything flagged positive, how much really was) and recall (of everything that really was positive, how much got flagged) can be reported separately, since a classifier can trade one for the other without changing its overall accuracy at all.

### Background

`pred_class` and `true_class` are each a two-way argmax over one row of the two-unit output layer — the smallest possible instance of Chapter 14.2's `max_reduce_kernel` idea, computed directly with one comparison instead of a reduction loop:

```mojo
struct PerformanceMetrics:
    var tp: Float32
    var tn: Float32
    var fp: Float32
    var fn: Float32

    fn update_metrics(inout self, predictions: Matrix, targets: Matrix):
        for i in range(predictions.rows):
            var pred_class = 1 if predictions.get(i, 1) > predictions.get(i, 0) else 0
            var true_class = 1 if targets.get(i, 1) > targets.get(i, 0) else 0
            if pred_class == 1 and true_class == 1: self.tp += 1.0
            elif pred_class == 0 and true_class == 0: self.tn += 1.0
            elif pred_class == 1 and true_class == 0: self.fp += 1.0
            else: self.fn += 1.0

    fn get_accuracy(self) -> Float32:
        var total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0.0 else 0.0

    fn get_f1_score(self) -> Float32:
        var prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0.0 else 0.0
        var rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0.0 else 0.0
        return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
```

### Worked Example 20.5.1 — A small confusion matrix, every metric computed

`tp=3, tn=2, fp=1, fn=1` (`7` samples total). Accuracy: `(3+2)/7 ≈ 0.7143`. Precision: `3/(3+1) = 0.75`. Recall: `3/(3+1) = 0.75`. F1: `2×0.75×0.75/(0.75+0.75) = 0.75` — precision and recall happen to be equal here (both denominators are `4`), which is exactly why F1 lands on the same value as each of them rather than somewhere between two different numbers, the way it would for an imbalanced pair.

## 20.6 Reference Implementations

The training log below is genuinely captured output — unlike most of this book's newly-composed chapters, this network's source survives with its own `Expected Output` comment block from an actual run, and the numbers are reproduced faithfully rather than invented. One caveat: the author's own source file has a garbled section inside `update_metrics` (an unrelated debug-print loop's code appears mixed into the metrics-accumulation loop, evidently a copy-paste artifact) — the version below reconstructs the clearly-intended, working logic, matching what this chapter's own Section 20.5 already presents, rather than reproducing that corruption verbatim. Separately, a few inline comments in that same source sketch illustrative intermediate values (a sample `A1` after ReLU, a sample `dA5`) that don't arithmetically satisfy this chapter's own formulas for any single consistent set of inputs — they read as rough, by-eye illustrations the author jotted down rather than an exact worked example, so Section 20.4's worked example builds its own small, fully self-consistent trace instead of reusing them.

```mojo
from memory import UnsafePointer
from random import random_float64, seed
from math import sqrt, exp, log, cos, sin

fn relu(x: Float32) -> Float32:
    return x if x > Float32(0.0) else Float32(0.0)

fn relu_derivative(x: Float32) -> Float32:
    return Float32(1.0) if x > Float32(0.0) else Float32(0.0)

fn sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))

fn sigmoid_derivative(x: Float32) -> Float32:
    var s = sigmoid(x)
    return s * (Float32(1.0) - s)

fn tanh_activation(x: Float32) -> Float32:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

fn tanh_derivative(x: Float32) -> Float32:
    var t = tanh_activation(x)
    return Float32(1.0) - t * t

struct Matrix:
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int
    var size: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.data = UnsafePointer[Float32].alloc(self.size)
        for i in range(self.size):
            self.data[i] = 0.0

    fn get(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn set(inout self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn he_init(inout self, fan_in: Int):
        var std_dev = sqrt(Float32(2.0) / Float32(fan_in))
        for i in range(self.size):
            var u1 = Float32(random_float64())
            var u2 = Float32(random_float64())
            var normal = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159 * u2)
            self.data[i] = normal * std_dev

    fn xavier_init(inout self, fan_in: Int, fan_out: Int):
        var limit = sqrt(Float32(6.0) / Float32(fan_in + fan_out))
        for i in range(self.size):
            var r = Float32(random_float64())
            self.data[i] = (r - Float32(0.5)) * Float32(2.0) * limit

    fn matmul(self, other: Matrix, inout result: Matrix):
        for i in range(self.rows):
            for j in range(other.cols):
                var s: Float32 = 0.0
                for k in range(self.cols):
                    s += self.get(i, k) * other.get(k, j)
                result.set(i, j, s)

    fn add_bias(inout self, bias: Matrix):
        for i in range(self.rows):
            for j in range(self.cols):
                var idx = i * self.cols + j
                self.data[idx] = self.data[idx] + bias.data[j]

    fn apply_relu(inout self):
        for i in range(self.size):
            self.data[i] = relu(self.data[i])

    fn apply_relu_derivative(inout self, input_matrix: Matrix):
        for i in range(self.size):
            self.data[i] = relu_derivative(input_matrix.data[i])

    fn apply_tanh(inout self):
        for i in range(self.size):
            self.data[i] = tanh_activation(self.data[i])

    fn apply_tanh_derivative(inout self, input_matrix: Matrix):
        for i in range(self.size):
            self.data[i] = tanh_derivative(input_matrix.data[i])

    fn apply_sigmoid(inout self):
        for i in range(self.size):
            self.data[i] = sigmoid(self.data[i])

    fn apply_sigmoid_derivative(inout self, input_matrix: Matrix):
        for i in range(self.size):
            self.data[i] = sigmoid_derivative(input_matrix.data[i])

    fn elementwise_multiply(inout self, other: Matrix):
        for i in range(self.size):
            self.data[i] = self.data[i] * other.data[i]

    fn transpose(self, inout result: Matrix):
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(j, i, self.get(i, j))

    fn sum_rows(self, inout result: Matrix):
        for j in range(self.cols):
            var s: Float32 = Float32(0.0)
            for i in range(self.rows):
                s += self.get(i, j)
            result.data[j] = s

fn compute_mse_loss(predictions: Matrix, targets: Matrix) -> Float32:
    var total: Float32 = 0.0
    for i in range(predictions.size):
        var diff = predictions.data[i] - targets.data[i]
        total += diff * diff
    return total / Float32(predictions.size)

fn mse_loss_gradient(inout grad_out: Matrix, predictions: Matrix, targets: Matrix, batch_size: Int):
    var scale = Float32(2.0) / Float32(batch_size)
    for i in range(predictions.size):
        grad_out.data[i] = scale * (predictions.data[i] - targets.data[i])

struct PerformanceMetrics:
    var tp: Float32
    var tn: Float32
    var fp: Float32
    var fn: Float32

    fn update_metrics(inout self, predictions: Matrix, targets: Matrix):
        for i in range(predictions.rows):
            var pred_class = 1 if predictions.get(i, 1) > predictions.get(i, 0) else 0
            var true_class = 1 if targets.get(i, 1) > targets.get(i, 0) else 0
            if pred_class == 1 and true_class == 1: self.tp += 1.0
            elif pred_class == 0 and true_class == 0: self.tn += 1.0
            elif pred_class == 1 and true_class == 0: self.fp += 1.0
            else: self.fn += 1.0

    fn get_accuracy(self) -> Float32:
        var total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0.0 else 0.0

    fn get_f1_score(self) -> Float32:
        var prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0.0 else 0.0
        var rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0.0 else 0.0
        return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
```

### Expected Output

The dataset is `500` synthetic samples with a decision boundary that mixes a spiral, an XOR pattern, and a circular boundary, plus `5%` label noise for realism — hard enough that a linear model cannot separate it, which is the whole point of the `4` hidden layers:

```
🧠 Enhanced Mojo Neural Network with 4 Hidden Layers
===================================================
Configuration:
  Dataset size: 500 samples
  Architecture: 2 -> 24 -> 16 -> 12 -> 8 -> 2
  Activations: ReLU (hidden) + Sigmoid (output)
  Learning rate: 0.02 (higher for faster convergence)
  Epochs: 2000
Generated 500 samples with 57% positive class

Training Progress:
------------------
Epoch    0 | Loss: 0.256953
Epoch  100 | Loss: 0.226363
Epoch  200 | Loss: 0.212773
...
Epoch 1900 | Loss: 0.149607
Training Complete!

============================================================
IMPROVED NEURAL NETWORK PERFORMANCE
============================================================
Dataset Size:    500 samples
Accuracy:        77.80%
Precision:       84.80%
Recall:          74.39%
F1-Score:        79.25%

Confusion Matrix:
                 Predicted
                 0    1
Actual    0     177   38
          1      73  212
FAIR performance
============================================================
```

## Chapter Summary

A linear layer's initialization scheme tracks what happens to its output immediately afterward, not its position in the network: He initialization's `std = sqrt(2/fan_in)` compensates for ReLU discarding roughly half its input, while Xavier's `limit = sqrt(6/(fan_in+fan_out))` keeps a saturating activation's input from drifting into its flat regions — verified on `W1`'s `fan_in=2` (the largest standard deviation this network's five layers use) and `W5`'s `fan_in=8, fan_out=2`. Every activation's derivative here — ReLU's hard `0`/`1` mask, sigmoid's `output·(1-output)`, tanh's `1-output²` — is the identical formula Chapter 16.5 already derived for the registered `ReluOp`/`SigmoidOp`/`TanhOp`, reached through a hand-written scalar function instead of a `Differentiable.backward` call. The loss function and its gradient, though, are not as tightly coupled as they look: `compute_mse_loss` divides by every output element while `mse_loss_gradient` divides by the sample count alone, leaving the code's actual training gradient exactly `output_dim` times larger than the true derivative of the number labeled "Loss" — harmless for which direction gradient descent moves in, but a real gap between what the training log reports and what the math backing it actually says, verified directly on `[0.8,0.3]` against `[1.0,0.0]`. The full forward-then-backward chain — traced completely on a two-layer, one-sample miniature of the same pattern — is Chapter 16 and 17's chain rule and reverse-mode traversal, applied entirely by hand against a `Matrix` struct with no `Tensor`, `GraphNode`, or registry anywhere in sight; both routes are mathematically the chain rule, arrived at by genuinely different code. Finally, precision, recall, and F1 come from a confusion matrix fed by the smallest possible instance of Chapter 14.2's argmax idea — a single comparison between two output units per sample — and can diverge sharply from plain accuracy whenever a classifier's two kinds of mistakes aren't equally common.

## Self-Check Questions

1. `W3` in this network has `fan_in = 16`. Compute its He-initialization `std_dev`, and explain in one sentence why a layer with a larger `fan_in` gets a *smaller* standard deviation than `W1`'s.
2. `relu_derivative(0.0)` returns `0.0`, not `1.0`. Where else in this book has this exact same convention, for this exact same reason, already been established?
3. For `predictions = [0.6, 0.9]`, `targets = [0.0, 1.0]`, and `batch_size = 2` (two samples, but only one output row shown here for the arithmetic), compute `compute_mse_loss`'s reported loss and `mse_loss_gradient`'s returned gradient. By what factor do they disagree with the loss's own true analytical derivative, and does that factor match the pattern Section 20.3 established?
4. In Worked Example 20.4.1's two-layer trace, `dZ1 = dA1` exactly, with no scaling at all. Why — what specific fact about `Z1`'s three values makes this true, and would it still be true if one of `Z1`'s entries had been negative?
5. A confusion matrix has `tp=8, tn=1, fp=1, fn=0`. Compute accuracy, precision, recall, and F1, and explain why accuracy alone would make this classifier look far better than precision and recall reveal it to be.

## Where We Go Next

Chapter 21 (`part6/02-advanced-features.md`) extends this network with the pieces production frameworks add on top: custom autograd functions for operations that don't fit a standard registry (the `CustomFunction` framework from Chapter 16.7, applied to a real training pipeline instead of a bisection toy example), higher-order derivatives, model serialization, and the debugging tools that catch a wrong gradient — like Section 20.3's loss/gradient scale mismatch — before it silently corrupts a training run instead of merely rescaling one.

## Worked Solutions

**1.** `std_dev = sqrt(2/16) = sqrt(0.125) ≈ 0.3536` — noticeably smaller than `W1`'s `1.0`. A layer with more inputs sums more terms into each output value before any nonlinearity is applied, so each individual weight needs to contribute less on average to keep the *total* variance of that sum from growing simply because there are more terms being added together.

**2.** Chapter 16.5's `ReluOp.backward`, which builds its mask with `greater_than_zero_mask(inputs[0])` — a strict `>` comparison that assigns `0`, not `1`, to an input of exactly `0.0`. Both places make the same choice for the same underlying reason: ReLU's true derivative is undefined at exactly `x=0` (a corner, not a smooth slope), so any implementation has to pick one of the two one-sided derivatives as a subgradient, and both this chapter's `relu_derivative` and Chapter 16.5's `ReluOp` pick `0`.

**3.** `diff = [0.6-0.0, 0.9-1.0] = [0.6, -0.1]`. `compute_mse_loss`: `size=2`, `total = 0.6² + (-0.1)² = 0.36+0.01=0.37`, `L = 0.37/2 = 0.185`. True analytical gradient of that `L`: `dL/dpred_i = (2/2)·diff_i = diff_i = [0.6, -0.1]`. `mse_loss_gradient` with `batch_size=2`: `scale = 2/2 = 1.0`, so it returns `[1.0×0.6, 1.0×(-0.1)] = [0.6, -0.1]` — here, the two agree exactly, with no factor of disagreement at all! This is not a contradiction of Section 20.3's finding: the mismatch factor is `predictions.size / batch_size = output_dim`, and with `batch_size=2` passed in for what is genuinely a `2`-row, `1`-column-per-row shape (`predictions.size=2`, `output_dim=1` here), the factor is `2/2=1` — the trap only produces a visible discrepancy when `output_dim > 1`, exactly as it does for this chapter's real `2`-output-unit network.

**4.** `Z1 = [1, 2, 3]` — every entry strictly positive, so `relu_derivative` returns `1.0` for all three positions, and multiplying `dA1` elementwise by a vector of all `1`s leaves it completely unchanged: `dZ1 = dA1 ⊙ [1,1,1] = dA1`. This would NOT still hold if any entry of `Z1` were negative or zero: `relu_derivative` would return `0.0` at that position, and the corresponding entry of `dZ1` would be forced to exactly `0`, regardless of what `dA1` held there — the same "zeroed on the way forward, zeroed again on the way back, for a different reason each time" behavior Chapter 16's Worked Example 16.5.1 traced directly for `ReluOp`.

**5.** Accuracy: `(8+1)/10 = 0.90`. Precision: `8/(8+1) ≈ 0.889`. Recall: `8/(8+0) = 1.0`. F1: `2×0.889×1.0/(0.889+1.0) ≈ 0.941`. All four numbers actually look reasonably strong here — but the *scenario* to watch for is a heavily imbalanced dataset where `tn` dominates: if instead `tp=1, tn=90, fp=0, fn=9` (predicting "negative" almost every time on a dataset that's `90%` negative), accuracy would be `91/100=0.91` — misleadingly high — while recall would be a dismal `1/10=0.10`, revealing that the classifier is essentially failing to find positives at all. Accuracy alone can't distinguish "a genuinely strong classifier" from "a classifier that has learned to exploit a skewed class balance," which is exactly why precision and recall are reported separately rather than folded into one number.
