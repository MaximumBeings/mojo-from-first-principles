# Chapter 11: Neural Network Layers

Parts 1 through 5 built a general-purpose tensor and autograd framework; Part 6 specializes it into the one application every AD framework is ultimately measured against — training a neural network. This chapter builds a 5-layer network (`2 → 24 → 16 → 12 → 8 → 2`) entirely from the primitives already in the book: `Matrix` from [Chapter 4](../part2/02-matrix-operations.md), the chain rule from [Chapter 7](../part4/01-backward-function-implementation.md), and gradient accumulation from [Chapter 8](../part4/02-gradient-computation-engine.md).

## 11.1 Linear Layer Implementation

A linear (fully-connected) layer is `Z = X @ W + b` — the matmul from Chapter 4 plus a broadcast bias add from Chapter 3. The one new piece is *initialization*: a layer feeding into ReLU wants He initialization (scaled for the fact that ReLU zeroes out half its inputs on average), while a layer feeding into a saturating activation like tanh or sigmoid wants Xavier initialization (scaled to keep activations from vanishing or exploding as the signal passes through several layers).

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
        # C[i,j] = sum_k A[i,k] * B[k,j]  -- Chapter 4's matmul, reused verbatim
        for i in range(self.rows):
            for j in range(other.cols):
                var s: Float32 = 0.0
                for k in range(self.cols):
                    s += self.get(i, k) * other.get(k, j)
                result.set(i, j, s)

    fn add_bias(inout self, bias: Matrix):
        """Z = XW + b -- every row gets the same bias vector (Chapter 3.4 broadcasting)."""
        for i in range(self.rows):
            for j in range(self.cols):
                var idx = i * self.cols + j
                self.data[idx] = self.data[idx] + bias.data[j]
```

The network's five layers wire straight from this: `W1: [2,24]` He-initialized (feeds ReLU), `W2: [24,16]` and `W3: [16,12]` likewise, `W4: [12,8]` Xavier-initialized (feeds tanh), `W5: [8,2]` Xavier-initialized (feeds sigmoid) — the initialization scheme tracks the activation *downstream* of each weight matrix, not the layer's position in the network.

## 11.2 Activation Functions

Every activation is implemented alongside its derivative, since the backward pass in Section 11.4 needs both:

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

Applied element-wise across a whole layer's activations with the SIMD-friendly loop pattern from [Chapter 10](../part5/02-performance-optimization.md#101-simd-vectorization):

```mojo
fn apply_relu(inout self: Matrix):
    for i in range(self.size):
        self.data[i] = relu(self.data[i])
```

## 11.3 Loss Functions

Mean squared error and its gradient are the reduction and element-wise-subtract operations from Part 2, composed:

```mojo
fn compute_mse_loss(predictions: Matrix, targets: Matrix) -> Float32:
    """L = (1/N) * sum((pred - target)^2)"""
    var total: Float32 = 0.0
    for i in range(predictions.size):
        var diff = predictions.data[i] - targets.data[i]
        total += diff * diff
    return total / Float32(predictions.size)

fn mse_loss_gradient(inout grad_out: Matrix, predictions: Matrix, targets: Matrix, batch_size: Int):
    """dL/dPred = (2/N) * (pred - target)"""
    var scale = Float32(2.0) / Float32(batch_size)
    for i in range(predictions.size):
        grad_out.data[i] = scale * (predictions.data[i] - targets.data[i])
```

## 11.4 Optimizer Framework

The full training step chains everything above: a forward pass through all five layers, MSE loss, a backward pass applying the chain rule layer by layer (exactly the reverse traversal from [Chapter 8](../part4/02-gradient-computation-engine.md#81-reverse-mode-ad-implementation)), and a plain gradient-descent parameter update `theta = theta - alpha * grad(theta)`:

```mojo
# Forward pass, layer by layer
X.matmul(W1, Z1); Z1.add_bias(b1); A1.copy_from(Z1); A1.apply_relu()
A1.matmul(W2, Z2); Z2.add_bias(b2); A2.copy_from(Z2); A2.apply_relu()
A2.matmul(W3, Z3); Z3.add_bias(b3); A3.copy_from(Z3); A3.apply_relu()
A3.matmul(W4, Z4); Z4.add_bias(b4); A4.copy_from(Z4); A4.apply_tanh()
A4.matmul(W5, Z5); Z5.add_bias(b5); A5.copy_from(Z5); A5.apply_sigmoid()

var loss = compute_mse_loss(A5, Y)

# Backward pass -- output layer
mse_loss_gradient(dA5, A5, Y, batch)
dZ5.apply_sigmoid_derivative(Z5); dZ5.elementwise_multiply(dA5)
A4.transpose(A4_T); A4_T.matmul(dZ5, dW5); dZ5.sum_rows(db5)

# Backward pass -- hidden layer 4 (tanh)
W5.transpose(W5_T); dZ5.matmul(W5_T, dA4)
dZ4.apply_tanh_derivative(Z4); dZ4.elementwise_multiply(dA4)
A3.transpose(A3_T); A3_T.matmul(dZ4, dW4); dZ4.sum_rows(db4)

# Backward pass -- hidden layers 3, 2, 1 (ReLU) -- same pattern, one layer at a time
W4.transpose(W4_T); dZ4.matmul(W4_T, dA3)
dZ3.apply_relu_derivative(Z3); dZ3.elementwise_multiply(dA3)
A2.transpose(A2_T); A2_T.matmul(dZ3, dW3); dZ3.sum_rows(db3)

W3.transpose(W3_T); dZ3.matmul(W3_T, dA2)
dZ2.apply_relu_derivative(Z2); dZ2.elementwise_multiply(dA2)
A1.transpose(A1_T); A1_T.matmul(dZ2, dW2); dZ2.sum_rows(db2)

W2.transpose(W2_T); dZ2.matmul(W2_T, dA1)
dZ1.apply_relu_derivative(Z1); dZ1.elementwise_multiply(dA1)
X.transpose(X_T); X_T.matmul(dZ1, dW1); dZ1.sum_rows(db1)

# Gradient descent update: theta = theta - alpha * grad(theta)
for i in range(W1.size): W1.data[i] -= lr * dW1.data[i]
for i in range(b1.size): b1.data[i] -= lr * db1.data[i]
for i in range(W2.size): W2.data[i] -= lr * dW2.data[i]
for i in range(b2.size): b2.data[i] -= lr * db2.data[i]
for i in range(W3.size): W3.data[i] -= lr * dW3.data[i]
for i in range(b3.size): b3.data[i] -= lr * db3.data[i]
for i in range(W4.size): W4.data[i] -= lr * dW4.data[i]
for i in range(b4.size): b4.data[i] -= lr * db4.data[i]
for i in range(W5.size): W5.data[i] -= lr * dW5.data[i]
for i in range(b5.size): b5.data[i] -= lr * db5.data[i]
```

The dataset is 500 synthetic samples with a decision boundary that mixes a spiral, an XOR pattern, and a circular boundary, plus 5% label noise for realism — hard enough that a linear model cannot separate it, which is the whole point of the 4 hidden layers.

### Expected Output

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

Precision, recall, and F1 come from the same confusion-matrix bookkeeping used anywhere classification quality is measured — true positives, true negatives, false positives, false negatives — computed via the argmax over the two-unit output layer described in [Chapter 5.2](../part2/03-reduction-operations.md#52-minmax-operations):

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

Chapter 12 extends this network with the pieces production frameworks add on top: custom autograd functions for operations that don't fit the standard registry, higher-order derivatives, model serialization, and the debugging tools that catch a wrong gradient before it silently corrupts a training run.
