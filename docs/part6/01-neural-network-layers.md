# Chapter 11: Neural Network Layers

A neural network is a composition of the tensor operations already built: affine transforms, nonlinearities, reductions, and parameter updates. This chapter keeps each component numerically checkable and uses stable formulas suitable for a modern implementation.

## 11.1 Linear layers compose matmul and broadcast

For a batch `X`, weights `W`, and bias `b`, a linear layer computes `Z=X@W+b`. The bias is broadcast across batch rows.

```mojo
def linear(x: Tensor, weight: Tensor, bias: Tensor) raises -> Tensor:
    var out = matmul(x, weight)
    return broadcast_add_2d(out, bias)
```

**Manual worked example.** With `x=[2,3]`, `W=[[1,4],[2,5]]`, and `b=[0.5,-1]`, matmul gives `[2×1+3×2,2×4+3×5]=[8,23]`; bias gives `[8.5,22]`.

## 11.2 Activations need stable formulas

ReLU is a branch. Sigmoid should avoid evaluating a huge positive exponential when `x` is very negative; the two algebraically equivalent branches keep exponent arguments nonpositive.

```mojo
def stable_sigmoid(x: Float32) -> Float32:
    if x >= 0:
        return 1 / (1 + exp(-x))
    var z = exp(x)
    return z / (1 + z)

def relu(x: Float32) -> Float32:
    return x if x > 0 else 0
```

**Manual worked example.** `sigmoid(0)=1/(1+1)=0.5`; `relu(-2)=0` and `relu(3)=3`. For `x=-100`, the negative branch evaluates `exp(-100)`, a tiny safe value, instead of `exp(100)`, which may overflow.

## 11.3 Classification uses log-sum-exp

Softmax followed by cross-entropy should be implemented as one log-sum-exp expression. Subtracting the maximum logit preserves the exact probabilities while avoiding overflow.

```mojo
def cross_entropy_two(logit0: Float64, logit1: Float64, target: Int) raises -> Float64:
    if target < 0 or target > 1:
        raise Error("target must be 0 or 1")
    var maximum = max(logit0, logit1)
    var log_z = maximum + log(exp(logit0 - maximum) + exp(logit1 - maximum))
    return log_z - (logit0 if target == 0 else logit1)
```

**Manual worked example.** For logits `[2,1]` and target 0, maximum is 2 and `log_z=2+log(1+e^-1)≈2.31326`. Loss is `2.31326-2=0.31326`. Direct softmax gives class-0 probability `e²/(e²+e¹)≈0.73106`, and `-log(0.73106)=0.31326`.

## 11.4 AdamW separates momentum from weight decay

AdamW maintains first and second moment estimates, bias-corrects them, and applies decoupled weight decay. The scalar update below is the per-parameter rule a tensor kernel vectorizes.

```mojo
def adamw_step(
    mut weight: Float64, mut first: Float64, mut second: Float64,
    grad: Float64, step: Int, learning_rate: Float64,
    beta1: Float64 = 0.9, beta2: Float64 = 0.999,
    epsilon: Float64 = 1e-8, decay: Float64 = 0.01,
):
    first = beta1 * first + (1 - beta1) * grad
    second = beta2 * second + (1 - beta2) * grad * grad
    var m_hat = first / (1 - beta1 ** step)
    var v_hat = second / (1 - beta2 ** step)
    weight -= learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + decay * weight)
```

**Manual worked example.** At step 1 with weight 2, gradient 0.5, zero moments, learning rate 0.1, and no decay: first moment is 0.05, second is 0.00025, bias-corrected values are 0.5 and 0.25. The normalized update is `0.1×0.5/sqrt(0.25)=0.1`, so weight becomes approximately 1.9.

## 11.5 Training checks the smallest overfit case

Before reporting a large benchmark, train on a tiny deterministic batch and require the loss to fall. This checks forward values, backward gradients, and the optimizer together.

```text
test protocol:
1. Fix the random seed.
2. Train on four examples until the model can overfit them.
3. Require finite loss and gradients at every step.
4. Run centered gradient checks on a sampled set of parameters.
5. Only then benchmark a larger dataset.
```

**Manual worked example.** If one scalar prediction is 0.8 for target 1, binary cross-entropy is `-log(0.8)≈0.2231`. A correct update should raise that prediction and lower the loss; a move to 0.9 lowers it to `-log(0.9)≈0.1054`.

This chapter avoids invented epoch logs: numerical results belong to a versioned example, fixed seed, hardware record, and reproducible command.
