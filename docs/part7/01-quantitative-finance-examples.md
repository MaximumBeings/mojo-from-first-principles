# Chapter 13: Quantitative Finance Examples

Part 7 is where this framework earns its "Financial Computing Ready" design principle from the book's introduction. Every primitive is reused as-is: the Struct-of-Arrays layout from Part 0.3 prices a whole bond portfolio in parallel, the reduction kernels from Chapter 5 aggregate it into portfolio-level risk, the custom-function framework from Chapter 12 differentiates through a numerical solver, and the GPU kernel design from Part 5 is what makes doing this for thousands of instruments at once practical.

## 13.1 Bond Pricing with Automatic Differentiation

A zero-coupon bond pays a single fixed amount at maturity and nothing before it, so its price is one discounting formula: `PV = FaceValue · e^(-yield · time)`. Pricing a portfolio of them is an embarrassingly parallel problem — every bond's price is independent of every other bond's — which makes it the cleanest possible demonstration of the SoA-plus-GPU-kernel pattern from Chapters 9 and 2:

```mojo
struct ZeroCouponBondSystemSoA:
    """SoA layout (Chapter 9.2): one contiguous array per field,
    not one struct per bond -- coalesced reads across the whole portfolio."""
    var face_value: UnsafePointer[Scalar[DType.float32]]
    var time_to_maturity: UnsafePointer[Scalar[DType.float32]]
    var risk_free_rate: UnsafePointer[Scalar[DType.float32]]
    var credit_spread: UnsafePointer[Scalar[DType.float32]]
    var present_value: UnsafePointer[Scalar[DType.float32]]
    var yield_to_maturity: UnsafePointer[Scalar[DType.float32]]
    var duration: UnsafePointer[Scalar[DType.float32]]
    var portfolio_weight: UnsafePointer[Scalar[DType.float32]]
    var num_bonds: Int
```

### Step 1: Computing Bond Prices

```mojo
fn compute_bond_prices_kernel(
    present_value: UnsafePointer[Scalar[DType.float32]],
    yield_to_maturity: UnsafePointer[Scalar[DType.float32]],
    duration: UnsafePointer[Scalar[DType.float32]],
    face_value: UnsafePointer[Scalar[DType.float32]],
    time_to_maturity: UnsafePointer[Scalar[DType.float32]],
    risk_free_rate: UnsafePointer[Scalar[DType.float32]],
    credit_spread: UnsafePointer[Scalar[DType.float32]],
    num_bonds: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_bonds:
        var total_yield = risk_free_rate[idx] + credit_spread[idx]
        yield_to_maturity[idx] = total_yield
        # Continuous compounding: PV = FV * e^(-r*t)
        present_value[idx] = face_value[idx] * exp(-total_yield * time_to_maturity[idx])
        # A zero-coupon bond's Macaulay duration equals its time to maturity exactly --
        # there's only one cash flow, so it's also the *only* cash flow's weighted time.
        duration[idx] = time_to_maturity[idx]
```

Aggregating 1,024 bonds into a total portfolio value is the tree-reduction pattern from [Chapter 5.1](../part2/03-reduction-operations.md#51-sum-and-mean-reductions), applied to `present_value` instead of a loss:

```mojo
var num_blocks = (NUM_BONDS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
ctx.enqueue_function[compute_bond_prices_kernel](...)
ctx.synchronize()

ctx.enqueue_function[sum_reduce_kernel](partial_sums, bond_system.present_value, NUM_BONDS, ...)
var total_portfolio_value = tensor_sum(partial_sums, reduction_threads)
```

### Expected Output

```
=== STRUCT OF ARRAYS (SoA) ZERO COUPON BOND SYSTEM ===
Number of bonds: 1024
Threads per block: 64

=== MEMORY LAYOUT COMPARISON ===
Struct of Arrays (SoA) - This implementation:
  Memory: face:[f0,f1,f2...] maturity:[m0,m1,m2...] ...
  Access: Coalesced, optimal bandwidth

=== COMPUTING BOND PRICES ===
Total Portfolio Value: $ 847213.5

=== FINANCIAL ANALYSIS ===
Portfolio Analytics:
  Weighted Average Yield: 3.51 %
  Weighted Average Maturity: 14.87 years
  Portfolio Duration: 14.87 years
  Interest Rate Risk: 1% rate increase -> ~14.87 % price decline
```

Because every gradient rule in this book flows through registered ops (Chapter 7), `present_value`'s gradient with respect to `risk_free_rate` — the bond's **DV01**, its dollar sensitivity to a one-basis-point rate move — falls straight out of `backward()` with no separate finite-difference calculation: `d(PV)/d(yield) = -time_to_maturity * PV`, the derivative of the exponential discount factor, exactly the `ExpOp.backward` rule from [Chapter 7.2](../part4/01-backward-function-implementation.md#72-element-wise-operation-gradients) applied to a financial input instead of a neural-network activation.

## 13.2 Credit Spread and Risk Analytics

A *risky* bond (one with default risk) needs an extra yield above the risk-free curve to compensate a buyer for that risk — the **Z-spread**. Unlike the zero-coupon case, a coupon-paying bond's price is a sum of discounted cash flows with no closed form for the spread that reprices it to a given market price, so this is solved numerically:

```mojo
alias ISSUE_PRICE = 98.0
alias RISK_FREE_RATE = 0.03
alias COUPON_RATE = 0.03
alias NOTIONAL = 100.0
alias TOTAL_PAYMENTS = 8     # 2 years, quarterly

fn calculate_bond_price(spread: Float64) -> Float64:
    """Market Price = sum_{n=1..N} CF_n / (1 + (r+s)/m)^(n)"""
    var discounted_value = Float64(0.0)
    var payments_per_year = Float64(4.0)
    for x in range(1, TOTAL_PAYMENTS + 1):
        var coupon_payment = (3.0 / 12.0) * COUPON_RATE * NOTIONAL
        var discount_factor = power(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, Float64(x))
        discounted_value += coupon_payment / discount_factor
    var final_discount_factor = power(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, Float64(TOTAL_PAYMENTS))
    discounted_value += NOTIONAL / final_discount_factor
    return discounted_value

fn objective_function(spread: Float64) -> Float64:
    return calculate_bond_price(spread) - ISSUE_PRICE

fn bisection_method(a: Float64, b: Float64, tolerance: Float64) -> Float64:
    var left = a; var right = b; var iterations = 0
    while abs(right - left) > tolerance and iterations < 100:
        var mid = (left + right) / 2.0
        if abs(objective_function(mid)) < tolerance:
            return mid
        elif objective_function(mid) * objective_function(left) < 0:
            right = mid
        else:
            left = mid
        iterations += 1
    return (left + right) / 2.0
```

### Expected Output

```
=== Z-SPREAD CALCULATION FOR RISKY BONDS ===
Bond Parameters:
Issue Price: 98.0
Maturity: 2 years
Risk-free rate: 0.03
Coupon rate: 0.03
Notional: 100.0
Total payments: 8

Market Price with Zero Spread: 99.99999999999996

Solving for z-spread using bisection method...

=== RESULTS ===
The zSpread on a Risky Bond is:
0.010460522770881654

The Yield To Maturity on the Bond:
0.04046052277088165

=== VERIFICATION ===
Target market price: 98.0
Calculated price with optimal spread: 98.00000040566188
Difference: 4.0566187919921504e-07
```

104.6 basis points of z-spread over the risk-free curve is this bond's compensation for credit risk — the framework's autodiff makes this differentiable, not just solvable: `ZSpreadSolve.backward` from [Chapter 12.1](../part6/02-advanced-features.md#121-custom-autograd-functions) turns `d(spread)/d(market_price)` into an ordinary gradient the optimizer in Chapter 11 can use, which is what "Greeks via automatic differentiation" means in practice — sensitivities that would otherwise need a separate closed-form derivation for every new instrument type instead come for free from the same backward pass that trains a neural network.

## 13.3 Portfolio Optimization

`portfolio_weight[i] = present_value[i] / total_portfolio_value` turns individual bond prices into portfolio weights with one more elementwise-divide kernel; portfolio duration — the risk metric a desk actually manages against — is the weight-duration inner product, which is a multiply followed by the same sum-reduction used to total portfolio value in Section 13.1:

```mojo
fn compute_portfolio_duration_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    duration: UnsafePointer[Scalar[DType.float32]],
    portfolio_weight: UnsafePointer[Scalar[DType.float32]],
    num_bonds: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_bonds:
        output[idx] = duration[idx] * portfolio_weight[idx]   # weighted contribution
# followed by sum_reduce_kernel over `output` -> portfolio duration
```

Because this whole pipeline — price, weight, weighted duration, total — is built from registered, differentiable ops, `backward()` from a target portfolio duration produces the gradient of that duration with respect to every bond's face value, directly usable by a rebalancing optimizer deciding how much of each bond to hold to hit a duration target, without deriving the sensitivity by hand for each new candidate portfolio.

## 13.4 Monte Carlo Simulations with Gradients

Monte Carlo pricing values a path-dependent instrument by averaging its discounted payoff across many simulated price paths — an expectation approximated by a large sample mean, which is exactly `tensor_mean` from [Chapter 5.1](../part2/03-reduction-operations.md#51-sum-and-mean-reductions) applied to a payoff computed per path:

```mojo
fn simulate_gbm_paths(
    paths: UnsafePointer[Scalar[DType.float32]],   # [num_paths] terminal prices
    s0: Float32, mu: Float32, sigma: Float32, dt: Float32, num_steps: Int,
    num_paths: Int, seed_base: Int,
):
    """Geometric Brownian motion: S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_paths:
        var s = s0
        for step in range(num_steps):
            var z = sample_standard_normal(seed_base + idx * num_steps + step)  # Box-Muller, Chapter 11.1
            s *= exp((mu - sigma * sigma / Float32(2.0)) * dt + sigma * sqrt(dt) * z)
        paths[idx] = s

fn monte_carlo_option_price(
    terminal_prices: UnsafePointer[Scalar[DType.float32]], strike: Float32,
    risk_free_rate: Float32, maturity: Float32, num_paths: Int,
) -> Float32:
    var payoffs = UnsafePointer[Scalar[DType.float32]].alloc(num_paths)
    for i in range(num_paths):
        var payoff = terminal_prices[i] - strike
        payoffs[i] = payoff if payoff > Float32(0.0) else Float32(0.0)   # call option payoff
    var mean_payoff = tensor_mean(payoffs, num_paths)
    payoffs.free()
    return mean_payoff * exp(-risk_free_rate * maturity)   # discount back to present value
```

Run this simulation as a node the graph records rather than a one-off calculation, and `backward()` from the resulting option price differentiates straight through the discounting, the payoff, and the simulated paths themselves — producing Delta (`d(price)/d(s0)`), Vega (`d(price)/d(sigma)`), and Rho (`d(price)/d(risk_free_rate)`) as ordinary gradients from the same reverse pass built in Part 4, rather than the finite-difference "bump and reprice" every one of those sensitivities traditionally requires — re-running the entire simulation once per Greek. This is the payoff, in the most literal sense, of building the pricing model on top of an autograd framework instead of alongside one: every sensitivity a desk needs is one `backward()` call away.

That closes the book's arc: Part 0 taught the language, Parts 1–4 built a tensor and autograd engine, Part 5 made it fast, Part 6 proved it on a neural network, and Part 7 has now proven the same machinery on the domain the framework was designed for from its very first design principle — financial computing, where "differentiable" and "auditable" have to mean the same thing.
