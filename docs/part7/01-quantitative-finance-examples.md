# Chapter 22: Quantitative Finance Examples — Where a Wrong Gradient Costs Real Money

> "A model that merely prices an instrument is a calculator. A model that *differentiates* through pricing it is a risk desk — the difference is whether every sensitivity a trader needs comes free with the price, or has to be derived by hand, once, for every new instrument."

## What you will understand by the end of this chapter

- How the same `PV = FV·e^(-yield·time)` formula that prices one zero-coupon bond scales into a 1,024-bond GPU portfolio through the Struct-of-Arrays pattern from Chapter 3 and Chapter 18 — and a real, confirmed bug in how that portfolio's total gets summed that silently drops 87.5% of the book from the reported number.
- Why a coupon-paying bond's z-spread has no closed-form solution and has to be found by bisection instead — verified in this chapter to seven decimal places against a real captured run, not an invented one.
- Why portfolio duration is nothing more than a weighted average, and how this book's registered ops turn "how much does this rebalancing hurt if rates rise" into an ordinary gradient instead of a hand-derived formula for every new portfolio.
- Why Monte Carlo pricing needs many simulated paths rather than one, and why running the simulation as a single graph node turns every option Greek into one `backward()` call instead of one full re-simulation per sensitivity.
- Why "differentiable" and "auditable" turn out to be the same requirement once real money is involved — illustrated by this chapter's own two independently-confirmed bugs, found the same way every other bug in this book was found: by checking a claimed number against the code that supposedly produced it.

## What you need to know first

- Exponentials and their gradient rule: Chapter 12.3 (Power and Exponential Functions) for `exp`, and `ExpOp.backward`'s output-reuse trick from Chapter 16.2 — the bond pricing formula's derivative is that exact rule applied to a discount factor instead of an activation.
- Struct-of-Arrays memory layout from Chapter 3.3, and Chapter 18.2's memory-coalescing analysis of this exact eight-field `ZeroCouponBondSystemSoA` struct.
- The multi-round reduction pattern from Chapter 14.1 — specifically the `while current_size > 1` requirement, because this chapter's own kernel usage is what happens when that requirement is skipped.
- `backward()` from Chapter 17, and the implicit-function-theorem `CustomFunction` pattern from Chapter 21.1 — the z-spread solver reuses it verbatim on the same bond.
- `tensor_mean` from Chapter 14.1 and ordinary elementwise ops from Chapter 12 — Monte Carlo pricing is built from nothing more exotic than those two.

## 22.1 Bond Pricing with Automatic Differentiation `[FOUNDATIONAL]`

### Intuition

A zero-coupon bond is the simplest possible IOU: pay less today for a promise to receive a fixed, larger amount on a fixed future date, with nothing in between. The gap between what you pay and what you're promised is rent — paid partly for the pure inconvenience of waiting (the risk-free rate) and partly as compensation for the chance the promise doesn't get kept (credit spread). `PV = FV·e^(-yield·time)` is just that rent, applied continuously: the longer the wait or the shakier the promise, the more today's price shrinks relative to the payoff at the end.

### Background

Pricing one bond is one exponential. Pricing a portfolio of a thousand of them is a thousand *independent* exponentials — every bond's price depends on nothing but its own four numbers — which is exactly the embarrassingly-parallel shape a GPU kernel wants, and exactly why this struct is Struct-of-Arrays (Chapter 3.3) rather than one record per bond: a kernel that reads every bond's `risk_free_rate` wants those values contiguous, not scattered one field into each of a thousand separate records, and Chapter 18.2 already measured that difference on this exact struct as an `8×` reduction in memory transactions.

```mojo
struct ZeroCouponBondSystemSoA:
    """SoA layout (Chapter 3.3): one contiguous array per field,
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

Aggregating a portfolio into a total value is the tree-reduction pattern from [Chapter 14.1](../part2/03-reduction-operations.md#51-sum-and-mean-reductions), applied to `present_value` instead of a loss:

```mojo
var num_blocks = (NUM_BONDS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
ctx.enqueue_function[compute_bond_prices_kernel](...)
ctx.synchronize()

ctx.enqueue_function[sum_reduce_kernel](partial_sums, bond_system.present_value, NUM_BONDS, ...)
var total_portfolio_value = tensor_sum(partial_sums, reduction_threads)
```

### Worked Example 22.1.1 — Three real bonds, priced by hand

The book's own bond-generation logic (`face_choices = [1000.0, 5000.0, 10000.0]` cycling by index, `time_to_maturity = 0.25 + (i mod 120)×0.25`, `risk_free_rate = 0.02 + (i mod 31)×0.001`, `credit_spread = 0.001 + (i mod 30)×0.001`) is fully deterministic — no randomness anywhere — so its first three bonds can be priced by hand and checked against exactly what the kernel would compute:

| Bond | Face | Maturity | Risk-free | Spread | Yield | Discount factor `e^(-yield·t)` | Present Value |
|---|---|---|---|---|---|---|---|
| 0 | \$1,000 | 0.25 yr | 2.0% | 0.10% | 2.10% | `e^(-0.00525) ≈ 0.994764` | \$994.7638 |
| 1 | \$5,000 | 0.50 yr | 2.1% | 0.20% | 2.30% | `e^(-0.01150) ≈ 0.988566` | \$4,942.8294 |
| 2 | \$10,000 | 0.75 yr | 2.2% | 0.30% | 2.50% | `e^(-0.01875) ≈ 0.981425` | \$9,814.2469 |

Each row is the same one-line formula three times over with different inputs — this is what "embarrassingly parallel" means concretely: bond 1's price needs nothing from bond 0's or bond 2's.

### Worked Example 22.1.2 — DV01 from `backward()`, not from a second pricing run

Bond 2's DV01 — its dollar sensitivity to a one-basis-point rise in its own yield — is `d(PV)/d(yield) = -time_to_maturity × PV`, exactly `ExpOp.backward`'s output-reuse rule (Chapter 16.2) applied to this discount factor: `-0.75 × 9814.2469 ≈ -7360.6852` per unit of yield, or `-7360.6852 × 0.0001 ≈ -\$0.7361` per basis point. Checking that analytic formula the way every backward rule in this book has been checked — central finite difference, `ε=10⁻⁶` — gives `-7360.685158...`, matching to eight significant figures. The framework never re-prices the bond at a bumped yield to get this number; it falls straight out of the same `backward()` pass that would already be running to train a neural network, applied here to a financial input instead of an activation.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| sum_reduce_kernel (Chapter 14.1) is DESIGNED to run inside a      |
| `while current_size > 1` loop, halving the array once per round   |
| until one value remains. The driver code above launches it        |
| exactly ONCE, with only `reduction_threads = min(THREADS_PER_     |
| BLOCK, NUM_BONDS // 2)` threads -- for NUM_BONDS=1024 and          |
| THREADS_PER_BLOCK=64, that's 64 threads, each combining ONE pair   |
| of adjacent bonds. 64 threads x 2 bonds per thread = 128 bonds     |
| actually read. The other 896 bonds -- 87.5% of the portfolio --    |
| are never touched, and total_portfolio_value silently reports the |
| sum of the first 128 bonds as if it were the whole book, with no  |
| error, no warning, and no size check anywhere that would catch it.|
+------------------------------------------------------------------+
```

### Expected Output

Independently reconstructing this portfolio's numbers from the bond-generation logic above turns up a real discrepancy worth stating plainly. Summing all 1,024 bonds correctly gives a total portfolio value of **≈\$2,831,177**, a weighted average yield of **≈4.82%**, and a weighted average maturity (equal to portfolio duration for zero-coupon bonds) of **≈11.19 years**. Running the reduction exactly as the code above is actually written — the single-launch, 128-bond version the `[COMMON TRAP]` above describes — gives a total of **≈\$364,131** and, because `compute_portfolio_weights_kernel` still divides every one of the 1,024 bonds' present values by that truncated total, portfolio weights that sum to **≈7.78** instead of `1.0` (a single-number sanity check that would have caught this bug immediately), a nonsensical weighted "yield" of **≈37.4%**, and a nonsensical weighted "maturity" of **≈87.0 years**. Neither of those two honestly-computed outcomes matches the numbers a previous pass of this book reported for this section (a portfolio value of \$847,213.50, a 3.51% weighted yield, and a 14.87-year weighted maturity and duration) — those figures do not reproduce from this deterministic bond data under either the correct reduction or the reduction as literally coded, which means they were never actually captured output and are corrected here rather than carried forward. This is a genuinely different situation from Section 22.2 below, where every digit *does* reproduce exactly.

## 22.2 Credit Spread and Risk Analytics `[FOUNDATIONAL]`

### Intuition

Imagine two friends each ask to borrow money and promise to pay it back in two years. One has never missed a payment in their life; the other has a spottier record. You'd lend to the first at whatever the "going rate" is — call it the risk-free rate. The second one has to offer *more* to get the same loan, because you need extra compensation for the real chance they don't pay you back in full. That extra amount, expressed as yield above the risk-free rate, is the Z-spread — and unlike the first friend's loan, there's no simple formula for exactly how much extra is enough; you have to solve for the number that makes the deal fair given the price the market is actually charging.

### Background

A *risky* bond's price is a sum of several discounted cash flows, each one bent by the *same* unknown spread, so there's no algebraic way to isolate that spread the way `PV = FV·e^(-yield·time)` isolates a zero-coupon bond's yield. Bisection finds it instead: guess a spread, price the bond, compare the price to the market's actual price, and narrow the guess toward whichever half of the bracket contains the answer.

| | Zero-coupon yield (Section 22.1) | Z-spread (this section) |
|---|---|---|
| Cash flows | One, at maturity | Several, one per coupon plus principal |
| Solvable how | Closed form: rearrange `PV=FV·e^(-y·t)` for `y` | Numerically: bisection on the pricing formula |
| Differentiable how | Ordinary chain rule through `exp` | Implicit function theorem — Chapter 21.1's `CustomFunction` pattern |

```mojo
alias ISSUE_PRICE = 98.0
alias RISK_FREE_RATE = 0.03
alias COUPON_RATE = 0.03
alias NOTIONAL = 100.0
alias TOTAL_PAYMENTS = 8     # 2 years, quarterly
alias TOLERANCE = 1e-8

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

### Worked Example 22.2.1 — Solving a real bond's z-spread, verified to seven decimal places

A 2-year, quarterly, 3%-coupon bond with \$100 notional trading at \$98.00 against a 3% risk-free curve: bisection over `[-0.1, 0.1]` at `TOLERANCE=1e-8` converges in **25 iterations** to a spread of `0.010460522770881654` — 104.60523 basis points — for a yield to maturity of `4.046052277088165%`. Pricing the bond back at that spread gives `98.00000040566188`, a difference from the \$98.00 target of `4.0566187919921504 × 10⁻⁷` — both numbers reproduce exactly, to every digit shown, from re-running this bisection independently. The bond's undiscounted cash flows are simple to check by hand: eight quarterly coupons of `(3/12)×0.03×100 = \$0.75` each (`\$6.00` total) plus a `\$100` principal repayment, `\$106.00` undiscounted — every dollar of which gets discounted a little less than the risk-free curve alone would, because the market is demanding 104.6 basis points of extra compensation to hold this particular bond instead of a Treasury.

`ZSpreadSolve.backward` from [Chapter 21.1](../part6/02-advanced-features.md#121-custom-autograd-functions) turns `d(spread)/d(market_price)` into an ordinary gradient — Chapter 21.1's own worked example computed exactly this derivative for this exact bond (`≈-0.0052912`) and confirmed it against an independent re-solve — which is what "Greeks via automatic differentiation" means in practice: a sensitivity that would otherwise need a separate closed-form derivation for every new instrument type instead comes free from the same backward pass that trains a neural network.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| bisection_method's loop condition is                              |
| `while abs(right-left) > tolerance and iterations < 100`.          |
| The `iterations < 100` half of that condition is a silent escape   |
| hatch: if a caller ever passes a tolerance small enough (or a      |
| bracket wide enough) that convergence would genuinely need more    |
| than 100 halvings, the loop exits anyway and returns whatever      |
| midpoint it has -- with no error, no flag, and no way for the      |
| caller to tell "converged" apart from "gave up at iteration 100."  |
| This bond converges comfortably in 25 iterations, well under the   |
| cap, but nothing in the function's return value would reveal it if |
| it hadn't.                                                         |
+------------------------------------------------------------------+
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

Unlike Section 22.1's, this block is genuinely real: independently re-running `calculate_bond_price` and `bisection_method` exactly as written above, with no changes, reproduces every one of these digits.

## 22.3 Portfolio Optimization `[FOUNDATIONAL]`

### Intuition

Think of a seesaw with several weights placed at different distances from the center. Each weight's contribution to how hard the whole board tips isn't just its own weight — it's weight *times* distance from the fulcrum. A bond far from "now" (a long maturity) is a heavy weight sitting far out on the plank: even a modest position size can dominate the portfolio's overall tip when rates move, exactly the way bond C dominates the example below despite being the smallest position.

### Background

Portfolio weight is a bond's share of total value; portfolio duration is the weighted average of individual durations — the same weight-times-distance-summed-together idea as the seesaw, computed as one elementwise multiply followed by the same sum-reduction used to total portfolio value in Section 22.1:

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

### Worked Example 22.3.1 — A clean three-bond portfolio

| Bond | Present Value | Time to Maturity (duration) |
|---|---|---|
| A | \$400 | 2 years |
| B | \$350 | 5 years |
| C | \$250 | 10 years |

Total portfolio value: `400 + 350 + 250 = 1000`. Weights: `w_A = 0.40`, `w_B = 0.35`, `w_C = 0.25` — summing to `1.0`, as portfolio weights always must. Portfolio duration: `0.40×2 + 0.35×5 + 0.25×10 = 0.8 + 1.75 + 2.5 = 5.05` years. Read that the way a trading desk does: a 1% parallel rise in rates should cost this portfolio roughly 5.05%, or about \$50.50 on the \$1000 book — with bond C, despite being the *smallest* position, contributing the *most* to that risk (`2.5` of the `5.05` total), because its 10-year maturity makes it far more rate-sensitive per dollar than A or B.

### Worked Example 22.3.2 — The same math, on Section 22.1's actual bonds

Section 22.1's three real bonds (present values \$994.7638, \$4,942.8294, \$9,814.2469; durations 0.25, 0.50, 0.75 years) sum to a real total of `\$15,751.8400`. Weights: `w_0 = 994.7638/15751.8400 ≈ 0.06315`, `w_1 = 4942.8294/15751.8400 ≈ 0.31379`, `w_2 = 9814.2469/15751.8400 ≈ 0.62305` — again summing to `1.0`. Portfolio duration: `0.06315×0.25 + 0.31379×0.50 + 0.62305×0.75 ≈ 0.01579 + 0.15690 + 0.46729 ≈ 0.63998` years — a very short-duration book, because these three bonds all mature within a year, unlike Worked Example 22.3.1's deliberately longer-dated illustration.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| The weights in both worked examples above sum to exactly 1.0 --   |
| that is not a coincidence, it is the single cheapest correctness  |
| check this whole pipeline has. Section 22.1's [COMMON TRAP] showed |
| the real system's single-launch sum_reduce_kernel producing        |
| portfolio weights that sum to ~7.78 instead of 1.0 for the full    |
| 1,024-bond book. Anyone computing THIS section's weighted duration |
| downstream of that bug would get a number built on weights that    |
| don't sum to one -- and the bug would have been caught immediately |
| by the same one-line check (`sum(weights) == 1.0`) that both       |
| worked examples above quietly satisfy.                              |
+------------------------------------------------------------------+
```

Because this whole pipeline — price, weight, weighted duration, total — is built from registered, differentiable ops, `backward()` from a target portfolio duration produces the gradient of that duration with respect to every bond's face value, directly usable by a rebalancing algorithm deciding how much of each bond to hold to hit a duration target — the same kind of parameter update Chapter 20.4 performs during training, applied here to a portfolio instead of a network's weights.

## 22.4 Monte Carlo Simulations with Gradients `[FOUNDATIONAL]`

### Intuition

Ask one friend to guess how a coin flip sequence will go, and you learn nothing reliable. Ask ten thousand friends to each simulate their own sequence and average their answers, and the average converges to the true odds — not because any one guess was right, but because the errors in individual guesses cancel out across enough of them. Monte Carlo option pricing is that averaging trick applied to "what will this stock be worth in a year": simulate many independent possible futures, price the option on each one, and average.

### Background

| | Closed-form (Black-Scholes-style) | Monte Carlo (this section) |
|---|---|---|
| When it applies | Payoff has a known analytic solution | Any payoff, including path-dependent ones with no closed form |
| Cost | One formula evaluation | Many simulated paths, more for a tighter estimate |
| Greeks | Differentiate the formula once, by hand | `backward()` through the whole simulation, once, for every Greek at once |

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
            var z = sample_standard_normal(seed_base + idx * num_steps + step)  # Box-Muller, Chapter 20.1
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

### Worked Example 22.4.1 — Five paths, one call option, priced by hand

A stock starting at \$100 is simulated forward and five independent paths land at terminal prices `[95, 102, 108, 130, 90]`. Pricing a call option with strike `K=100` — a contract paying `max(S_T - K, 0)`: payoffs are `[0, 2, 8, 30, 0]`, mean payoff `(0+2+8+30+0)/5 = 8`. Discounting at a 3% risk-free rate over one year, `e^(-0.03) ≈ 0.970446`, gives an option price of `8 × 0.970446 ≈ 7.7636`. Every path that finished below the strike contributed a hard `0`, never a negative number — the option's entire value comes from the upside paths, which is exactly why averaging over many paths, rather than trusting one "expected" path, is necessary to price an asymmetric payoff correctly.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Estimating a Greek by "bump and reprice" -- re-run the ENTIRE     |
| simulation with s0 nudged up slightly, subtract the original      |
| price, divide by the bump -- is a well-known trap when each rerun |
| draws FRESH random paths: the difference between two independently|
| sampled Monte Carlo estimates is dominated by sampling noise, not  |
| by the actual sensitivity, unless both runs reuse the exact same   |
| underlying random numbers ("common random numbers"). backward()   |
| through a single recorded simulation sidesteps this entirely -- it|
| differentiates the ONE set of paths that was actually simulated,  |
| producing Delta, Vega, and Rho from that one run with no re-       |
| sampling and therefore no extra noise to control for at all.       |
+------------------------------------------------------------------+
```

Run this simulation as a node the graph records rather than a one-off calculation, and `backward()` from the resulting option price differentiates straight through the discounting, the payoff, and the simulated paths themselves — producing Delta (`d(price)/d(s0)`), Vega (`d(price)/d(sigma)`), and Rho (`d(price)/d(risk_free_rate)`) as ordinary gradients from the same reverse pass built in Part 4, rather than the finite-difference "bump and reprice" every one of those sensitivities traditionally requires. This is the payoff, in the most literal sense, of building the pricing model on top of an autograd framework instead of alongside one: every sensitivity a desk needs is one `backward()` call away.

## 22.5 Reference Implementations

```mojo
# ── 22.1 Bond Pricing (Struct-of-Arrays) ────────────────────────
struct ZeroCouponBondSystemSoA:
    var face_value: UnsafePointer[Scalar[DType.float32]]
    var time_to_maturity: UnsafePointer[Scalar[DType.float32]]
    var risk_free_rate: UnsafePointer[Scalar[DType.float32]]
    var credit_spread: UnsafePointer[Scalar[DType.float32]]
    var present_value: UnsafePointer[Scalar[DType.float32]]
    var yield_to_maturity: UnsafePointer[Scalar[DType.float32]]
    var duration: UnsafePointer[Scalar[DType.float32]]
    var portfolio_weight: UnsafePointer[Scalar[DType.float32]]
    var num_bonds: Int

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
        present_value[idx] = face_value[idx] * exp(-total_yield * time_to_maturity[idx])
        duration[idx] = time_to_maturity[idx]

fn compute_portfolio_weights_kernel(
    portfolio_weight: UnsafePointer[Scalar[DType.float32]],
    present_value: UnsafePointer[Scalar[DType.float32]],
    total_portfolio_value: Float32,
    num_bonds: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_bonds:
        if total_portfolio_value > Float32(0.0):
            portfolio_weight[idx] = present_value[idx] / total_portfolio_value
        else:
            portfolio_weight[idx] = Float32(0.0)

fn compute_portfolio_duration_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    duration: UnsafePointer[Scalar[DType.float32]],
    portfolio_weight: UnsafePointer[Scalar[DType.float32]],
    num_bonds: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_bonds:
        output[idx] = duration[idx] * portfolio_weight[idx]

# ── 22.2 Credit Spread (Z-spread) ────────────────────────────────
alias ISSUE_PRICE = 98.0
alias RISK_FREE_RATE = 0.03
alias COUPON_RATE = 0.03
alias NOTIONAL = 100.0
alias TOTAL_PAYMENTS = 8
alias TOLERANCE = 1e-8

fn calculate_bond_price(spread: Float64) -> Float64:
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

# ── 22.4 Monte Carlo Simulation ──────────────────────────────────
fn simulate_gbm_paths(
    paths: UnsafePointer[Scalar[DType.float32]],
    s0: Float32, mu: Float32, sigma: Float32, dt: Float32, num_steps: Int,
    num_paths: Int, seed_base: Int,
):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < num_paths:
        var s = s0
        for step in range(num_steps):
            var z = sample_standard_normal(seed_base + idx * num_steps + step)
            s *= exp((mu - sigma * sigma / Float32(2.0)) * dt + sigma * sqrt(dt) * z)
        paths[idx] = s

fn monte_carlo_option_price(
    terminal_prices: UnsafePointer[Scalar[DType.float32]], strike: Float32,
    risk_free_rate: Float32, maturity: Float32, num_paths: Int,
) -> Float32:
    var payoffs = UnsafePointer[Scalar[DType.float32]].alloc(num_paths)
    for i in range(num_paths):
        var payoff = terminal_prices[i] - strike
        payoffs[i] = payoff if payoff > Float32(0.0) else Float32(0.0)
    var mean_payoff = tensor_mean(payoffs, num_paths)
    payoffs.free()
    return mean_payoff * exp(-risk_free_rate * maturity)
```

### Expected Output

This chapter's four sections have three different honest provenances, and they are not interchangeable. Section 22.2's z-spread output block is genuinely real: independently re-running `calculate_bond_price` and `bisection_method` exactly as written, with no changes, reproduces `0.010460522770881654`, `98.00000040566188`, and `4.0566187919921504e-07` to every digit shown. Section 22.1's portfolio-level aggregate figures are not real captured output — the deterministic bond-generation data reproduces neither a correctly-summed total nor the total the code's actual single-launch reduction bug would produce, so the honestly-recomputed alternatives in that section's own "Expected Output" replace an earlier, unreproducible set of numbers rather than repeating them. Section 22.4's Monte Carlo example was never claimed as captured output at all — five paths and their payoffs are a hand-constructed illustration, verified arithmetically in Worked Example 22.4.1, not a GPU run.

## Chapter Summary

This closing chapter pointed the whole framework at the domain its first design principle named: financial computing, where a wrong number isn't a lower accuracy score but a mispriced position. Section 22.1 priced a bond portfolio with the same exponential discounting formula this book has used since Chapter 12, and along the way found a real, confirmed bug — `sum_reduce_kernel` launched once instead of in the multi-round loop Chapter 14.1 itself established, silently dropping 87.5% of a 1,024-bond portfolio from its own reported total. Section 22.2 solved a genuinely unsolvable-in-closed-form problem (a coupon bond's z-spread) by bisection, and unlike Section 22.1, its numbers are real: they reproduce to every digit shown. Section 22.3 showed that portfolio duration is nothing more than a weighted average, and that the one-line sanity check `sum(weights) == 1.0` would have caught Section 22.1's bug immediately had anyone run it. Section 22.4 showed why Monte Carlo pricing needs many paths rather than one, and why differentiating through a single simulation is not just faster than "bump and reprice" for each Greek — it avoids a real, well-documented Monte Carlo pitfall (fresh random paths per bump) entirely.

## Self-Check Questions

1. Section 22.1's `[COMMON TRAP]` describes a single-launch reduction that reads only `2 × min(THREADS_PER_BLOCK, NUM_BONDS // 2)` bonds. For a larger portfolio of `NUM_BONDS = 2048` with the same `THREADS_PER_BLOCK = 64`, how many bonds would actually get summed, and what fraction of the portfolio does that leave out?
2. Section 22.2's bisection runs for exactly 25 iterations to reach `TOLERANCE = 1e-8` starting from bracket `[-0.1, 0.1]` (width `0.2`). Derive the minimum number of halvings needed to shrink a width-`0.2` bracket below `1e-8`, and confirm it matches the 25 actually observed.
3. Add a fourth bond D (\$500 present value, 3-year duration) to Worked Example 22.3.1's three-bond portfolio. Compute the new total, all four weights, and the new portfolio duration.
4. Using Worked Example 22.1.2's method, compute bond 1's DV01 (from Worked Example 22.1.1: face \$5,000, maturity 0.5 years, yield 2.30%, present value \$4,942.8294).
5. Worked Example 22.4.1 prices a call option (`payoff = max(S_T - K, 0)`) on paths `[95, 102, 108, 130, 90]` with `K=100`. Reprice a *put* option (`payoff = max(K - S_T, 0)`) on the same five paths and the same strike, and explain in one sentence why a put's value comes from a different subset of the paths than a call's does.

## Where We Go Next

There is no Chapter 23 — Part 7 closes the book's arc. Part 0 taught the language; Parts 1 through 4 built a tensor and autograd engine from first principles; Part 5 made it fast; Part 6 proved it on a neural network and gave it the escape hatches (custom functions, higher-order derivatives, serialization, debugging tools) a trustworthy framework needs; and Part 7 has now proven the same machinery on the domain the very first design principle named — financial computing, where "differentiable" and "auditable" have to mean the same thing, and where this chapter's own two independently-confirmed bugs are the proof that checking, not assuming, is what "auditable" actually requires in practice. The Appendix's Practice Quiz is the natural next stop for anyone who wants to check what stuck.

## Worked Solutions

**1.** `reduction_threads = min(64, 2048 // 2) = min(64, 1024) = 64` — unchanged from the 1,024-bond case, because `THREADS_PER_BLOCK` is still the smaller of the two. Bonds actually summed: `2 × 64 = 128`, exactly as before. Fraction left out: `(2048 - 128) / 2048 = 1920/2048 ≈ 93.75%` — *worse* than the 1,024-bond portfolio's 87.5%, because the number of bonds actually touched by this bug is capped at `2 × THREADS_PER_BLOCK` regardless of how large the portfolio grows, so the fraction silently dropped only gets worse as the book scales up.

**2.** A bracket of width `w` needs `n` halvings to shrink below tolerance `tol` when `w / 2ⁿ ≤ tol`, i.e. `n ≥ log₂(w/tol)`. Here `w=0.2`, `tol=10⁻⁸`: `n ≥ log₂(0.2 / 10⁻⁸) = log₂(2×10⁷) ≈ 24.25`, and since `n` must be a whole number of halvings, `n = 25` — matching the 25 iterations actually observed exactly, because bisection's convergence rate is deterministic and depends only on the starting width and the tolerance, not on where the root happens to sit inside the bracket.

**3.** New total: `400 + 350 + 250 + 500 = 1500`. New weights: `w_A = 400/1500 ≈ 0.2667`, `w_B = 350/1500 ≈ 0.2333`, `w_C = 250/1500 ≈ 0.1667`, `w_D = 500/1500 ≈ 0.3333` — summing to `1.0`. New portfolio duration: `0.2667×2 + 0.2333×5 + 0.1667×10 + 0.3333×3 ≈ 0.5333 + 1.1667 + 1.6667 + 1.0 = 4.3667` years — lower than the original `5.05` years, because bond D's short 3-year duration and its large 33% weight both pull the weighted average down, even though the total portfolio grew.

**4.** `DV01 = -time_to_maturity × PV × 0.0001 = -0.5 × 4942.8294 × 0.0001 ≈ -\$0.2471` per basis point — smaller in magnitude than bond 2's `-\$0.7361` from Worked Example 22.1.2, consistent with bond 1's shorter maturity (0.5 years versus 0.75) making it less sensitive to a yield move, exactly the same "longer maturity means more rate-sensitive per dollar" relationship Worked Example 22.3.1 established for bond C.

**5.** Put payoffs on `[95, 102, 108, 130, 90]` with `K=100`: `max(100-95,0)=5`, `max(100-102,0)=0`, `max(100-108,0)=0`, `max(100-130,0)=0`, `max(100-90,0)=10` → `[5, 0, 0, 0, 10]`. Mean payoff: `(5+0+0+0+10)/5 = 3`. Discounted at the same `e^(-0.03) ≈ 0.970446`: `3 × 0.970446 ≈ 2.9113`. A call's value comes entirely from paths that finished *above* the strike (102, 108, 130 here); a put's value comes entirely from paths that finished *below* it (95 and 90 here) — the two option types are mirror images of the same asymmetric-payoff idea, each one paying off on the opposite side of the strike from the other.
