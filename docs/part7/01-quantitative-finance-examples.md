# Chapter 13: Quantitative Finance Examples

Financial examples are useful because units, signs, and independent formulas make errors visible. The calculations here are educational validation cases, not investment advice or production valuation models; real desks add calendars, day-count conventions, curves, settlement, credit events, and model governance.

## 13.1 Zero-coupon pricing and DV01

Under continuous compounding, a zero-coupon bond price is `P=F·exp(-yT)`. Its yield derivative is `dP/dy=-T·P`. A positive DV01 is conventionally the approximate price loss for a one-basis-point yield rise: `DV01=-dP/dy×10⁻⁴=T·P×10⁻⁴`.

```mojo
@fieldwise_init
struct ZeroCouponResult(Copyable, Movable):
    var price: Float64
    var dprice_dyield: Float64
    var dv01: Float64

def price_zero(face: Float64, annual_yield: Float64, years: Float64) -> ZeroCouponResult:
    var price = face * exp(-annual_yield * years)
    var derivative = -years * price
    return ZeroCouponResult(price, derivative, -derivative * 1e-4)
```

**Manual worked example.** For face 100, yield 5%, and maturity 2 years, `P=100e^-0.1≈90.48374`. Derivative is `-2×90.48374≈-180.96748` dollars per unit yield, so DV01 is `180.96748×0.0001≈$0.01810`. A one-basis-point rise should lower price by about 1.81 cents.

## 13.2 Z-spread via a bracketed solver

A z-spread adds one constant spread to each discount rate so discounted cash flows match market price. Bisection is slower than Newton's method but preserves a bracket and cannot diverge when the objective is continuous and changes sign.

```mojo
def two_year_price(spread: Float64) -> Float64:
    var discount = 1.03 + spread
    return 4 / discount + 104 / (discount * discount)

def solve_spread(market: Float64, mut left: Float64, mut right: Float64) raises -> Float64:
    var f_left = two_year_price(left) - market
    if f_left * (two_year_price(right) - market) > 0:
        raise Error("spread root is not bracketed")
    for _ in range(80):
        var middle = (left + right) / 2
        var f_middle = two_year_price(middle) - market
        if f_left * f_middle <= 0:
            right = middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2
```

**Manual worked example.** A 2-year annual 4% coupon bond pays 4 after year 1 and 104 after year 2. At market price 98 and base rate 3%, bisection solves `4/(1.03+s)+104/(1.03+s)²=98`, giving `s≈0.0207678`, or 207.68 basis points. Substitution returns 98 to rounding.

## 13.3 The implicit derivative of a calibrated spread

If `f(s,P)=model_price(s)-P=0`, then `ds/dP=1/(d model_price/ds)`. The derivative is negative: a higher observed price implies a lower spread.

```mojo
def spread_price_sensitivity(spread: Float64) -> Float64:
    var discount = 1.03 + spread
    var dmodel_dspread = -4 / (discount * discount) - 208 / (discount * discount * discount)
    return 1 / dmodel_dspread
```

**Manual worked example.** At `s≈0.0207678`, `dmodel/ds≈-182.90745`, so `ds/dP≈-0.00546725` spread units per dollar. A one-cent higher price therefore lowers calibrated spread by about `0.00005467`, or 0.547 basis points.

## 13.4 Portfolio duration is value-weighted

For small parallel yield shifts, portfolio modified duration is the market-value-weighted average of component durations. Weights must sum to one and should be derived from current values, not face amounts.

```mojo
def weighted_duration(values: List[Float64], durations: List[Float64]) raises -> Float64:
    if len(values) != len(durations) or len(values) == 0:
        raise Error("duration inputs are empty or mismatched")
    var total = compensated_sum(values)
    var result = 0.0
    for i in range(len(values)):
        result += values[i] / total * durations[i]
    return result
```

**Manual worked example.** Values `[400,350,250]` sum to 1,000, so weights are `[0.40,0.35,0.25]`. Durations `[2,5,10]` give `0.4×2+0.35×5+0.25×10=0.8+1.75+2.5=5.05` years. A 1% rise predicts an approximate 5.05%, or $50.50, loss on the $1,000 portfolio.

## 13.5 Monte Carlo price, uncertainty, and pathwise Delta

A Monte Carlo estimate is incomplete without sampling uncertainty. For a European call under geometric Brownian motion, use discounted payoffs for price and the pathwise derivative `1(ST>K)·ST/S0` for Delta. Production implementations use reproducible counter-based random streams, antithetic pairs, and often control variates.

```mojo
def call_payoff(terminal: Float64, strike: Float64) -> Float64:
    return max(terminal - strike, 0)

def call_pathwise_delta(terminal: Float64, spot: Float64, strike: Float64) -> Float64:
    return terminal / spot if terminal > strike else 0
```

**Manual worked example.** With terminal prices `[95,102,108,130,90]`, strike 100, and spot 100, payoffs are `[0,2,8,30,0]`, mean 8. At 3% for one year, price is `8e^-0.03≈7.7636`. Delta samples are `[0,1.02,1.08,1.30,0]`; their discounted mean is `0.68e^-0.03≈0.6601`.

## 13.6 Report a confidence interval

For independent payoff samples, estimated standard error is sample standard deviation divided by the square root of path count. A rough 95% interval is estimate ±1.96 standard errors; variance reduction changes the variance estimator and must be reflected in the report.

```mojo
def confidence_half_width(sample_std: Float64, paths: Int) raises -> Float64:
    if paths < 2:
        raise Error("at least two paths are required")
    return 1.96 * sample_std / sqrt(Float64(paths))
```

**Manual worked example.** If discounted payoff standard deviation is 12 and 10,000 paths are independent, standard error is `12/sqrt(10000)=0.12`; the 95% half-width is `1.96×0.12≈0.2352`. Report a price of 7.76 as roughly `[7.53,8.00]`, not as false machine precision.

The same discipline closes the book: derive the scalar quantity, verify it with concrete numbers and units, express its derivative, then scale the computation across tensors and GPUs without changing the contract.
