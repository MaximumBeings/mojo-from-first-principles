# Appendix B: Practice Quiz

Use these questions as retrieval practice. Work each numerical answer on paper before opening the explanation; the point is to reconstruct the invariant, not recognize a phrase.

## Foundations

These questions check Mojo 1.0 syntax, ownership, shape, and layout.

1. Which keyword declares all functions in Mojo 1.0?
2. Why must a GPU kernel length argument use `Int32` or another fixed-width type instead of `Int`?
3. What are the row-major strides for shape `[2,3,4]`?
4. What flat offset corresponds to coordinate `[1,2,3]` for those strides?
5. What stride represents a broadcast axis that rereads one physical value?

??? success "Answers"
    1. `def`.
    2. Host and device platform-sized integers may have different widths; a fixed width makes the ABI unambiguous.
    3. `[12,4,1]`.
    4. `1×12+2×4+3=23`.
    5. Zero.

## Autograd

These questions check graph identity, reverse traversal, and gradient accumulation.

1. For `w=x*y+x` at `x=3`, `y=4`, what are `dw/dx` and `dw/dy`?
2. Why does the tape store value IDs instead of copied tensor values inside nodes?
3. Why must gradient updates use `+=` rather than assignment?
4. Can the first-order tape in Chapters 6–8 compute second derivatives as written?

??? success "Answers"
    1. `dw/dx=y+1=5`; `dw/dy=x=3`.
    2. Repeated uses then route to one value and one gradient slot, independent of copying or pointer aliasing.
    3. A value can reach the loss along multiple paths; the multivariable chain rule sums their contributions.
    4. No. Backward operations must themselves be recorded, or the framework needs forward-over-reverse support.

## Numerics and performance

These questions check stable reductions, launch geometry, and benchmark interpretation.

1. How many 256-thread blocks cover 1,000 elements, and how many threads take the tail guard?
2. A length-10 loop uses SIMD width 4. How many elements are vectorized and how many are scalar tail work?
3. Why is `mean(x²)-mean(x)²` risky for variance?
4. Why is unsynchronized host enqueue time not a GPU kernel benchmark?

??? success "Answers"
    1. Four blocks launch 1,024 threads; 24 are outside the logical range.
    2. Eight vectorized elements and two tail elements.
    3. Subtracting two large, nearly equal values can lose the small variance to cancellation; Welford's algorithm is more stable.
    4. Enqueue is asynchronous and may finish long before device work completes; time device events or synchronized boundaries.

## Finance

These questions check signs, units, and independent financial calculations.

1. A continuously compounded zero has price `P=F·exp(-yT)`. What is `dP/dy`?
2. If price is 90.48 and maturity is 2 years, what is positive DV01 approximately?
3. Why is calibrated spread sensitivity to market price negative?
4. What information must accompany a Monte Carlo price?

??? success "Answers"
    1. `-T·P`.
    2. `T·P×10⁻⁴=2×90.48×0.0001≈$0.01810`.
    3. Higher price requires less discounting, hence a lower spread.
    4. At minimum: path count, seed/stream policy, estimator, standard error or confidence interval, model parameters, and implementation version.
