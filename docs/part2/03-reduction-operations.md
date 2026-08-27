# Chapter 14: Reduction Operations — Collapsing Many Values Into One

> "Every loss function in this book ends the same way: a tensor of per-example errors collapses into a single scalar a training loop can actually act on. Every operation before this chapter preserved shape — the same number of positions went in as came out. A reduction is the first place that stops being true on purpose, and it's also the first place where doing it the parallel-obvious way — every thread adds its value into one shared accumulator — is a race condition, not an optimization."

**What you will understand by the end of this chapter:**

- The tree-reduction pattern — pair, combine, halve, repeat — traced by hand on `[1, 4, 9, 16]` round by round until one value remains, and why every pair *within* a round is independent while rounds themselves must run in strict sequence
- Why `max_reduce_kernel` carries an index buffer alongside its value buffer, and how that surviving index becomes the one position the backward pass's gradient flows through, with every other position receiving exactly zero
- The L2 norm as sum-of-squares-then-square-root, and gradient-norm clipping as the one place in this book where a norm's *value* changes what happens next rather than just measuring something
- Variance and standard deviation as sum-and-mean applied twice — once for the mean itself, once for the mean of squared deviations from it — traced on a full eight-value dataset down to an exact variance of `4` and standard deviation of `2`
- Why `tensor_sum`'s own per-round kernel launch is written as a comment rather than working code, and what that means for trusting the `30` this chapter's own text says it returns

**What you need to know first:**

- Chapter 4 and Chapter 12 (kernel launch mechanics and the one-thread-per-position pattern — `sum_reduce_kernel` and `max_reduce_kernel` both modify that pattern so that one thread now reads *two* input positions and writes one output position, instead of one-to-one)
- Chapter 11.2 and 11.3 (the allocate-then-free discipline `tensor_sum`'s scratch buffer and `l2_norm`'s squared-values buffer both follow — allocate once up front, free exactly once at the end, the same shape as the arena and pool patterns those sections built)
- Chapter 5 (why floating-point addition isn't perfectly associative — the fact this chapter's whole tree-reduction discipline exists to make parallel-safe, rather than just fast)

<a id="51-sum-and-mean-reductions"></a>
## 14.1 Sum and Mean Reductions `[FOUNDATIONAL]`

### Intuition

A straight-line loop that adds every element into one running total is easy to write and easy to get wrong in parallel: if a thousand GPU threads all try to add into that same total simultaneously, most of those additions are lost — two threads reading the current total, both adding their own value, and both writing back overwrite each other, so only one addition out of every colliding pair actually survives. **Tree reduction** avoids this by never letting two threads touch the same memory at the same time: pair up elements and combine each pair, then pair up *those* results and combine again, halving the array's length each round until one value remains. Every combine within a single round reads and writes entirely disjoint memory, so a round's worth of pairs really can run at once — the discipline is only that one round has to finish completely before the next round starts, since round `N+1`'s inputs are round `N`'s outputs.

### Background

```mojo
fn sum_reduce_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    """One round of pairwise reduction: size elements -> ceil(size/2)."""
    var tid = Int(thread_idx.x)
    if tid * 2 + 1 < size:
        output[tid] = input[tid * 2] + input[tid * 2 + 1]
    elif tid * 2 < size:
        output[tid] = input[tid * 2]      # odd leftover, pass through
    else:
        output[tid] = Float32(0.0)

fn tensor_sum(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    """Repeated halving until a single scalar remains."""
    var current = input
    var current_size = size
    var scratch = UnsafePointer[Scalar[DType.float32]].alloc(size)
    while current_size > 1:
        var next_size = (current_size + 1) // 2
        # launch sum_reduce_kernel with current_size threads, writing into scratch
        current = scratch
        current_size = next_size
    var result = current[0]
    scratch.free()
    return result

fn tensor_mean(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    return tensor_sum(input, size) / Float32(size)
```

`sum_reduce_kernel` is one round: thread `tid` combines input positions `2·tid` and `2·tid+1`, unless `2·tid` is the very last element of an odd-length input, in which case it passes that lone value through unchanged (the `elif` branch) — an explicit `else` branch even zero-fills any thread that has no work at all, so every output position this round writes gets a defined value, never leftover memory from a previous round.

### Worked Example 14.1.1 — Tree reduction on `[1, 4, 9, 16]`, round by round

```
Round 0: [ 1,  4,  9, 16]
           \  /    \  /
Round 1: [  5   ,   25  ]
             \       /
Round 2: [     30      ]
```

Round `1` pairs `(1,4)→5` and `(9,16)→25`. Round `2` pairs `(5,25)→30`. Three additions total — the same count a straight-line loop would perform — but the two additions inside round `1` are fully independent (thread `0` touches only `input[0]` and `input[1]`; thread `1` touches only `input[2]` and `input[3]`), so a GPU can run both at once, then run round `2`'s single addition once both of round `1`'s results exist. `tensor_mean` divides the final sum by the count: `30 / 4 = 7.5`.

### Worked Example 14.1.2 — An odd-length array across three rounds

`[2, 6, 3, 8, 5]` (five elements, forcing the odd-leftover branch). Round `0`: `next_size = (5+1)//2 = 3`. Thread `0`: `2+6=8`. Thread `1`: `3+8=11`. Thread `2`: `tid*2=4`, which is `< 5` but `tid*2+1=5` is not — the leftover branch fires, passing `input[4]=5` straight through. Round `0` output: `[8, 11, 5]`.

Round `1`: `current_size=3`, `next_size=(3+1)//2=2`. Thread `0`: `8+11=19`. Thread `1`: `tid*2=2 < 3` but `tid*2+1=3` is not — leftover branch again, passing `5` straight through. Round `1` output: `[19, 5]`.

Round `2`: `current_size=2`, `next_size=1`. Thread `0`: `19+5=24`. Final result: `24` — matching a direct sum of the original five values, `2+6+3+8+5=24`, exactly.

```
[COMMON TRAP]  tensor_sum's own kernel launch is a comment, not code

Read the while loop in tensor_sum literally: each iteration computes
next_size, then the very next line is a comment --
"# launch sum_reduce_kernel with current_size threads, writing into
scratch" -- describing what should happen, not code that makes it
happen. No enqueue_function call, no synchronize, nothing writes a
single value into scratch at any point. The very next line, `current =
scratch`, executes unconditionally regardless, and scratch was
allocated with UnsafePointer.alloc(size) -- which reserves memory
without zeroing or initializing it. Taken completely literally, this
function's `while` loop only ever reassigns two local variables; by the
time it exits, `current` points at scratch, and `scratch` still holds
whatever bytes happened to already be in that memory, not a reduced
sum. `result = current[0]` would then return uninitialized garbage, not
`30`.

The `30` this chapter's own text reports for tensor_sum([1,4,9,16], 4)
describes what sum_reduce_kernel's per-round logic actually computes
when it IS launched each round -- verified independently in Worked
Examples 14.1.1 and 14.1.2 above by tracing that kernel's logic by
hand -- not what the driver shown here literally does as written. The
launch-and-synchronize plumbing between rounds is elided from this
chapter's source material the same way a textbook might write "insert
appropriate boilerplate here"; the algorithm being described is real
and correct, but this specific function body, read as executable Mojo,
is not a complete implementation of it.
```

Reduction is also the identical primitive Part 7's bond-pricing chapter uses to total a portfolio's present value — the same "sum a buffer of floats via pairwise halving" operation that sums squared errors for a loss function also sums discounted cash flows for a bond book, one pairwise round at a time.

<a id="52-minmax-operations"></a>
## 14.2 Min/Max Operations `[FOUNDATIONAL]`

### Intuition

Min and max reductions run the identical tree pattern Section 14.1 established, with the combine step swapped from addition to a comparison — but a comparison-based reduction needs to remember more than the winning *value*, because the backward pass needs to know exactly which input position produced it. `d(max(x))/dx_i` is `1` for the single index that held the maximum and `0` everywhere else — a gradient that flows through exactly one position, not a share distributed across all of them the way sum's gradient does.

### Background

```mojo
fn max_reduce_kernel(
    output_val: UnsafePointer[Scalar[DType.float32]],
    output_idx: UnsafePointer[Int],
    input: UnsafePointer[Scalar[DType.float32]],
    in_idx: UnsafePointer[Int],
    size: Int,
):
    var tid = Int(thread_idx.x)
    if tid * 2 + 1 < size:
        var left = input[tid * 2]
        var right = input[tid * 2 + 1]
        if left >= right:
            output_val[tid] = left
            output_idx[tid] = in_idx[tid * 2]
        else:
            output_val[tid] = right
            output_idx[tid] = in_idx[tid * 2 + 1]
    elif tid * 2 < size:
        output_val[tid] = input[tid * 2]
        output_idx[tid] = in_idx[tid * 2]
```

Every round now carries two parallel buffers instead of one: `output_val`/`input` hold the running maximum, and `output_idx`/`in_idx` hold *which original position* that value came from — seeded, on the very first round, with `in_idx[i] = i` for every `i`, and thereafter simply forwarding whichever index won each comparison.

### Worked Example 14.2.1 — Tracking both the value and its origin

`[3, 7, 2, 9]`, with `in_idx = [0, 1, 2, 3]` seeding the first round. Round `1`: thread `0` compares `left=3` (`in_idx[0]=0`) against `right=7` (`in_idx[1]=1`) — `7 ≥ 3` is false the other way, so `right` wins: `output_val[0]=7`, `output_idx[0]=1`. Thread `1` compares `left=2` (`in_idx[2]=2`) against `right=9` (`in_idx[3]=3`) — `9` wins: `output_val[1]=9`, `output_idx[1]=3`. Round `1` output: values `[7, 9]`, indices `[1, 3]`. Round `2`: thread `0` compares `7` (index `1`) against `9` (index `3`) — `9` wins again: final value `9`, final index `3`. The maximum, `9`, and the position it came from, index `3` in the *original* array, survive together through every round — exactly the pair a later `argmax`-based backward pass needs, since it has to route a gradient back to index `3` specifically and nowhere else.

### Worked Example 14.2.2 — An odd length, and the branch that goes silent

`[5, 2, 9, 1, 7]` with `in_idx = [0,1,2,3,4]`. Round `0`, `next_size = 3`: thread `0` compares `5` vs `2` → `5` wins, index `0`. Thread `1` compares `9` vs `1` → `9` wins, index `2`. Thread `2`: `tid*2=4 < 5` but `tid*2+1=5` is not — the `elif` fires, passing `input[4]=7` and `in_idx[4]=4` straight through unchanged. Round `0` output: values `[5, 9, 7]`, indices `[0, 2, 4]`. Round `1`, `next_size=2`: thread `0` compares `5` vs `9` → `9` wins, index `2`. Thread `1`: `tid*2=2 < 3` but `tid*2+1=3` is not — leftover branch, passing `7`/index `4` through. Round `1` output: values `[9, 7]`, indices `[2, 4]`. Round `2`: thread `0` compares `9` vs `7` → `9` wins, final index `2`. The maximum is `9`, at original position `2` — matching a direct scan of `[5, 2, 9, 1, 7]` by eye.

```
[COMMON TRAP]  No else branch means unwritten, not zero-filled

Compare max_reduce_kernel's structure to sum_reduce_kernel's directly:
sum_reduce_kernel has three branches -- pair, leftover, and an explicit
`else: output[tid] = Float32(0.0)` that zero-fills any thread with no
real work. max_reduce_kernel has only the first two. A thread whose
`tid` satisfies neither `tid*2+1 < size` nor `tid*2 < size` -- which
happens whenever a round is launched with more threads than
`ceil(size/2)` actually calls for -- writes nothing to output_val[tid]
or output_idx[tid] at all. Nothing crashes, because the buffer memory
still exists; the position simply keeps whatever value was already
sitting there, silently, whether that's a stale result left over from
a previous round's reuse of the same buffer or uninitialized memory on
the very first round. A caller who assumes max_reduce_kernel's output
buffer is fully, freshly written every round -- the way
sum_reduce_kernel's explicit else branch guarantees -- can end up
folding a leftover value from round N-2 into round N's comparison,
silently, with no signal that anything went wrong.
```

The `argmax` this produces feeds directly into classification metrics — the neural-network-layers chapter's performance-tracking code compares the argmax of a two-unit output layer against the true label exactly this way — and into the sparse gradient a later backward pass scatters back through only the winning path, leaving every other position's gradient at exactly zero.

## 14.3 Norm Calculations `[FOUNDATIONAL]`

### Intuition

The L2 norm collapses an entire vector into one number measuring its overall size — useful both as a loss function in its own right (regression error is often exactly this) and as a safety valve during training: if a single gradient vector grows unexpectedly large, rescaling it by its own norm can shrink it back to a sane magnitude before it ever reaches an optimizer step.

### Background

```mojo
fn l2_norm(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    var squared = UnsafePointer[Scalar[DType.float32]].alloc(size)
    for i in range(size):
        squared[i] = input[i] * input[i]
    var sum_sq = tensor_sum(squared, size)
    squared.free()
    return sqrt(sum_sq)

fn clip_grad_norm(mut grad: UnsafePointer[Scalar[DType.float32]], size: Int, max_norm: Float32):
    var norm = l2_norm(grad, size)
    if norm > max_norm:
        var scale = max_norm / norm
        for i in range(size):
            grad[i] *= scale
```

`l2_norm` squares every entry into a scratch buffer, reduces that buffer with `tensor_sum` (inheriting the same launch caveat flagged in Section 14.1's `[COMMON TRAP]` — the algorithm is correct, the driver as literally shown isn't complete), and takes one square root of the total. `clip_grad_norm` only ever *shrinks* a gradient, never grows one: when the norm already fits under `max_norm`, the function does nothing at all.

### Worked Example 14.3.1 — The 3-4-5 triangle

`[3, 4]`: squares are `[9, 16]`, sum is `25`, square root is `5` — the familiar 3-4-5 right triangle, and a useful sanity check that `l2_norm` is wired correctly before trusting it on a real, high-dimensional gradient vector.

### Worked Example 14.3.2 — Clipping a gradient back to a safe magnitude

Take that same `[3, 4]` vector as a gradient with `max_norm = 2.0`. `l2_norm` returns `5.0`, which exceeds `2.0`, so clipping fires: `scale = 2.0 / 5.0 = 0.4`. Rescaling every entry: `[3 × 0.4, 4 × 0.4] = [1.2, 1.6]`. Checking the result's own norm confirms the clip worked exactly as intended: `√(1.2² + 1.6²) = √(1.44 + 2.56) = √4.0 = 2.0` — landing precisely on `max_norm`, not merely under it, because clipping always rescales to *exactly* the limit rather than to some smaller, arbitrary value. Without this safety valve, one unusually large gradient — from a numerically unstable loss spike, for instance — could otherwise take a single, enormous, destabilizing optimizer step.

## 14.4 Statistical Functions `[FOUNDATIONAL]`

### Intuition

Variance and standard deviation are sum-and-mean applied twice in sequence — first to find the mean itself, then again to find the mean of how far every value strays from that mean, squared. Placing this section last in the chapter is deliberate: there is no new reduction primitive here, only Sections 14.1's `tensor_sum`/`tensor_mean` reused twice over.

### Background

```mojo
fn tensor_variance(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    var mean = tensor_mean(input, size)
    var sq_dev = UnsafePointer[Scalar[DType.float32]].alloc(size)
    for i in range(size):
        var d = input[i] - mean
        sq_dev[i] = d * d
    var variance = tensor_mean(sq_dev, size)
    sq_dev.free()
    return variance

fn tensor_std(input: UnsafePointer[Scalar[DType.float32]], size: Int) -> Float32:
    return sqrt(tensor_variance(input, size))
```

This is the numerically safer of the two common variance formulas: computing the mean first and then the mean of *actual* squared deviations from it, rather than the single-pass shortcut `mean(x²) − mean(x)²`, which can lose precision to catastrophic cancellation when two large, nearly-equal numbers are subtracted. `tensor_variance` pays for that safety with a second full pass over the data (and a second scratch buffer), which is a fine trade for a quantity computed once at the end of an epoch, not once per training step.

### Worked Example 14.4.1 — Variance and standard deviation on eight values

`[2, 4, 4, 4, 5, 5, 7, 9]` — a textbook example precisely because its answer comes out round. Mean: `(2+4+4+4+5+5+7+9)/8 = 40/8 = 5`. Squared deviations from that mean: `(2-5)²=9`, `(4-5)²=1` (three times), `(5-5)²=0` (twice), `(7-5)²=4`, `(9-5)²=16` — giving `[9, 1, 1, 1, 0, 0, 4, 16]`. Mean of those: `(9+1+1+1+0+0+4+16)/8 = 32/8 = 4` — the variance. Standard deviation is `√4 = 2`, matching what `tensor_variance` and `tensor_std` return on this exact input.

Batch normalization layers in the neural-network-layers chapter call exactly this pair, per-feature, to normalize activations before every hidden layer — subtracting the mean and dividing by the standard deviation is what keeps a network's internal activations from drifting to extreme values as training progresses — and Part 7's risk analytics call the same pair across a bond portfolio's yields to size a Value-at-Risk estimate.

## Chapter Summary

Every reduction in this chapter shares the same tree shape Section 14.1 established: pair up values, combine each pair, halve the array, repeat until one value remains — a discipline that exists because letting many threads write into one shared accumulator is a race condition, and because floating-point addition isn't perfectly associative, so *how* values get combined can change the answer, not just how fast it arrives. `sum_reduce_kernel`'s per-round logic is correct and was independently traced by hand on both an even-length array (`[1,4,9,16] → 30`) and an odd-length one (`[2,6,3,8,5] → 24`, across three full rounds) — but `tensor_sum`'s own driver never actually calls it, leaving its per-round kernel launch as a comment rather than working code, a gap this chapter flagged explicitly rather than pretending the `30` it reports came from running the code exactly as shown. `max_reduce_kernel` extends the same pattern with a parallel index buffer, so the position of the maximum survives every round alongside its value — precisely what a sparse backward pass needs to route a gradient through one winning index and zero everywhere else — though unlike `sum_reduce_kernel`, it has no `else` branch, so an over-launched round leaves some output positions silently unwritten rather than zero-filled. The L2 norm reduces a vector to one measure of size via sum-of-squares-then-square-root, checked against the exact 3-4-5 triangle, and gradient-norm clipping uses that same norm to rescale an overlarge gradient down to precisely `max_norm`, never below it. Variance and standard deviation close the chapter by reusing sum-and-mean twice over — once for the mean, once for the mean of squared deviations from it — the numerically safer two-pass approach, verified end to end on an eight-value dataset down to an exact variance of `4` and standard deviation of `2`.

## Self-Check Questions

1. Trace `sum_reduce_kernel` over `[10, 20, 30, 40, 50]` (five elements) round by round the way Worked Example 14.1.2 traced `[2,6,3,8,5]`, and report the final sum along with a direct check against adding all five values.
2. Trace `max_reduce_kernel` over `[4, 12, 7, 12, 3]` (note the tie at value `12`) with `in_idx = [0,1,2,3,4]`. Which original index does the final result report, and why does the `left >= right` comparison (rather than a strict `>`) determine the answer in the case of a tie?
3. A gradient vector `[6, 8]` is passed to `clip_grad_norm` with `max_norm = 3.0`. What does `l2_norm` return, what is `scale`, and what is the clipped vector? Verify the clipped vector's own norm equals `max_norm` exactly.
4. Compute the variance and standard deviation of `[1, 1, 1, 1, 5, 5, 5, 5]` by hand, following the same two-pass method Worked Example 14.4.1 used (mean first, then mean of squared deviations from it).
5. `tensor_sum`'s `while` loop is read by a colleague who assumes it performs a real GPU launch every iteration, since the comment describes one. What specifically would they need to check in the actual compiled/linked implementation to confirm whether `scratch` is genuinely being populated each round, versus retaining whatever `UnsafePointer.alloc(size)` happened to return?

## Where We Go Next

Chapter 15 (`part3/01-graph-node-architecture.md`) turns to *recording* the compositions this chapter and the two before it made possible — element-wise, matrix, and reduction operations now cover every arithmetic primitive the rest of the book composes — as a graph, so the framework knows how to run any of them backward.

## Worked Solutions

**1.** Round `0` (`current_size=5`, `next_size=3`): thread `0`: `10+20=30`. Thread `1`: `30+40=70`. Thread `2`: leftover, passes `50` through. Round `0` output: `[30, 70, 50]`. Round `1` (`current_size=3`, `next_size=2`): thread `0`: `30+70=100`. Thread `1`: leftover, passes `50` through. Round `1` output: `[100, 50]`. Round `2` (`next_size=1`): thread `0`: `100+50=150`. Final sum: `150`. Direct check: `10+20+30+40+50=150` — matches exactly.

**2.** Round `0`: thread `0` compares `4` (index `0`) vs `12` (index `1`) — `12` wins, index `1`. Thread `1` compares `7` (index `2`) vs `12` (index `3`) — `12` wins, index `3`. Thread `2`: leftover, passes `3`/index `4` through. Round `0`: values `[12, 12, 3]`, indices `[1, 3, 4]`. Round `1`: thread `0` compares `12` (index `1`) vs `12` (index `3`) — a genuine tie. `left >= right` evaluates `12 >= 12`, which is `True`, so `left` wins: the result is index `1`, not index `3`. Thread `1`: leftover, passes `3`/index `4` through. Round `1`: values `[12, 3]`, indices `[1, 4]`. Round `2`: `12` (index `1`) beats `3` — final index `1`. The `>=` (rather than strict `>`) means ties are broken in favor of the *earlier*-encountered operand in each individual comparison — here, the value that started at index `1` — not the later one.

**3.** `l2_norm([6,8]) = √(36+64) = √100 = 10.0`. Since `10.0 > 3.0`, clipping fires: `scale = 3.0/10.0 = 0.3`. Clipped vector: `[6×0.3, 8×0.3] = [1.8, 2.4]`. Verification: `√(1.8² + 2.4²) = √(3.24 + 5.76) = √9.0 = 3.0` — exactly `max_norm`.

**4.** Mean: `(1+1+1+1+5+5+5+5)/8 = 24/8 = 3`. Squared deviations: `(1-3)²=4` (four times), `(5-3)²=4` (four times) — every single deviation is `4`. Mean of those: `32/8 = 4` — the variance. Standard deviation: `√4 = 2`.

**5.** They would need to check whether an actual `ctx.enqueue_function[sum_reduce_kernel](...)` call (or equivalent) executes inside the loop and whether `ctx.synchronize()` (or the framework's equivalent barrier) runs before `current = scratch` takes effect — neither of which appears in the version of `tensor_sum` shown in this chapter, where that step is a comment. Confirming genuine population would mean checking, at minimum: that a kernel launch call exists at all: that it targets `scratch` as its output and the current round's data as input; that a synchronization point exists before the next round reads from `scratch`; and, ultimately, printing or inspecting `scratch`'s contents after one round to confirm they match `sum_reduce_kernel`'s expected pairwise sums rather than the buffer's original, uninitialized contents.
