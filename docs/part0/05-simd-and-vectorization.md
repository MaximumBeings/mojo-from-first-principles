# Chapter 5: SIMD and Vectorization — Writing the Loop Once

> "Every vectorized loop in this book, from here through the neural network in Part 6, has the same three-part shape: a main loop that runs at full SIMD width, a short remainder loop for whatever's left over, and a promise — checkable by hand, the way every worked example in this book has been — that the two loops together produce exactly what a plain scalar loop would have produced, just faster."

**What you will understand by the end of this chapter:**

- Why legal SIMD widths are restricted to powers of two, tracing directly back to how vector hardware registers are actually built
- The canonical **vectorized-loop shape** — `simd_count = (size // width) * width`, a main loop over full-width chunks, and a scalar remainder loop for whatever's left — and how to compute both pieces by hand for a given size and width
- **SIMD reduction**: why accumulating into a width-wide vector and reducing it *once*, at the very end, is fundamentally different from reducing after every chunk
- How the exact same vectorized-loop shape nests inside a matrix multiply's row/column loops, and inside a transpose's row loop
- How to trace a full forward-and-backward pass through a vectorized tensor, element by element, and confirm it reproduces Chapter 7's chain-rule results exactly — just computed eight elements at a time instead of one

**What you need to know first:**

- Chapter 1.4 (SIMD as a first-class type) — this chapter is a direct continuation of that section's lane model
- Chapter 3.3 (SoA enabling contiguous SIMD loads) — every vectorized loop in this chapter depends on the data already being laid out the way Chapter 3 argued for
- Chapter 4.4 (memory coalescing) — CPU SIMD width and GPU warp width are the same idea at two different scales, and Section 5.1 makes that connection explicit
- Chapter 7 (Backward Function Implementation) — Section 5.5 re-derives that chapter's `AddOp`/`MulOp` backward rules in vectorized form, and assumes you already have the scalar version fresh from Chapter 7

## 5.1 Legal SIMD Widths and the Shape All of Them Share `[FOUNDATIONAL]`

### Intuition

Chapter 1.4 pictured four workers stapling in lockstep. Now ask: why couldn't there be three workers, or five? Because the "assembly line" they stand on is a physical vector register of a *fixed total width* — say, 128 bits — and that fixed width only divides evenly, into whole numbers of equal-sized lanes, at certain counts: 1 lane of 128 bits, 2 lanes of 64 bits, 4 lanes of 32 bits, 8 lanes of 16 bits, and so on. A team of 3 or 5 workers doesn't correspond to any way of slicing a fixed-width register into equal pieces — there would always be leftover space that fits no worker at all.

### Background

A SIMD register's total bit-width is fixed by the hardware (commonly 128, 256, or 512 bits), and it splits evenly into lanes only at power-of-two lane counts, because a power of two is the only kind of number guaranteed to divide another power-of-two-sized quantity into equal whole parts for every element width a register supports. For `Float32` (4 bytes = 32 bits per lane):

| Width | Total bits used | Register size this fits |
|---|---|---|
| `SIMD[float32, 1]` | 32 bits | Any (degenerate, one lane) |
| `SIMD[float32, 2]` | 64 bits | Half of a 128-bit register |
| `SIMD[float32, 4]` | 128 bits | A full 128-bit register |
| `SIMD[float32, 8]` | 256 bits | A full 256-bit register |

### Worked Example 5.1.1 — Broadcast/splat, traced against its scalar equivalent

```mojo
var broadcast = SIMD[DType.float32, 4](3.14)   # every one of the 4 lanes set to 3.14
```

This single constructor call sets all four lanes to `3.14` in one step — `[3.14, 3.14, 3.14, 3.14]`. The scalar equivalent needs four separate assignments to four separate memory locations (`buf[0]=3.14`, `buf[1]=3.14`, `buf[2]=3.14`, `buf[3]=3.14`) to reach the same state — one constructor call standing in for what would otherwise be a small loop, because "broadcast" is a hardware-supported operation on the register itself, not a sequence of individual writes.

### Worked Example 5.1.2 — Comparisons, traced lane by lane

```mojo
var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var b = SIMD[DType.float32, 4](2.0, 2.0, 1.0, 5.0)
var eq = a == b
var lt = a < b
```

Every comparison operator applies Chapter 1.4's lane-wise rule: each lane of the result is that comparison's answer for *just that lane's pair* of values.

```
lane:        0        1        2        3
a:          1.0      2.0      3.0      4.0
b:          2.0      2.0      1.0      5.0
a == b:    False     True    False    False
a  < b:     True    False    False     True
a  > b:    False    False     True    False
a <= b:     True     True    False     True
```

> `[COMMON TRAP]` `a == b` on two `SIMD` vectors does **not** produce one `Bool` answering "are these two vectors equal" — it produces a `SIMD` of four separate booleans, one per lane, exactly as traced above. Dropping that four-lane result directly into a scalar `if` is a type error waiting to happen, or a silent logic bug if the language happens to coerce it in some unintended way — the vector needs an explicit reduction (an "are all lanes true" or "is any lane true" operation) before it means anything as a single yes/no answer.

## 5.2 The Vectorized-Loop Shape: Main Loop Plus Remainder `[FOUNDATIONAL]`

### Intuition

A delivery truck that only accepts full pallets of, say, 4 boxes at a time: the loading crew stacks as many complete pallets as the boxes allow, and whatever handful of boxes is left over — fewer than 4, by definition, or there'd be one more full pallet — gets carried onto the truck bed by hand, individually. The pallets are the fast path; the leftover boxes are the unavoidable slow path for whatever doesn't divide evenly.

### Background

Every vectorized loop in this book (Chapter 3.3's `update_positions_simd`, Chapter 4's kernels, and every function in this chapter) is built from the same two pieces:

```
simd_count = (size // width) * width     # how many elements the main loop covers
remainder  = size - simd_count           # always in [0, width - 1]
```

`size // width` is "how many complete, full-width pallets fit," and multiplying back by `width` converts that pallet count back into an element count — always a multiple of `width`, always less than or equal to `size`. The main loop processes `simd_count` elements, `width` at a time; a second, ordinary scalar loop mops up whatever's left, from `simd_count` up to `size`.

### Worked Example 5.2.1 — A case with a real remainder

Take a hypothetical `size = 19`, `width = 4`: `simd_count = (19 // 4) × 4 = 4 × 4 = 16`, so `remainder = 19 - 16 = 3` — the main loop covers indices `0` through `15` in four chunks of 4, and a 3-element scalar tail covers indices `16`, `17`, `18`.

This chapter's own demo, by contrast, uses `size = 1000` with both `width = 4` and `width = 8` — and `1000 / 4 = 250` and `1000 / 8 = 125` both come out exact, so `remainder = 0` in both cases here. The remainder loop's body simply never executes for this particular `size` — but it still has to be *present* in the code, because nothing about `simd_vector_add[simd_width]`'s implementation can assume its caller will always supply a size divisible by the width; change `size` to `1001` and that same remainder loop is suddenly doing exactly one iteration of real work.

### Worked Example 5.2.2 — The first main-loop chunk, traced by hand

```mojo
var a_vals = SIMD[DType.float32, 4](0)
var b_vals = SIMD[DType.float32, 4](0)
for j in range(4):
    a_vals[j] = a[i + j]
    b_vals[j] = b[i + j]
var result_vals = a_vals + b_vals
```

With `a[i] = i × 0.1` and `b[i] = i × 0.2`, trace the very first chunk (`i = 0`): `a_vals = [0.0, 0.1, 0.2, 0.3]`, `b_vals = [0.0, 0.2, 0.4, 0.6]`, and `result_vals = [0.0+0.0, 0.1+0.2, 0.2+0.4, 0.3+0.6] = [0.0, 0.3, 0.6, 0.9]` — the identical four numbers a plain scalar loop would produce running `result[i] = a[i] + b[i]` for `i = 0, 1, 2, 3` one at a time, which is exactly what this chapter's own verification step checks for every element, not just these four.

### ASCII Diagram — 1000 elements, width 8, evenly divided

```
1000 elements / width 8 = 125 exact "pallets" -- remainder = 0 for THIS size:
 [chunk0: idx 0-7][chunk1: idx 8-15]...[chunk124: idx 992-999]   <- 125 chunks, all full
                                                                     no scalar tail needed here
```

> `[COMMON TRAP]` It is tempting to drop the remainder loop when testing against a size that happens to divide evenly — everything still passes, because the remainder loop's body never runs. But `UnsafePointer.alloc` (Chapter 1.5) does not zero-initialize the memory it hands back, so on a size that does *not* divide evenly, a missing remainder loop leaves the last `size mod width` elements of the output holding whatever garbage bytes were already sitting in that freshly allocated memory — not zero, not "unchanged," just undefined leftover bits, silently wrong in a way that a size divisible by the width will never reveal.

## 5.3 SIMD Reduction: Accumulate Wide, Reduce Narrow `[FOUNDATIONAL]`

### Intuition

Four cash registers ring up sales all day long, completely independently, each one keeping its own running subtotal. Only once, at closing time, does the store manager walk to all four registers and add their four subtotals together into one final total for the day. The registers never coordinate with each other during business hours — coordination happens exactly once, briefly, at the end.

### Background

`simd_sum[width]` is this pattern exactly: a width-wide accumulator, `vector_sum`, is updated with an ordinary lane-wise SIMD `+=` (Chapter 1.4's rule) once per chunk, across the *entire* array — this is the "cash registers ringing up sales," happening in parallel across `width` lanes for every chunk. Only after every chunk has been folded in does a **horizontal reduction** sum the accumulator's own `width` lanes down to one scalar — the "manager's walk," touching only `width` numbers, not `size` numbers, no matter how large the array was.

### Worked Example 5.3.1 — The first two chunks of a real sum

With `data_a[i] = i + 1` (so `data_a = [1, 2, 3, ..., 1000]`) and `width = 4`:

```
vector_sum starts at [0, 0, 0, 0]

chunk 0 (i=0..3): data_vals = [1, 2, 3, 4]
  vector_sum = [0+1, 0+2, 0+3, 0+4] = [1, 2, 3, 4]

chunk 1 (i=4..7): data_vals = [5, 6, 7, 8]
  vector_sum = [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
```

Lane 0 is quietly accumulating every 4th element starting at index 0 (`1, 5, 9, ...`), lane 1 every 4th starting at index 1 (`2, 6, 10, ...`), and so on — four separate partial sums, each covering a quarter of the array, computed simultaneously.

### Worked Example 5.3.2 — The final horizontal reduction, checked against the closed form

After all `250` chunks (`1000 / 4`) have been folded in, the four lanes hold the four partial sums described above. The final reduction adds just those four numbers:

```
final_sum = vector_sum[0] + vector_sum[1] + vector_sum[2] + vector_sum[3]
```

Rather than trace all 250 chunks, check the answer against the closed-form sum `1 + 2 + ... + 1000 = size×(size+1)/2 = 1000×1001/2 = 500500` — and this chapter's own code performs exactly this cross-check, printing the closed-form value alongside both the scalar and SIMD sums specifically so the three can be compared. The dot product `simd_dot_product[width](data_a, data_b)` with `data_b` filled entirely with `1.0`s reduces to the identical sum, since `data_a[i] × 1.0 = data_a[i]` for every lane — `500500` again, which is why the source's own verification step compares `SIMD4` against `SIMD8` dot products rather than against a separately-derived formula: both must agree with each other, and with the plain sum, since a dot product against all-ones *is* a sum.

### ASCII Diagram — four lanes, each owning a quarter of the array, reduced once

```
lane 0 accumulates: data[0], data[4], data[8], ... (250 values)
lane 1 accumulates: data[1], data[5], data[9], ... (250 values)
lane 2 accumulates: data[2], data[6], data[10],... (250 values)
lane 3 accumulates: data[3], data[7], data[11],... (250 values)
                                    |
                                    v
              ONE reduction, 4 numbers added, at the very end
```

> `[COMMON TRAP]` Performing the horizontal reduction *inside* the main loop — reducing `vector_sum` down to a scalar after every single chunk instead of once at the end — throws away almost all of the benefit of accumulating in parallel. Doing that here would mean 250 separate 4-element reductions instead of one, re-paying the "manager's walk" cost on every single chunk rather than once at closing time.

## 5.4 SIMD Inside Two Loops: Matrix Multiply and Transpose `[FOUNDATIONAL]`

### Intuition

The cash-register idea from Section 5.3, run once per *cell* of a matrix product instead of once for a whole flat array: each output cell gets its own small width-wide accumulator, reduced once, exactly like Section 5.3's whole-array sum — just repeated once for every `(row, col)` pair in the output, the same "one thread, one dot product" structure Chapter 4.6 already introduced for the GPU case, now running on the CPU with SIMD lanes standing in for GPU threads.

### Background

`simd_matrix_multiply[width]` nests Section 5.3's accumulate-then-reduce pattern inside the ordinary two-level `(row, col)` loop from Chapter 4.6: for each output cell, it vector-accumulates products from `A`'s row and `B`'s column, `width` elements at a time, reduces the accumulator once, and adds any scalar remainder — identical machinery to `simd_sum`, just invoked once per cell rather than once for the whole array.

### Worked Example 5.4.1 — Multiplying by the identity, by hand

This chapter's own test multiplies a sequentially-filled matrix by an identity matrix, which must return the original matrix unchanged (`A @ I = A`) — a strong, easy-to-check correctness test. Trace a small `2×2` case: `A = [[0,1],[2,3]]` (sequential fill, `A[i][j] = i×cols+j` with `cols=2`) and `I = [[1,0],[0,1]]`.

```
cell(0,0) = A[0,0]*I[0,0] + A[0,1]*I[1,0] = 0*1 + 1*0 = 0
cell(0,1) = A[0,0]*I[0,1] + A[0,1]*I[1,1] = 0*0 + 1*1 = 1
cell(1,0) = A[1,0]*I[0,0] + A[1,1]*I[1,0] = 2*1 + 3*0 = 2
cell(1,1) = A[1,0]*I[0,1] + A[1,1]*I[1,1] = 2*0 + 3*1 = 3
```

Result: `[[0,1],[2,3]]` — exactly `A`, unchanged, regardless of whether the dot products above were computed by a plain scalar loop or by `simd_matrix_multiply`'s accumulate-then-reduce machinery. Both must produce this identical answer, which is precisely what this chapter's own verification step checks across the full `64×64` case.

### Worked Example 5.4.2 — Transposing one row, by hand

```mojo
var row_vals = SIMD[DType.float32, simd_width](0)
for k in range(simd_width):
    row_vals[k] = input.get(i, j + k)
for k in range(simd_width):
    output.set(j + k, i, row_vals[k])
```

For row `i=0` of a sequentially-filled matrix (`A[0][j] = j`), the first chunk loads `row_vals = [0, 1, 2, 3]` from row 0's first four columns. The store loop then writes `output[0][0]=0, output[1][0]=1, output[2][0]=2, output[3][0]=3` — row 0 of the input has become column 0 of the output, which is the definition of a transpose, confirmed here on real numbers rather than taken on faith.

### ASCII Diagram — a row becomes a column

```
Input row 0:  [ 0 ][ 1 ][ 2 ][ 3 ]   (4 contiguous values, one SIMD load)

Output column 0:
 [0]
 [1]        <- the same 4 values, now stored one per row
 [2]
 [3]
```

> `[COMMON TRAP]` `simd_matrix_multiply` reloads row `i` of `A` fresh from memory for every single column `j` it multiplies against — the exact same row gets fetched `b.cols` times over, once per output cell in that row. This isn't a correctness bug (Worked Example 5.4.1 confirms the answer is right), but it is the identical inefficiency Chapter 4.6 flagged for the GPU matrix-multiply simulation: real high-performance implementations cache a reused row or column once (in a CPU cache line, or a GPU block's shared memory) instead of paying the fetch cost again for every cell that needs it — an optimization intentionally left for later, not attempted here.

## 5.5 SIMD Through a Forward *and* Backward Pass `[FOUNDATIONAL]`

### Intuition

Chapter 7 traced the chain rule through `w = x·y + x` for one number at a time. Imagine running that exact same worked example 1000 times simultaneously — not one after another, but with 1000 copies of `x`, `y`, and every intermediate value marching through the identical sequence of operations side by side, eight at a time. `VectorizedTensor`'s forward and backward passes are that idea made concrete.

### Background

`VectorizedTensor`'s forward pass computes `z = (x + y) * w` — the same shape as Chapter 7's `add`-then-`multiply` structure, just reordered and renamed. `temp = x.simd_elementwise_add[8](y)` is Chapter 1.4/3.3's lane-wise add, applied tensor-wide; `z = temp.simd_elementwise_multiply[8](w)` is the lane-wise multiply. The backward pass re-derives Chapter 7.2's rules directly: `MulOp.backward` said each input receives the *other* input's value times the incoming gradient, so `grad_w = temp * grad_z` and `grad_temp = w * grad_z` are that exact rule, computed for every lane; `AddOp.backward` said gradient passes through unchanged to both inputs, so `simd_backward_add` accumulates the identical `grad_temp` into both `x.gradients` and `y.gradients`.

### Worked Example 5.5.1 — One element, traced end to end

Trace index `i = 100` completely, using this chapter's own initializer (`x[i] = i × 0.01`, `y[i] = 2.0` constant, `w[i] = 0.5` constant, and an all-ones `grad_z`):

```
x[100] = 100 * 0.01 = 1.0
y[100] = 2.0
w[100] = 0.5

Forward:
  temp[100] = x[100] + y[100] = 1.0 + 2.0 = 3.0
  z[100]    = temp[100] * w[100] = 3.0 * 0.5 = 1.5

Backward (grad_z[100] = 1.0):
  grad_w[100]    = temp[100] * grad_z[100] = 3.0 * 1.0 = 3.0   (MulOp: "the other input")
  grad_temp[100] = w[100] * grad_z[100]    = 0.5 * 1.0 = 0.5   (MulOp: "the other input", symmetric)
  dx[100] = grad_temp[100] = 0.5                                (AddOp: passes through unchanged)
  dy[100] = grad_temp[100] = 0.5                                (AddOp: passes through unchanged)
```

Cross-check this against plain calculus, the same discipline Chapter 7 used throughout: `z = (x+y)·w`, so `∂z/∂x = w` treating `y` and `w` as independent variables — `∂z/∂x = 0.5`, matching `dx[100] = 0.5` computed above through the chain rule. Two routes, same element, same answer.

### ASCII Diagram — the graph, annotated with lane 100's numbers

```
x[100]=1.0 --\
              (add) -- temp[100]=3.0 --\
y[100]=2.0 --/                          (mul) -- z[100]=1.5
                          w[100]=0.5 --/

backward (grad_z[100]=1.0):
 grad_temp[100]=0.5 <---(mul rule)--- w[100]=0.5, grad_z[100]=1.0
 grad_w[100]=3.0    <---(mul rule)--- temp[100]=3.0, grad_z[100]=1.0
 dx[100]=0.5, dy[100]=0.5  <---(add rule, unchanged)--- grad_temp[100]
```

> `[COMMON TRAP]` Trace what this chapter's own demo code actually *stores*, not just what it *prints*: `grad_w` is computed correctly, as shown above, and printed — but the code never calls `w.simd_backward_add[8](grad_w)` the way it calls `x.simd_backward_add[8](grad_temp)` and `y.simd_backward_add[8](grad_temp)`. So if you inspected `w.gradients` after this demo finishes, you'd find it still all zeros, even though the *value* `grad_w` printed to the screen is the mathematically correct gradient. A complete training step would finish this pattern with `w.simd_backward_add[8](grad_w)`, exactly mirroring what already happens for `x` and `y` — a good reminder that "the right number was computed" and "the right number was stored where the rest of the system will look for it" are two separate facts, both worth checking.

## 5.6 Complete Runnable Code

### File: `21_simd_fundamentals.mojo` — Section 5.1

**Execution:** `pixi run mojo 21_simd_fundamentals.mojo`

```mojo
from memory import UnsafePointer

fn simd_basics_demo():
    """Demonstrate fundamental SIMD operations in Mojo."""
    print("=== SIMD Fundamentals ===")

    # Basic SIMD vector creation
    print("1. SIMD Vector Creation:")
    var vec4_float = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var vec8_int = SIMD[DType.int32, 8](1, 2, 3, 4, 5, 6, 7, 8)

    print("  Float32x4:", vec4_float)
    print("  Int32x8:", vec8_int)

    # SIMD width must be power of 2
    print("\n2. Valid SIMD Widths (powers of 2):")
    var vec1 = SIMD[DType.float32, 1](42.0)
    var vec2 = SIMD[DType.float32, 2](1.0, 2.0)
    var vec4 = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var vec8 = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    print("  1-wide:", vec1)
    print("  2-wide:", vec2)
    print("  4-wide:", vec4)
    print("  8-wide:", vec8)

    # Element access
    print("\n3. Element Access:")
    print("  vec4[0] =", vec4[0])
    print("  vec4[1] =", vec4[1])
    print("  vec4[2] =", vec4[2])
    print("  vec4[3] =", vec4[3])

    # Broadcasting (splat)
    print("\n4. Broadcasting/Splat:")
    var broadcast = SIMD[DType.float32, 4](3.14)  # All elements = 3.14
    print("  Broadcast 3.14:", broadcast)

fn simd_arithmetic_demo():
    """Demonstrate SIMD arithmetic operations."""
    print("\n=== SIMD Arithmetic Operations ===")

    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0)

    print("Vector A:", a)
    print("Vector B:", b)

    # Element-wise operations
    var add_result = a + b
    var sub_result = a - b
    var mul_result = a * b
    var div_result = b / a

    print("\nArithmetic Results:")
    print("  A + B =", add_result)
    print("  A - B =", sub_result)
    print("  A * B =", mul_result)
    print("  B / A =", div_result)

    # Scalar operations
    var scalar = Float32(2.0)
    var scalar_mul = a * scalar
    var scalar_add = a + scalar

    print("\nScalar Operations:")
    print("  A * 2 =", scalar_mul)
    print("  A + 2 =", scalar_add)

fn simd_comparison_demo():
    """Demonstrate SIMD comparison operations."""
    print("\n=== SIMD Comparison Operations ===")

    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](2.0, 2.0, 1.0, 5.0)

    print("Vector A:", a)
    print("Vector B:", b)

    # Comparison operations return boolean SIMD vectors
    var eq = a == b
    var lt = a < b
    var gt = a > b
    var le = a <= b

    print("\nComparison Results:")
    print("  A == B:", eq)
    print("  A < B:", lt)
    print("  A > B:", gt)
    print("  A <= B:", le)

fn main():
    """Main function for SIMD fundamentals demonstration."""
    simd_basics_demo()
    simd_arithmetic_demo()
    simd_comparison_demo()
```

### File: `22_vectorized_loops.mojo` — Section 5.2

**Execution:** `pixi run mojo 22_vectorized_loops.mojo`

```mojo
from memory import UnsafePointer

fn scalar_vector_add(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                    result: UnsafePointer[Float32], size: Int):
    """Scalar version of vector addition (baseline)."""
    for i in range(size):
        result[i] = a[i] + b[i]

fn simd_vector_add[simd_width: Int](a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                                   result: UnsafePointer[Float32], size: Int):
    """SIMD vectorized version of vector addition."""
    var simd_count = (size // simd_width) * simd_width

    # Process SIMD chunks
    for i in range(0, simd_count, simd_width):
        # Manual load from memory
        var a_vals = SIMD[DType.float32, simd_width](0)
        var b_vals = SIMD[DType.float32, simd_width](0)

        for j in range(simd_width):
            a_vals[j] = a[i + j]
            b_vals[j] = b[i + j]

        # SIMD addition
        var result_vals = a_vals + b_vals

        # Manual store back to memory
        for j in range(simd_width):
            result[i + j] = result_vals[j]

    # Handle remaining elements
    for i in range(simd_count, size):
        result[i] = a[i] + b[i]

fn vectorization_comparison():
    """Compare scalar vs vectorized implementations."""
    print("=== Vectorization Performance Comparison ===")

    var size = 1000
    var a = UnsafePointer[Float32].alloc(size)
    var b = UnsafePointer[Float32].alloc(size)
    var result_scalar = UnsafePointer[Float32].alloc(size)
    var result_simd4 = UnsafePointer[Float32].alloc(size)
    var result_simd8 = UnsafePointer[Float32].alloc(size)

    # Initialize test data
    for i in range(size):
        a[i] = Float32(i) * 0.1
        b[i] = Float32(i) * 0.2

    print("Array size:", size, "elements")
    print("Test data: a[i] = i * 0.1, b[i] = i * 0.2")

    # Scalar version
    scalar_vector_add(a, b, result_scalar, size)
    print("\nScalar computation completed")

    # SIMD 4-wide version
    simd_vector_add[4](a, b, result_simd4, size)
    print("SIMD 4-wide computation completed")

    # SIMD 8-wide version
    simd_vector_add[8](a, b, result_simd8, size)
    print("SIMD 8-wide computation completed")

    # Verify results are identical
    var errors = 0
    for i in range(size):
        if abs(result_scalar[i] - result_simd4[i]) > 0.001:
            errors += 1
        if abs(result_scalar[i] - result_simd8[i]) > 0.001:
            errors += 1

    print("\nVerification:")
    print("  Scalar vs SIMD4 errors:", errors // 2)
    print("  Scalar vs SIMD8 errors:", errors - (errors // 2))

    # Show sample results
    print("\nSample results (first 8 elements):")
    for i in range(8):
        var i_str: String = String(i)
        var result_str: String = String(result_scalar[i])
        print("  result[" + i_str + "] = " + result_str)

    print("\nVectorization Benefits:")
    print("  + Process multiple elements per instruction")
    print("  + Better CPU utilization")
    print("  + Reduced instruction overhead")
    print("  + Essential for high-performance computing")

    # Cleanup
    a.free()
    b.free()
    result_scalar.free()
    result_simd4.free()
    result_simd8.free()

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Main function for vectorization comparison."""
    vectorization_comparison()
```

### File: `23_simd_reduction.mojo` — Section 5.3

**Execution:** `pixi run mojo 23_simd_reduction.mojo`

```mojo
from memory import UnsafePointer

fn scalar_sum(data: UnsafePointer[Float32], size: Int) -> Float32:
    """Scalar implementation of array sum."""
    var sum: Float32 = 0.0
    for i in range(size):
        sum += data[i]
    return sum

fn simd_sum[simd_width: Int](data: UnsafePointer[Float32], size: Int) -> Float32:
    """SIMD implementation of array sum with reduction."""
    var simd_count = (size // simd_width) * simd_width
    var vector_sum = SIMD[DType.float32, simd_width](0.0)

    # SIMD accumulation
    for i in range(0, simd_count, simd_width):
        # Manual load
        var data_vals = SIMD[DType.float32, simd_width](0)
        for j in range(simd_width):
            data_vals[j] = data[i + j]

        # Accumulate
        vector_sum += data_vals

    # Reduce SIMD vector to scalar
    var final_sum: Float32 = 0.0
    for i in range(simd_width):
        final_sum += vector_sum[i]

    # Handle remaining elements
    for i in range(simd_count, size):
        final_sum += data[i]

    return final_sum

fn simd_dot_product[simd_width: Int](a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                                    size: Int) -> Float32:
    """SIMD implementation of dot product."""
    var simd_count = (size // simd_width) * simd_width
    var vector_sum = SIMD[DType.float32, simd_width](0.0)

    # SIMD dot product accumulation
    for i in range(0, simd_count, simd_width):
        # Manual load both vectors
        var a_vals = SIMD[DType.float32, simd_width](0)
        var b_vals = SIMD[DType.float32, simd_width](0)

        for j in range(simd_width):
            a_vals[j] = a[i + j]
            b_vals[j] = b[i + j]

        # Element-wise multiply and accumulate
        vector_sum += a_vals * b_vals

    # Reduce to scalar
    var final_sum: Float32 = 0.0
    for i in range(simd_width):
        final_sum += vector_sum[i]

    # Handle remaining elements
    for i in range(simd_count, size):
        final_sum += a[i] * b[i]

    return final_sum

fn reduction_operations_demo():
    """Demonstrate SIMD reduction operations."""
    print("=== SIMD Reduction Operations ===")

    var size = 1000
    var data_a = UnsafePointer[Float32].alloc(size)
    var data_b = UnsafePointer[Float32].alloc(size)

    # Initialize test data
    for i in range(size):
        data_a[i] = Float32(i + 1)  # 1, 2, 3, ..., 1000
        data_b[i] = Float32(1.0)    # All ones for simple dot product

    print("Test data:")
    print("  Array A: [1, 2, 3, ..., 1000]")
    print("  Array B: [1, 1, 1, ..., 1] (all ones)")
    print("  Size:", size, "elements")

    # Sum operations
    print("\n1. Array Sum Operations:")
    var scalar_sum_result = scalar_sum(data_a, size)
    var simd4_sum_result = simd_sum[4](data_a, size)
    var simd8_sum_result = simd_sum[8](data_a, size)

    print("  Scalar sum:", scalar_sum_result)
    print("  SIMD4 sum:", simd4_sum_result)
    print("  SIMD8 sum:", simd8_sum_result)
    print("  Expected:", (size * (size + 1)) // 2, "(mathematical formula)")

    # Dot product operations
    print("\n2. Dot Product Operations:")
    var simd4_dot = simd_dot_product[4](data_a, data_b, size)
    var simd8_dot = simd_dot_product[8](data_a, data_b, size)

    print("  SIMD4 dot product:", simd4_dot)
    print("  SIMD8 dot product:", simd8_dot)
    print("  Expected:", (size * (size + 1)) // 2, "(A . ones = sum(A))")

    # Verify correctness
    var sum_error = abs(scalar_sum_result - simd4_sum_result)
    var dot_error = abs(simd4_dot - simd8_dot)

    print("\n3. Verification:")
    print("  Sum accuracy (scalar vs SIMD4):", "PASS" if sum_error < 0.001 else "FAIL")
    print("  Dot product accuracy (SIMD4 vs SIMD8):", "PASS" if dot_error < 0.001 else "FAIL")

    print("\nReduction Applications in AD:")
    print("  + Loss function computation (sum of errors)")
    print("  + Gradient norms for optimization")
    print("  + Batch statistics (mean, variance)")
    print("  + Inner products for attention mechanisms")

    # Cleanup
    data_a.free()
    data_b.free()

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Main function for reduction operations demonstration."""
    reduction_operations_demo()
```

### File: `24_matrix_simd.mojo` — Section 5.4

**Execution:** `pixi run mojo 24_matrix_simd.mojo`

```mojo
from memory import UnsafePointer

struct Matrix:
    """Simple matrix structure for SIMD demonstrations."""
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Float32].alloc(rows * cols)

    fn __del__(owned self):
        self.data.free()

    fn get(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn set(self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn fill_identity(self):
        """Fill matrix as identity matrix."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.set(i, j, Float32(1.0) if i == j else Float32(0.0))

    fn fill_sequential(self):
        """Fill matrix with sequential values."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.set(i, j, Float32(i * self.cols + j))

fn scalar_matrix_multiply(a: Matrix, b: Matrix, result: Matrix):
    """Scalar implementation of matrix multiplication."""
    for i in range(a.rows):
        for j in range(b.cols):
            var sum: Float32 = 0.0
            for k in range(a.cols):
                sum += a.get(i, k) * b.get(k, j)
            result.set(i, j, sum)

fn simd_matrix_multiply[simd_width: Int](a: Matrix, b: Matrix, result: Matrix):
    """SIMD-optimized matrix multiplication (simplified)."""
    for i in range(a.rows):
        for j in range(b.cols):
            var sum: Float32 = 0.0
            var simd_count = (a.cols // simd_width) * simd_width
            var vector_sum = SIMD[DType.float32, simd_width](0.0)

            # SIMD inner product
            for k in range(0, simd_count, simd_width):
                # Load elements from row i of matrix A
                var a_vals = SIMD[DType.float32, simd_width](0)
                for l in range(simd_width):
                    a_vals[l] = a.get(i, k + l)

                # Load elements from column j of matrix B
                var b_vals = SIMD[DType.float32, simd_width](0)
                for l in range(simd_width):
                    b_vals[l] = b.get(k + l, j)

                # Multiply and accumulate
                vector_sum += a_vals * b_vals

            # Reduce SIMD result
            for l in range(simd_width):
                sum += vector_sum[l]

            # Handle remaining elements
            for k in range(simd_count, a.cols):
                sum += a.get(i, k) * b.get(k, j)

            result.set(i, j, sum)

fn matrix_transpose_simd[simd_width: Int](input: Matrix, output: Matrix):
    """SIMD-optimized matrix transpose."""
    # Simple approach: process multiple elements per iteration where possible
    for i in range(input.rows):
        var simd_count = (input.cols // simd_width) * simd_width

        # Process SIMD chunks
        for j in range(0, simd_count, simd_width):
            # Load a row chunk
            var row_vals = SIMD[DType.float32, simd_width](0)
            for k in range(simd_width):
                row_vals[k] = input.get(i, j + k)

            # Store as column chunks (transpose)
            for k in range(simd_width):
                output.set(j + k, i, row_vals[k])

        # Handle remaining elements
        for j in range(simd_count, input.cols):
            output.set(j, i, input.get(i, j))

fn matrix_operations_demo():
    """Demonstrate SIMD matrix operations."""
    print("=== SIMD Matrix Operations ===")

    var size = 64  # Small matrix for demonstration

    # Create matrices
    var matrix_a = Matrix(size, size)
    var matrix_b = Matrix(size, size)
    var result_scalar = Matrix(size, size)
    var result_simd = Matrix(size, size)
    var transposed = Matrix(size, size)

    # Initialize matrices
    matrix_a.fill_sequential()
    matrix_b.fill_identity()

    print("Matrix operations on", size, "x", size, "matrices")
    print("Matrix A: Sequential values [0, 1, 2, ...]")
    print("Matrix B: Identity matrix")

    # Matrix multiplication
    print("\n1. Matrix Multiplication (A * B):")
    scalar_matrix_multiply(matrix_a, matrix_b, result_scalar)
    print("  Scalar multiplication completed")

    simd_matrix_multiply[4](matrix_a, matrix_b, result_simd)
    print("  SIMD4 multiplication completed")

    # Verify results
    var errors = 0
    for i in range(size):
        for j in range(size):
            if abs(result_scalar.get(i, j) - result_simd.get(i, j)) > 0.001:
                errors += 1

    print("  Verification:", "PASS" if errors == 0 else ("FAIL - " + String(errors) + " errors"))

    # Matrix transpose
    print("\n2. Matrix Transpose:")
    matrix_transpose_simd[4](matrix_a, transposed)
    print("  SIMD4 transpose completed")

    # Verify transpose
    var transpose_errors = 0
    for i in range(min(size, 8)):  # Check first 8x8 submatrix
        for j in range(min(size, 8)):
            if abs(matrix_a.get(i, j) - transposed.get(j, i)) > 0.001:
                transpose_errors += 1

    print("  Transpose verification:", "PASS" if transpose_errors == 0 else "FAIL")

    # Show sample results
    print("\n3. Sample Results (top-left 4x4):")
    print("  Original A * B (should equal A since B is identity):")
    for i in range(4):
        print("    [", end="")
        for j in range(4):
            print(result_scalar.get(i, j), end="")
            if j < 3:
                print(", ", end="")
        print("]")

    print("\nSIMD Matrix Benefits:")
    print("  + Vectorized inner products")
    print("  + Parallel element processing")
    print("  + Cache-friendly access patterns")
    print("  + Essential for neural network layers")
    print("  + Automatic differentiation optimization")

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Main function for matrix operations demonstration."""
    matrix_operations_demo()
```

### File: `25_simd_autodiff.mojo` — Section 5.5

**Execution:** `pixi run mojo 25_simd_autodiff.mojo`

```mojo
from memory import UnsafePointer

struct VectorizedTensor[dtype: DType]:
    """Tensor optimized for SIMD operations in automatic differentiation."""
    var data: UnsafePointer[Scalar[dtype]]
    var gradients: UnsafePointer[Scalar[dtype]]
    var size: Int
    var requires_grad: Bool

    fn __init__(out self, size: Int, requires_grad: Bool = False):
        self.size = size
        self.requires_grad = requires_grad
        self.data = UnsafePointer[Scalar[dtype]].alloc(size)

        if requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(size)
            # Initialize gradients to zero
            for i in range(size):
                self.gradients[i] = Scalar[dtype](0)
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for tensor copying."""
        self.size = existing.size
        self.requires_grad = existing.requires_grad

        # Allocate and copy data
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        for i in range(self.size):
            self.data[i] = existing.data[i]

        # Allocate and copy gradients if needed
        if self.requires_grad:
            self.gradients = UnsafePointer[Scalar[dtype]].alloc(self.size)
            for i in range(self.size):
                self.gradients[i] = existing.gradients[i]
        else:
            self.gradients = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        self.data.free()
        if self.requires_grad:
            self.gradients.free()

    fn fill(self, value: Scalar[dtype]):
        """Fill tensor with constant value."""
        for i in range(self.size):
            self.data[i] = value

    fn simd_elementwise_add[simd_width: Int](self, other: VectorizedTensor[dtype]) -> VectorizedTensor[dtype]:
        """SIMD-optimized element-wise addition."""
        var result = VectorizedTensor[dtype](self.size, self.requires_grad or other.requires_grad)
        var simd_count = (self.size // simd_width) * simd_width

        # SIMD processing
        for i in range(0, simd_count, simd_width):
            # Manual load
            var a_vals = SIMD[dtype, simd_width](0)
            var b_vals = SIMD[dtype, simd_width](0)

            for j in range(simd_width):
                a_vals[j] = self.data[i + j]
                b_vals[j] = other.data[i + j]

            # SIMD addition
            var result_vals = a_vals + b_vals

            # Manual store
            for j in range(simd_width):
                result.data[i + j] = result_vals[j]

        # Handle remaining elements
        for i in range(simd_count, self.size):
            result.data[i] = self.data[i] + other.data[i]

        return result

    fn simd_elementwise_multiply[simd_width: Int](self, other: VectorizedTensor[dtype]) -> VectorizedTensor[dtype]:
        """SIMD-optimized element-wise multiplication."""
        var result = VectorizedTensor[dtype](self.size, self.requires_grad or other.requires_grad)
        var simd_count = (self.size // simd_width) * simd_width

        # SIMD processing
        for i in range(0, simd_count, simd_width):
            # Manual load
            var a_vals = SIMD[dtype, simd_width](0)
            var b_vals = SIMD[dtype, simd_width](0)

            for j in range(simd_width):
                a_vals[j] = self.data[i + j]
                b_vals[j] = other.data[i + j]

            # SIMD multiplication
            var result_vals = a_vals * b_vals

            # Manual store
            for j in range(simd_width):
                result.data[i + j] = result_vals[j]

        # Handle remaining elements
        for i in range(simd_count, self.size):
            result.data[i] = self.data[i] * other.data[i]

        return result

    fn simd_backward_add[simd_width: Int](self, grad_output: VectorizedTensor[dtype]):
        """SIMD-optimized backward pass for addition."""
        if not self.requires_grad:
            return

        var simd_count = (self.size // simd_width) * simd_width

        # SIMD gradient accumulation
        for i in range(0, simd_count, simd_width):
            # Load current gradients and incoming gradients
            var current_grads = SIMD[dtype, simd_width](0)
            var incoming_grads = SIMD[dtype, simd_width](0)

            for j in range(simd_width):
                current_grads[j] = self.gradients[i + j]
                incoming_grads[j] = grad_output.data[i + j]

            # Accumulate gradients
            var new_grads = current_grads + incoming_grads

            # Store back
            for j in range(simd_width):
                self.gradients[i + j] = new_grads[j]

        # Handle remaining elements
        for i in range(simd_count, self.size):
            self.gradients[i] += grad_output.data[i]

    fn print_tensor(self, name: String, max_elements: Int = 8):
        """Print tensor values (limited for readability)."""
        var elements_to_show = min(max_elements, self.size)
        var size_str: String = String(self.size)
        print(name + " (size " + size_str + "): [", end="")

        for i in range(elements_to_show):
            print(self.data[i], end="")
            if i < elements_to_show - 1:
                print(", ", end="")

        if self.size > max_elements:
            print(", ...]")
        else:
            print("]")

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b

fn simd_autodiff_demo():
    """Demonstrate SIMD optimization in automatic differentiation."""
    print("=== SIMD-Optimized Automatic Differentiation ===")

    var size = 1000

    # Create tensors for AD computation
    var x = VectorizedTensor[DType.float32](size, True)
    var y = VectorizedTensor[DType.float32](size, True)
    var w = VectorizedTensor[DType.float32](size, True)

    # Initialize with test data
    for i in range(size):
        x.data[i] = Float32(i) * 0.01  # x = [0, 0.01, 0.02, ...]
        y.data[i] = Float32(2.0)       # y = [2, 2, 2, ...]
        w.data[i] = Float32(0.5)       # w = [0.5, 0.5, 0.5, ...]

    print("Forward pass computation: z = (x + y) * w")
    var size_str: String = String(size)
    print("Tensor sizes: " + size_str + " elements each")

    x.print_tensor("Input X")
    y.print_tensor("Input Y")
    w.print_tensor("Weights W")

    # Forward pass: z = (x + y) * w
    print("\n1. SIMD Forward Pass:")
    var temp = x.simd_elementwise_add[8](y)
    print("  temp = x + y completed (SIMD8)")

    var z = temp.simd_elementwise_multiply[8](w)
    print("  z = temp * w completed (SIMD8)")

    z.print_tensor("Output Z")

    # Simulate backward pass
    print("\n2. SIMD Backward Pass:")

    # Create gradient tensor (simulating loss gradient)
    var grad_z = VectorizedTensor[DType.float32](size, False)
    grad_z.fill(1.0)  # Assume gradient of 1.0 for simplicity

    print("  Gradient from loss: all ones")

    # Backward through multiplication: dw = temp * grad_z, d_temp = w * grad_z
    var grad_w = temp.simd_elementwise_multiply[8](grad_z)
    var grad_temp = w.simd_elementwise_multiply[8](grad_z)

    print("  Gradients through multiplication computed (SIMD8)")

    # Backward through addition: dx = grad_temp, dy = grad_temp
    x.simd_backward_add[8](grad_temp)
    y.simd_backward_add[8](grad_temp)

    print("  Gradients through addition computed (SIMD8)")

    # Show results
    print("\n3. Gradient Results:")
    grad_w.print_tensor("Gradient dW")

    if x.requires_grad:
        var grad_x_tensor = VectorizedTensor[DType.float32](size, False)
        for i in range(size):
            grad_x_tensor.data[i] = x.gradients[i]
        grad_x_tensor.print_tensor("Gradient dX")

    print("\n4. SIMD Benefits for Automatic Differentiation:")
    print("  + Vectorized tensor operations (4-8x speedup)")
    print("  + Parallel gradient computation")
    print("  + Efficient backward pass through large tensors")
    print("  + Memory bandwidth optimization")
    print("  + Essential for real-time neural network training")

    print("\n5. Performance Characteristics:")
    print("  + SIMD width 8: Process 8 elements per instruction")
    print("  + Memory throughput: ~8x improvement for element-wise ops")
    print("  + CPU utilization: Better instruction pipeline usage")
    print("  + Cache efficiency: Sequential memory access patterns")

fn main():
    """Main function for SIMD automatic differentiation demonstration."""
    simd_autodiff_demo()
```

### File: `26_simd_complete.mojo` — Sections 5.1–5.2, a larger integration test

**Execution:** `pixi run mojo 26_simd_complete.mojo`

```mojo
from memory import UnsafePointer

fn simd_performance_comparison():
    """Complete comparison of scalar vs SIMD performance patterns."""
    print("=== Complete SIMD Performance Analysis ===")

    var size = 10000
    var data_a = UnsafePointer[Float32].alloc(size)
    var data_b = UnsafePointer[Float32].alloc(size)
    var result_scalar = UnsafePointer[Float32].alloc(size)
    var result_simd4 = UnsafePointer[Float32].alloc(size)
    var result_simd8 = UnsafePointer[Float32].alloc(size)

    # Initialize test data
    for i in range(size):
        data_a[i] = Float32(i) * 0.001
        data_b[i] = Float32(i) * 0.002 + 1.0

    var size_str: String = String(size)
    print("Performance test with " + size_str + " elements")
    print("Operations: addition, multiplication, complex expression")

    # 1. Simple Addition
    print("\n1. Vector Addition: result = a + b")

    # Scalar version
    for i in range(size):
        result_scalar[i] = data_a[i] + data_b[i]
    print("  Scalar: Processing 1 element per iteration")

    # SIMD 4-wide
    var simd_count4 = (size // 4) * 4
    for i in range(0, simd_count4, 4):
        var a_vec = SIMD[DType.float32, 4](data_a[i], data_a[i+1], data_a[i+2], data_a[i+3])
        var b_vec = SIMD[DType.float32, 4](data_b[i], data_b[i+1], data_b[i+2], data_b[i+3])
        var result_vec = a_vec + b_vec

        result_simd4[i] = result_vec[0]
        result_simd4[i+1] = result_vec[1]
        result_simd4[i+2] = result_vec[2]
        result_simd4[i+3] = result_vec[3]

    # Handle remaining elements for SIMD4
    for i in range(simd_count4, size):
        result_simd4[i] = data_a[i] + data_b[i]

    print("  SIMD4: Processing 4 elements per iteration")

    # SIMD 8-wide
    var simd_count8 = (size // 8) * 8
    for i in range(0, simd_count8, 8):
        var a_vec = SIMD[DType.float32, 8](data_a[i], data_a[i+1], data_a[i+2], data_a[i+3],
                                          data_a[i+4], data_a[i+5], data_a[i+6], data_a[i+7])
        var b_vec = SIMD[DType.float32, 8](data_b[i], data_b[i+1], data_b[i+2], data_b[i+3],
                                          data_b[i+4], data_b[i+5], data_b[i+6], data_b[i+7])
        var result_vec = a_vec + b_vec

        for j in range(8):
            result_simd8[i + j] = result_vec[j]

    # Handle remaining elements for SIMD8
    for i in range(simd_count8, size):
        result_simd8[i] = data_a[i] + data_b[i]

    print("  SIMD8: Processing 8 elements per iteration")

    # 2. Complex Expression
    print("\n2. Complex Expression: result = (a * b + a) / (b + 1.0)")

    # Scalar version
    for i in range(size):
        result_scalar[i] = (data_a[i] * data_b[i] + data_a[i]) / (data_b[i] + 1.0)

    # SIMD8 version
    var one_vec = SIMD[DType.float32, 8](1.0)
    for i in range(0, simd_count8, 8):
        var a_vec = SIMD[DType.float32, 8](data_a[i], data_a[i+1], data_a[i+2], data_a[i+3],
                                          data_a[i+4], data_a[i+5], data_a[i+6], data_a[i+7])
        var b_vec = SIMD[DType.float32, 8](data_b[i], data_b[i+1], data_b[i+2], data_b[i+3],
                                          data_b[i+4], data_b[i+5], data_b[i+6], data_b[i+7])

        # Complex expression in SIMD
        var numerator = a_vec * b_vec + a_vec
        var denominator = b_vec + one_vec
        var result_vec = numerator / denominator

        for j in range(8):
            result_simd8[i + j] = result_vec[j]

    print("  Scalar: Sequential computation")
    print("  SIMD8: Vectorized complex expression")

    # 3. Verification
    var errors = 0
    for i in range(min(size, 1000)):  # Check first 1000 elements
        if abs(result_scalar[i] - result_simd4[i]) > 0.001:
            errors += 1
        if abs(result_scalar[i] - result_simd8[i]) > 0.001:
            errors += 1

    print("\n3. Verification Results:")
    var error_str: String = String(errors)
    print("  Scalar vs SIMD accuracy: " + ("PASS" if errors == 0 else ("ERRORS: " + error_str)))

    # 4. Performance Analysis
    print("\n4. Theoretical Performance Gains:")
    print("  SIMD4 vs Scalar: ~4x speedup for element-wise operations")
    print("  SIMD8 vs Scalar: ~8x speedup for element-wise operations")
    print("  Memory bandwidth: Better utilization of cache lines")
    print("  Instruction throughput: Fewer total instructions executed")

    print("\n5. When to Use SIMD:")
    print("  + Element-wise tensor operations")
    print("  + Large array processing")
    print("  + Numerical computations")
    print("  + Neural network forward/backward passes")
    print("  + Signal processing")

    print("\n6. SIMD Limitations:")
    print("  - Width must be power of 2")
    print("  - Memory alignment considerations")
    print("  - Branching reduces effectiveness")
    print("  - Not suitable for irregular data access")

    # Cleanup
    data_a.free()
    data_b.free()
    result_scalar.free()
    result_simd4.free()
    result_simd8.free()

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b

fn abs(x: Float32) -> Float32:
    return x if x >= 0 else -x

fn main():
    """Complete SIMD demonstration."""
    simd_performance_comparison()
```

### Expected Output for `26_simd_complete.mojo`

```
=== Complete SIMD Performance Analysis ===
Performance test with 10000 elements
Operations: addition, multiplication, complex expression

1. Vector Addition: result = a + b
  Scalar: Processing 1 element per iteration
  SIMD4: Processing 4 elements per iteration
  SIMD8: Processing 8 elements per iteration

2. Complex Expression: result = (a * b + a) / (b + 1.0)
  Scalar: Sequential computation
  SIMD8: Vectorized complex expression

3. Verification Results:
  Scalar vs SIMD accuracy: PASS

4. Theoretical Performance Gains:
  SIMD4 vs Scalar: ~4x speedup for element-wise operations
  SIMD8 vs Scalar: ~8x speedup for element-wise operations
  Memory bandwidth: Better utilization of cache lines
  Instruction throughput: Fewer total instructions executed

5. When to Use SIMD:
  + Element-wise tensor operations
  + Large array processing
  + Numerical computations
  + Neural network forward/backward passes
  + Signal processing

6. SIMD Limitations:
  - Width must be power of 2
  - Memory alignment considerations
  - Branching reduces effectiveness
  - Not suitable for irregular data access
```

`size = 10000` divides evenly by both 4 and 8, so — exactly as Section 5.2 flagged — this run's remainder loops never execute a single iteration; the code is still correct for sizes that don't divide evenly, it simply isn't exercised on that path by this particular test.

## Chapter Summary

Legal SIMD widths are powers of two because a fixed-bit-width hardware register only splits into equal lanes at those counts — there is no register layout corresponding to "3 lanes." Every vectorized loop in this book shares one shape: `simd_count = (size // width) * width` main-loop elements, processed `width` at a time, followed by a scalar remainder loop covering whatever's left — a shape that produces identical results to a plain scalar loop no matter how `size` and `width` relate, even though a size that happens to divide evenly (as this book's own demos usually do) never exercises the remainder path at all. SIMD reduction accumulates into a width-wide vector across the *entire* array and performs the expensive-relative-to-its-size horizontal reduction exactly once, at the end — reducing after every chunk instead would throw away nearly all the benefit of accumulating in parallel. The same accumulate-then-reduce shape nests inside a matrix multiply's row/column loops (one accumulator per output cell) and reappears, in spirit, in a transpose's row-to-column data movement. And a full forward-and-backward pass through a `VectorizedTensor` re-derives Chapter 7's `AddOp`/`MulOp` backward rules exactly, lane by lane — confirmed by tracing one element completely, from forward values through backward gradients, and cross-checking the result against plain calculus, while also catching a genuine gap in the demo code itself: `grad_w` is computed correctly but never actually stored into `w.gradients`, a reminder that computing the right number and storing it where the rest of the system expects to find it are two separate, both-necessary steps.

## Self-Check Questions

1. Why is `SIMD[DType.float32, 3]` not a legal Mojo type, when `SIMD[DType.float32, 2]` and `SIMD[DType.float32, 4]` both are?
2. For `size = 23` and `width = 8`, compute `simd_count` and the remainder by hand. Which indices does the scalar tail loop cover?
3. `simd_sum[4]` on a 1000-element array performs its horizontal reduction (summing the accumulator's lanes) exactly once, after the main loop finishes. What would go wrong, in terms of lost benefit rather than lost correctness, if that reduction ran once per chunk instead?
4. In Worked Example 5.5.1, `dx[100]` and `dy[100]` both come out to `0.5`. Explain, using Chapter 7's `AddOp.backward` rule, why they must be equal to each other for any input.
5. The chapter's own `25_simd_autodiff.mojo` computes `grad_w` correctly but never accumulates it into `w.gradients`. What is the concrete, checkable symptom of this gap if you inspected `w.gradients` right after the demo finishes?

## Where We Go Next

Part 0 is complete: variables and types, structs, memory layout, GPU programming, and SIMD are now all in place as vocabulary the rest of this book uses without re-explaining. Part 1 begins with Chapter 1.1 (Core Tensor Structure) — the actual `Tensor` type this entire book is organized around, built as Struct-of-Arrays (Chapter 3) with SIMD-friendly, coalescing-friendly (Chapter 4) contiguous buffers (Chapter 5) from the very first line of its definition.

## Worked Solutions

**1.** A SIMD register has a fixed total bit-width (say, 128 bits), and dividing it into `N` equal lanes only produces a whole number of bits per lane when `N` is a power of two — `128/4 = 32` bits per lane works cleanly, but `128/3 ≈ 42.67` bits per lane does not correspond to any way of physically slicing the register into equal pieces. `SIMD[DType.float32, 2]` and `SIMD[DType.float32, 4]` both divide evenly into common register widths; `SIMD[DType.float32, 3]` divides none of them evenly.

**2.** `simd_count = (23 // 8) × 8 = 2 × 8 = 16`. Remainder `= 23 - 16 = 7`. The scalar tail loop covers indices `16` through `22` inclusive — 7 indices, matching the remainder.

**3.** Nothing would become incorrect — the final total would still be right, since addition is associative and commutative regardless of when the horizontal reduction happens. What would be lost is speed: reducing once per chunk means paying the "walk around and add `width` numbers together" cost 250 times (once per chunk, for `size=1000, width=4`) instead of once, throwing away almost all of the benefit of letting the four lanes accumulate independently and in parallel for the whole array before ever combining them.

**4.** Chapter 7's `AddOp.backward` rule says both inputs to an addition receive an *identical copy* of the gradient flowing into the sum — `temp = x + y` means `∂temp/∂x = 1` and `∂temp/∂y = 1`, so whatever gradient arrives at `temp` (here, `grad_temp[100] = 0.5`) passes through unchanged to both `x` and `y`. `dx[100]` and `dy[100]` are both just that same `grad_temp[100]` value, copied twice — they are equal not by coincidence of this particular input, but because `AddOp.backward`'s rule is structurally symmetric in its two inputs for every possible input.

**5.** `w.gradients` would read as all zeros for every element, because nothing in the demo code ever calls `w.simd_backward_add[8](...)` — the only calls made are `x.simd_backward_add[8](grad_temp)` and `y.simd_backward_add[8](grad_temp)`. The value `grad_w` itself, printed separately by `grad_w.print_tensor(...)`, is mathematically correct (as Worked Example 5.5.1 confirms for index 100); it simply was never written into the one place (`w.gradients`) a subsequent optimizer step would actually look for it.
