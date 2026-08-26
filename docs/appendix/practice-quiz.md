# Appendix B: Practice Quiz

Test your recall of Part 0's GPU-programming and SIMD-matrix material. Answers are collapsed — click to reveal.

## Multiple Choice

**1. What is the execution command to run the Mojo GPU demo?**

A) `pixi run mojo build mojo_gpu_demo.mojo`
B) `pixi run mojo mojo_gpu_demo.mojo`
C) `python mojo_gpu_demo.mojo`
D) `mojo compile mojo_gpu_demo.mojo`

??? success "Answer"
    **B) `pixi run mojo mojo_gpu_demo.mojo`**

    This compiles and runs the file in one command. Option A only builds an executable without running it; C and D use syntax that doesn't apply to Mojo.

**2. In the GPU basics demo, what are the expected results for `result[0:5]`?**

A) `[0.0, 3.0, 6.0, 9.0, 12.0]`
B) `[1.0, 2.0, 3.0, 4.0, 5.0]`
C) `[0.0, 2.0, 4.0, 6.0, 8.0]`
D) `[0.0, 1.0, 4.0, 9.0, 16.0]`

??? success "Answer"
    **A) `[0.0, 3.0, 6.0, 9.0, 12.0]`**

    With `a[i] = i` and `b[i] = i * 2`, `result[i] = i + i*2 = i*3`, giving `[0, 3, 6, 9, 12]` for the first five elements — see [Chapter 3.1](../part2/01-element-wise-operations.md#31-addition-and-subtraction).

**3. What matrix size is used in the SIMD matrix operations demo?**

A) 32 × 32
B) 64 × 64
C) 128 × 128
D) 1000 × 1000

??? success "Answer"
    **B) 64 × 64** — `var size = 64`, chosen as a balance between demonstrating the concept and keeping demo runtime short. See [Chapter 4.1](../part2/02-matrix-operations.md#41-matrix-multiplication).

## True / False

**4. The `Matrix` struct automatically manages GPU memory allocation.**

??? success "Answer"
    **FALSE.** `Matrix` uses `UnsafePointer[Float32].alloc()`, which allocates CPU memory. This demo's matrix operations are CPU-side SIMD; the GPU-memory-management story is [Chapter 2.3](../part1/06-memory-management-system.md#23-gpu-memory-management).

**5. Matrix transpose swaps rows and columns: `output[j][i] = input[i][j]`.**

??? success "Answer"
    **TRUE.** `matrix_transpose_simd` implements exactly this by writing `output.set(j + k, i, row_vals[k])`. See [Chapter 4.2](../part2/02-matrix-operations.md#42-transpose-operations).

**6. The demo verifies correctness by comparing scalar and SIMD matrix multiplication results.**

??? success "Answer"
    **TRUE.** `scalar_matrix_multiply` and `simd_matrix_multiply[4]` run on the same inputs and are compared element-by-element with a `0.001` tolerance — the same "check against a known-correct baseline" pattern used for every optimized kernel in this book, including [gradient checking](../part6/02-advanced-features.md#124-debugging-and-profiling-tools) in Chapter 12.

## Fill in the Blank

**7. The `Matrix` struct uses `UnsafePointer[_______]` for memory allocation.**

??? success "Answer"
    **`Float32`** — `UnsafePointer[Float32].alloc(rows * cols)`.

**8. According to the sample results, what is the value at position `[1,0]` in the result matrix?**

??? success "Answer"
    **`64.0`** — the second row of the sample output is `[64.0, 65.0, 66.0, 67.0]`, so `[1,0] = 64.0` (matrix `A` filled with sequential values, multiplied by an identity `B`). See the [expected output](../part2/02-matrix-operations.md#41-matrix-multiplication) in Chapter 4.1.

**9. The SIMD matrix multiplication uses a compile-time parameter called `_______` to determine vector width.**

??? success "Answer"
    **`simd_width`** — `fn simd_matrix_multiply[simd_width: Int](...)`, resolved at compile time per [Chapter 10.3](../part5/02-performance-optimization.md#103-compile-time-optimizations).

**10. Matrix B in the demo is initialized as an `_______` matrix using `fill_identity()`.**

??? success "Answer"
    **identity** — multiplying any matrix `A` by an identity matrix returns `A` unchanged, which is exactly why it's used to verify the multiplication implementation.

## Answer Key

| # | Type | Answer |
|---|---|---|
| 1 | Multiple Choice | B |
| 2 | Multiple Choice | A |
| 3 | Multiple Choice | B |
| 4 | True/False | FALSE |
| 5 | True/False | TRUE |
| 6 | True/False | TRUE |
| 7 | Fill in the Blank | Float32 |
| 8 | Fill in the Blank | 64.0 |
| 9 | Fill in the Blank | simd_width |
| 10 | Fill in the Blank | identity |
