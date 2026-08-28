# Mojo From First Principles — Under Development

**Building a GPU-Native Automatic Differentiation Framework, One Layer at a Time**

!!! warning "This book is under active development"
    Chapters are being rewritten one at a time into a fuller format — intuition, technical background, and hand-worked numeric examples ahead of every block of code. Finished so far: all of **Part 0 — Mojo Foundations** (Chapters 1–5: Variables and Types, Struct Design Patterns, Memory Layout Strategies, GPU Programming Introduction, and SIMD and Vectorization), plus all six chapters of **Part 1 — Core Tensor Infrastructure** (Chapter 6: The Tensor — shape, strides, creation, devices, and validation; Chapter 7: Memory Layout Design — strides revisited, zero-copy views, alignment/padding, and broadcasting; Chapter 8: Tensor Creation Functions — factories, random generation, and I/O; Chapter 9: Specialized Tensor Types — identity, diagonal, sparse, and triangular structures; Chapter 10: Device Abstraction Layer — a second, minimal device-discovery implementation plus device-aware memory allocation; Chapter 11: Memory Management System — shared-ownership reference counting, bump allocation, and GPU memory pooling), plus all three chapters of **Part 2 — Basic Tensor Operations** (Chapter 12: Element-wise Operations — addition, subtraction, multiplication, division, power, exponential, and GPU-kernel broadcasting; Chapter 13: Matrix Operations — tensor contraction and matrix multiplication traced both scalar and SIMD lane-by-lane, transpose, reshape/view semantics, and structure-aware linear algebra via Chapter 9's specialized tensors; Chapter 14: Reduction Operations — tree-reduction sum/mean, max/argmax, L2 norm and gradient clipping, and variance/standard deviation), plus the first chapter of **Part 3 — Computational Graph Foundation** (Chapter 15: Graph Node Architecture — the `Differentiable` trait, `GraphNode`/`ComputationGraph` recording, and topological ordering, checked against how reverse-mode automatic differentiation is implemented in production), plus both chapters of **Part 4 — Automatic Differentiation Engine** (Chapter 16: Backward Function Implementation — the multivariable chain rule as sum-over-paths, `AddOp`/`MulOp`/`ExpOp`'s backward rules, `MatMulOp`'s backward derived from index-summation first principles and verified on both gradients, a full registry expansion covering `SubOp`/`DivOp`/`PowOp`/`LogOp`/`SqrtOp`, the activation and trigonometric gradients `ReluOp`/`SigmoidOp`/`TanhOp`/`SinOp`/`CosOp`, the reduction and shape gradients `SumOp`/`MaxOp`/`ReshapeOp`/`TransposeOp`, and the implicit function theorem for differentiating through an iterative solver; Chapter 17: Gradient Computation Engine — the full reverse pass traced end to end, a precise gap found between `GraphNode.grad` and the fields `accumulate_gradient` actually writes (with the `grad_fn_index`-based fix), the tensor-aliasing question from Chapter 16 resolved definitively, and a broadcasting-gradient-reduction gap verified against Chapter 12.4's own numbers). Parts 1 through 4 are now fully revised and form a complete, working autograd engine with a registry of gradient rules checked against a real 74-operation reference implementation, plus all of **Part 5 — GPU Acceleration and Performance** (Chapter 18: GPU Kernel Implementation — ceiling-division launch configurations traced to the exact wasted thread, memory coalescing counted in actual transactions on a real eight-field bond record, a naive convolution kernel's redundant reads counted exactly and then removed with a shared-memory tile, and warp-level shuffle instructions with the one bounds-check interaction that can silently break them; Chapter 19: Performance Optimization Techniques — SIMD vectorization's main-loop-plus-remainder shape, loop unrolling and loop fusion disentangled into two genuinely different savings and counted separately, compile-time specialization as the real mechanism behind this book's zero-cost-abstraction claim, and a benchmarking harness built around the one measurement mistake — an unsynchronized GPU timer — that invalidates everything else), plus the first chapter of **Part 6 — Neural Network Building Blocks** (Chapter 20: Neural Network Layers — He and Xavier initialization traced through an actual Box-Muller draw, a full five-layer forward/backward/update pass hand-verified two layers deep, a genuine confirmed scale-mismatch bug between this chapter's loss and its own gradient function, and the architectural fact that this network's hand-written backprop never once calls into the `Tensor`/`GraphNode` autograd engine built across Parts 3 and 4 — built from real author source with genuinely captured training output, not a fabricated one). The rest of Part 6 and all of Part 7 are still in their original, denser draft form and will be revised in the same style, chapter by chapter.

> "The gap between writing a training loop that runs and understanding every allocation, stride, and gradient it touches is exactly the size of this book."

This is a practitioner's guide to [Mojo](https://www.modular.com/mojo) — the systems-programming language that pairs Python-like ergonomics with C/C++-level performance, SIMD vectorization, and direct GPU kernel authorship — taught the way the language rewards being learned: by building something real. The running project is an automatic differentiation (AD) engine, built from the language's fundamentals all the way up to a trained neural network and a set of production financial models.

Every chapter grows the same codebase. You start with Mojo's type system and memory model, build a tensor library with proper strides and views, wire up a computational graph and a reverse-mode differentiation engine, push the hot paths onto the GPU, assemble neural network layers on top, and finish by pointing the whole framework at real quantitative-finance problems — bond pricing, credit spreads, and portfolio risk — where a wrong gradient costs real money.

The design principles carried through the whole book:

- **Performance-first.** GPU-native execution with Struct-of-Arrays (SoA) memory layouts chosen for coalesced access, not convenience.
- **Memory safety.** Mojo's ownership system and lifetime tracking replace manual bookkeeping without giving up raw-pointer performance.
- **Zero runtime overhead.** Tensor shapes are checked and specialized at compile time wherever Mojo allows it.
- **Financial computing ready.** The tensor and autograd primitives are validated against portfolio analytics, bond pricing, and risk calculations — domains with zero tolerance for silent numerical error.

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .lg .middle } **14 chapters**

    Part 0 through Part 7, from variable declaration to Monte Carlo gradients.

- :material-code-tags:{ .lg .middle } **40+ code demos**

    Every section ships a runnable `.mojo` file and its expected output.

- :material-file-document-multiple:{ .lg .middle } **2 appendices**

    Environment setup for cloud GPUs, plus a self-check quiz.

- :material-finance:{ .lg .middle } **Real financial models**

    Zero-coupon bond pricing and Z-spread risk analytics, GPU-accelerated.

</div>

## How the book is organized

| Part | Focus |
|---|---|
| **Part 0 — Mojo Foundations** | Variables and types, structs, memory layout (AoS vs SoA), GPU programming basics, SIMD |
| **Part 1 — Core Tensor Infrastructure** | Tensor structs, strides/views/alignment/broadcasting, factory and random tensor creation, specialized (identity/diagonal/sparse/triangular) tensors, device abstraction, memory management |
| **Part 2 — Basic Tensor Operations** | Element-wise ops, matrix operations and contraction, reductions |
| **Part 3 — Computational Graph Foundation** | Graph nodes, gradient function traits, topological sort |
| **Part 4 — Automatic Differentiation Engine** | The backward pass, chain rule, reverse-mode AD, gradient accumulation |
| **Part 5 — GPU Acceleration and Performance** | Custom CUDA-style kernels, memory coalescing, SIMD, benchmarking |
| **Part 6 — Neural Network Building Blocks** | Linear layers, activations, losses, optimizers, a full trained network |
| **Part 7 — Financial Computing Applications** | Bond pricing, credit spread/risk analytics, portfolio optimization, Monte Carlo |

Start with [Getting Started](getting-started.md) to stand up a Mojo + GPU environment, or jump straight into [Part 0: Mojo Foundations](part0/01-variables-and-types.md) if your toolchain is already installed.

!!! note "Source material"
    This site was assembled from the author's working notes, GPU demo scripts, and financial-computing prototypes — each chapter in Parts 2 through 7 introduces the concepts the same way the framework itself was built: starting from a small, runnable Mojo example and generalizing it into the production pattern used earlier in the book.
