# Mojo From First Principles

**Building a GPU-Native Automatic Differentiation Framework, One Layer at a Time**

> "The gap between writing a training loop that runs and understanding every allocation, stride, and gradient it touches is exactly the size of this book."

This is a practitioner's guide to [Mojo](https://mojolang.org/) — the systems-programming language that pairs Python-like ergonomics with explicit ownership, SIMD vectorization, and portable GPU kernels — taught by building an automatic differentiation (AD) engine. This edition targets **Mojo 1.0.0**.

Every chapter grows the same codebase. You start with Mojo's type system and memory model, build a tensor library with proper strides and views, wire up a computational graph and a reverse-mode differentiation engine, push the hot paths onto the GPU, assemble neural network layers on top, and finish by pointing the whole framework at real quantitative-finance problems — bond pricing, credit spreads, and portfolio risk — where a wrong gradient costs real money.

The design principles carried through the whole book:

- **Performance-first.** GPU-native execution with Struct-of-Arrays (SoA) memory layouts chosen for coalesced access, not convenience.
- **Safety first, unsafety contained.** Graph identity uses owned value IDs and raw pointers appear only at allocation or kernel boundaries.
- **Measure before specializing.** Dynamic shapes stay dynamic; compile-time specialization is reserved for measured hot paths.
- **Numerically checkable.** Each implementation is preceded or followed by a concrete hand calculation and a result the reader can verify.

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .lg .middle } **14 chapters**

    Part 0 through Part 7, from variable declaration to Monte Carlo gradients.

- :material-code-tags:{ .lg .middle } **Worked code demos**

    Every implementation is paired with a step-by-step numerical trace.

- :material-file-document-multiple:{ .lg .middle } **2 appendices**

    Environment setup for cloud GPUs, plus a self-check quiz.

- :material-finance:{ .lg .middle } **Real financial models**

    Zero-coupon bond pricing and Z-spread risk analytics, GPU-accelerated.

</div>

## How the book is organized

The sequence moves from language and memory invariants to tensor operations, graph differentiation, accelerator execution, and finally applications. Each part depends only on contracts established earlier, so no concept is required before it is introduced.

| Part | Focus |
|---|---|
| **Part 0 — Mojo Foundations** | Variables and types, structs, memory layout (AoS vs SoA), GPU programming basics, SIMD |
| **Part 1 — Core Tensor Infrastructure** | Tensor structs, strides/views/alignment/broadcasting, factory and random tensor creation, specialized (identity/diagonal/sparse/triangular) tensors, device abstraction, memory management |
| **Part 2 — Basic Tensor Operations** | Element-wise ops, matrix operations and contraction, reductions |
| **Part 3 — Computational Graph Foundation** | Graph-owned value IDs, operation nodes, reverse topological order |
| **Part 4 — Automatic Differentiation Engine** | The backward pass, chain rule, reverse-mode AD, gradient accumulation |
| **Part 5 — GPU Acceleration and Performance** | Portable `std.gpu` kernels, coalescing, tiling, SIMD, benchmarking |
| **Part 6 — Neural Network Building Blocks** | Linear layers, activations, losses, optimizers, a full trained network |
| **Part 7 — Financial Computing Applications** | Bond pricing, credit spread/risk analytics, portfolio optimization, Monte Carlo |

Start with [Getting Started](getting-started.md) to stand up a Mojo + GPU environment, or jump straight into [Part 0: Mojo Foundations](part0/01-variables-and-types.md) if your toolchain is already installed.

!!! note "How to read the code"
    Complete programs contain their imports and `main()`. Shorter implementation excerpts assume types introduced earlier in the chapter. In both cases, work the concrete numbers by hand first; then compare the implementation with that trace. Do not treat illustrative benchmark output as a hardware guarantee.
