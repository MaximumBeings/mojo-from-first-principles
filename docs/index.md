# Mojo From First Principles — Under Development

**Building a GPU-Native Automatic Differentiation Framework, One Layer at a Time**

!!! warning "This book is under active development"
    Chapters are being rewritten one at a time into a fuller format — intuition, technical background, and hand-worked numeric examples ahead of every block of code. Finished so far: all of **Part 0 — Mojo Foundations** (Chapters 1–5: Variables and Types, Struct Design Patterns, Memory Layout Strategies, GPU Programming Introduction, and SIMD and Vectorization), plus the first chapter of **Part 1 — Core Tensor Infrastructure** (Chapter 6: The Tensor — shape, strides, creation, devices, and validation). The rest of Part 1 onward is still in its original, denser draft form and will be revised in the same style, chapter by chapter.

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
