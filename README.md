# Mojo From First Principles

A concept-first technical book that builds a tensor and reverse-mode automatic-differentiation framework, then applies it to GPU kernels, neural networks, and quantitative finance.

This edition targets **Mojo 1.0.0**. Every section begins with prose, and every implementation is paired with a concrete hand calculation or verification trace.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install "mojo==1.0.0"
mkdocs build --strict
```

The rendered book is published at <https://maximumbeings.github.io/mojo-from-first-principles/>.
