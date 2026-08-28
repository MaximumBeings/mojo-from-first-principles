# Getting Started

Mojo compiles to native GPU and CPU code through MLIR/LLVM. **Mojo reached 1.0 on August 11, 2026** — after most of this book's example code was written — and 1.0 deprecates the `fn` function-declaration keyword in favor of `def` (same non-raising semantics, just one keyword instead of two; `fn` still compiles today with a warning, but the changelog states that warning becomes a hard compile error in the *next* release after 1.0). Most of this book's Mojo listings still declare functions with `fn`, so expect deprecation warnings when you run them against a real 1.0 toolchain until that migration pass happens — the warnings are noise, not a sign the example is wrong, but they're a real, known gap this page won't pretend doesn't exist.

## Prerequisites

- A supported host: **macOS 15 (Sequoia) or later** on Apple silicon with Xcode 16+; **Linux** on x86-64-v3 (Haswell-class, ~2013 or newer) or ARM64 (AWS Graviton2-class or newer) with glibc 2.34+ (Ubuntu 22.04 LTS or later); or **Windows via WSL** (no native Windows support).
- For the GPU chapters (Part 0.4 onward): an NVIDIA GPU with driver 580+ — Blackwell and Hopper are continuously tested, Ada Lovelace/Ampere/Turing are tested but not continuously, and pre-Turing GPUs need manual configuration — or an AMD GPU with ROCm driver 6.3.3+. The CPU-only chapters (Part 0.1–0.3, Part 6) run on any supported host with no GPU at all.
- Python 3.10+ (Pixi manages this for you inside the project environment; you don't need a matching system Python).

Check what's already on the machine before installing anything:

```bash
uname -sm
python3 --version
mojo --version 2>/dev/null && echo "Mojo already installed!" || echo "Mojo not found - proceed with installation"
pixi --version 2>/dev/null && echo "Pixi already installed!" || echo "Pixi not found - proceed with installation"
nvidia-smi 2>/dev/null && echo "NVIDIA GPU detected!" || echo "No NVIDIA GPU found or driver issues"
```

## 1. Create an isolated environment

[Pixi](https://pixi.sh) is Modular's package and environment manager for Mojo. Now that Mojo has reached 1.0, install against the **stable** `max` channel rather than `max-nightly` — nightly is for tracking unreleased changes, which is no longer what a book targeting a fixed release wants:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pixi --version
```

```bash
pixi init mojo_demo \
  -c https://conda.modular.com/max/ \
  -c conda-forge \
  && cd mojo_demo

pixi add mojo
pixi run mojo --version
```

The final command should print `Mojo 1.0.x`. This book's GPU chapters were originally built against `pixi add max` (the fuller MAX toolchain bundle, which is what wraps `gpu.host`/`gpu.id` and the rest of the accelerator API this book's GPU kernels import) rather than the plain `mojo` package shown above — if a GPU chapter's imports don't resolve after `pixi add mojo` alone, add the toolchain package too:

```bash
pixi add max
```

## 2. Compile a smoke test

```bash
echo 'def main():
    var x = Float32(3)
    var y = Float32(4)
    print(x * y + x)' > hello.mojo

pixi run mojo hello.mojo
```

`3×4=12`, then `12+3=15` — the smoke test should print `15.0`. This uses `def`, already the form this page's own examples use, and already Mojo 1.0's standard going forward.

## 3. Build an executable

Running a source file compiles and executes it in one step; `mojo build` keeps the compiled native executable around, which matters for repeatable benchmarks where you don't want compilation time mixed into a timing measurement:

```bash
pixi run mojo build hello.mojo -o hello
./hello
```

Both the interpreted run from Step 2 and this compiled binary should print the identical `15.0` — the build step changes *when* compilation happens, never *what* the program computes. If they ever disagree, suspect a stale build artifact or a mismatched environment before suspecting the language.

## 4. Run the book's examples

Save any of this book's `.mojo` files into your project directory and run it the same way:

```bash
pixi run mojo mojo_gpu_demo.mojo
```

Expect a deprecation warning naming `fn` on most of this book's listings, for the reason given at the top of this page — the program will still run and produce the output shown in each chapter's "Expected Output" block. Rough hardware expectations for a 1000×1000 matrix multiply, for the chapters that report one:

| GPU | Time |
|---|---|
| T4 (16GB) | ~45–60 ms |
| A10 (24GB) | ~25–35 ms |
| RTX 3080 | ~20–30 ms |
| A100 | ~15–25 ms |

## 5. Verify GPU availability

GPU support is optional for Mojo itself but required starting with Part 0.4. Confirm the platform is actually visible *before* a chapter's code tries to allocate a device buffer — a CPU-only program compiling successfully proves nothing about GPU readiness:

```bash
# NVIDIA (Linux)
nvidia-smi

# AMD (Linux, ROCm)
rocm-smi

# Apple silicon
xcrun -sdk macosx metal --version
```

On NVIDIA or AMD, the GPU model and driver version should appear in the command's output — cross-check the driver version against the minimums in Prerequisites above. On Apple silicon, the Metal command needs Xcode 16 or later to resolve at all; a missing command here means an environment problem, not a bug in the kernel you're about to run.

## 6. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `pixi: command not found` | `PATH` not updated in this shell | `source ~/.bashrc` |
| `mojo: command not found` | Invoking `mojo` outside the Pixi environment | Prefix every invocation with `pixi run`, e.g. `pixi run mojo --version` |
| `max` / `mojo` package not found | Channel misconfigured, or still pointed at `max-nightly` | `pixi info` to check the active channels; re-init against `https://conda.modular.com/max/` |
| Deprecation warning naming `fn` | This book's code predates Mojo 1.0's `fn`→`def` unification | Expected for now on most listings — see the note at the top of this page; the program still runs |
| GPU not detected | Driver missing, wrong generation, or WSL/GPU passthrough not configured | Confirm with `nvidia-smi`/`rocm-smi` first; the CPU-only chapters work without a GPU at all |

Continue to [Part 0: Mojo Foundations](part0/01-variables-and-types.md) once the smoke test prints `15.0`.

## 7. Keep your work

```bash
git init
git add *.mojo pixi.toml
git commit -m "Initial Mojo project"
```

See [Appendix A](appendix/installation-setup.md#a4-preserve-cloud-work) for the fuller version of this step — a `tar` archive and an off-instance copy, for a cloud GPU instance whose disk won't survive termination.
