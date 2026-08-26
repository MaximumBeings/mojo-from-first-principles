# Getting Started

This edition targets **Mojo 1.0.0**, released August 11, 2026. Pinning the compiler matters: Mojo is evolving quickly, and examples that used `fn`, unprefixed standard-library imports, `UnsafePointer[T].alloc()`, or host-sized `Int` GPU arguments belong to older language generations.

## Prerequisites

CPU chapters need only a supported Mojo host. GPU chapters additionally need a supported accelerator and toolchain. Mojo 1.0 supports Apple silicon on macOS 15+, Linux on x86-64 or ARM64, and Windows through WSL; Python 3.10–3.14 is supported. On NVIDIA, use a Turing-or-newer GPU and a current driver, or configure a compatible external `ptxas` as the official requirements describe.

```bash
uname -sm
python3 --version
mojo --version 2>/dev/null || true
```

**Manual worked example.** A suitable Apple host might print `Darwin arm64`, Python `3.14.x`, and `Mojo 1.0.0`. A Linux GPU host should print `Linux x86_64`; run `nvidia-smi` separately and confirm that the GPU and driver appear before attempting Chapter 9.

## 1. Create an isolated stable environment

An isolated environment makes the book reproducible and prevents a later nightly compiler from silently changing behavior. The stable Python package is the shortest cross-platform route.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install "mojo==1.0.0"
mojo --version
```

**Manual worked example.** The four state changes are: install the environment manager, create `.venv`, activate it, and place Mojo 1.0.0 inside it. The final command must begin with `Mojo 1.0.0`; if it reports a nightly or older 0.x build, stop and correct the environment before judging an example.

## 2. Compile a smoke test

Mojo 1.0 uses `def` for functions. A tiny program checks the parser, compiler, runtime loader, and terminal path before the book adds tensors or GPU code.

```mojo
def main():
    var x = Float32(3)
    var y = Float32(4)
    print(x * y + x)
```

Save it as `hello.mojo`, then run:

```bash
mojo hello.mojo
```

**Manual worked example.** The multiply runs first: `3×4=12`. Addition then reuses `x`: `12+3=15`. The program must print `15.0`, the same running value used in Chapters 6–8.

## 3. Build an executable

Running a source file compiles and executes it. `mojo build` keeps the native executable, which is useful for repeatable benchmarks.

```bash
mojo build hello.mojo -o hello
./hello
```

**Manual worked example.** Both commands should produce the same numerical result, 15. The first creates the executable; the second runs that artifact without asking the compiler to rebuild the source.

## 4. Verify GPU availability

GPU support is optional for Mojo itself but required for Chapter 9 onward. Verify the platform before allocating device buffers; do not infer GPU readiness merely because CPU code compiled.

```bash
# NVIDIA Linux
nvidia-smi

# Apple silicon
xcrun -sdk macosx metal --version
```

**Manual worked example.** On NVIDIA, identify the GPU model and driver row and compare them with Mojo's current compatibility table. On Apple silicon, the Metal command must resolve from Xcode 16 or later. A missing command is an environment failure, not a kernel bug.

## 5. Read example labels accurately

The book uses two kinds of code. A **complete program** includes imports and `main()` and can be saved as shown. An **implementation excerpt** focuses on one algorithm and assumes the chapter's previously introduced types. Every code block is followed by a manual trace with concrete inputs; verify that trace before optimizing or composing the code.

```text
concept → hand calculation → implementation → compiler/run check → optimization
```

**Manual worked example.** For vector addition `[1,2]+[10,20]`, calculate `[11,22]` first. Run the scalar implementation and compare. Only after equality is established should SIMD width or GPU block size change. This order prevents a fast wrong answer from becoming the baseline.

## 6. Troubleshooting

Classify failures by layer. Installation errors happen before parsing; language migration errors name syntax or APIs; GPU environment errors appear while compiling or launching device code; numerical mismatches occur only after a program runs.

| Symptom | Likely cause | Action |
|---|---|---|
| `mojo: command not found` | Environment not active | `source .venv/bin/activate` |
| Error at `fn` | Pre-1.0 source | Replace declarations/function types with `def` |
| `UnsafePointer...alloc` missing | Pre-1.0 allocation API | Use free-standing `alloc[T](count)` or a safe owning buffer |
| GPU rejects `Int` argument | Mojo 1.0 device ABI rule | Pass `Int32`/`UInt32`/`Int64`/`UInt64` and convert inside the kernel |
| GPU unavailable | Driver/toolchain mismatch | Recheck the official platform requirements |

Continue to [Part 0: Mojo Foundations](part0/01-variables-and-types.md) once the smoke test prints 15.
