# Getting Started

Mojo compiles to native GPU and CPU code through MLIR/LLVM, so the fastest path to running the examples in this book is a Linux box with an NVIDIA GPU. Everything below has been exercised on AWS `g4dn.xlarge` (T4), Lambda Cloud (A10 / Quadro RTX 6000 / A100), and plain Ubuntu 22.04.

## Prerequisites

- An NVIDIA GPU with CUDA support (T4, A10, RTX-series, or newer) for the GPU chapters — the CPU-only chapters (Part 0.1–0.3, Part 6) run anywhere.
- Ubuntu 18.04+ or a similar Linux distribution.
- Python 3.8+ (Mojo's toolchain is distributed through the Python-based `pixi` package manager).

Check what's already on the machine before installing anything:

```bash
mojo --version 2>/dev/null && echo "Mojo already installed!" || echo "Mojo not found - proceed with installation"
pixi --version 2>/dev/null && echo "Pixi already installed!" || echo "Pixi not found - proceed with installation"
nvidia-smi 2>/dev/null && echo "NVIDIA GPU detected!" || echo "No NVIDIA GPU found or driver issues"
```

## 1. Launch a GPU instance

**AWS (recommended for beginners):** `g4dn.xlarge` (1× T4, 16GB), Deep Learning AMI (Ubuntu 20.04/22.04), security group open on port 22.

```bash
chmod 400 your_key.pem
ssh -i your_key.pem ubuntu@<EC2_PUBLIC_IP>
```

**Lambda Cloud (alternative):** A10 (24GB), Quadro RTX 6000, or A100 on the Ubuntu 22.04 + Lambda Stack image.

```bash
chmod 400 your_key.pem
ssh -i your_key.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

!!! warning "Lambda instances are ephemeral by default"
    If you do **not** attach a persistent filesystem, everything on a Lambda instance is lost on termination — there is no recovery. Attach at least a 100GB filesystem for any real project, and there is no "Stop" option, only Restart (keeps data while running) and Terminate (deletes the instance *and* its ephemeral storage). Back up before terminating:
    ```bash
    tar -czvf backup.tar.gz ~/quickstart/
    scp -i your_key.pem ubuntu@<INSTANCE_PUBLIC_IP>:~/backup.tar.gz .
    ```

## 2. System packages

```bash
sudo apt update && sudo apt install -y \
    curl git unzip build-essential \
    libpython3-dev python3-pip pkg-config

# Verify CUDA is visible to the OS
nvidia-smi
nvcc --version
```

## 3. Install Pixi and Mojo

[Pixi](https://pixi.sh) is Modular's package and environment manager for Mojo — it resolves the `max` (Modular Accelerated Xecution) toolchain, which bundles the Mojo compiler.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pixi --version
```

Create a project against Modular's nightly channel and add the toolchain:

```bash
pixi init mojo_demo \
  -c https://conda.modular.com/max-nightly/ \
  -c conda-forge \
  && cd mojo_demo

pixi add max
pixi run mojo --version

# Smoke test
echo 'def main(): print("Hello, Mojo!")' > hello.mojo
pixi run mojo hello.mojo
```

You do not need to run `pixi login` — the workshop-style flow above pulls Mojo directly from the `max-nightly` channel via Pixi.

## 4. Run the book's examples

Save any of this book's `.mojo` files into your project directory and run it the same way:

```bash
pixi run mojo mojo_gpu_demo.mojo
```

Expected shape of the output (exact numbers vary by GPU):

```
Using Mojo version: 24.5.0
Hardware detected: Tesla T4

=== SIMD Vector Operations ===
Vector A: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Vector B: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
A + B: [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
SIMD width: 8 elements processed simultaneously

=== Matrix Multiplication Performance ===
Matrix size: 1000x1000
Mojo time: 45.2 ms
Performance: 44.2 GFLOPS
```

Rough hardware expectations for a 1000×1000 matrix multiply:

| GPU | Time |
|---|---|
| T4 (16GB) | ~45–60 ms |
| A10 (24GB) | ~25–35 ms |
| RTX 3080 | ~20–30 ms |
| A100 | ~15–25 ms |

## 5. Compilation model, in brief

Mojo source → **MLIR** → optimization passes (vectorization, loop fusion) → **LLVM IR** → native machine code. It's ahead-of-time compiled, so a `pixi run mojo build file.mojo -o binary` step produces a standalone executable:

```bash
pixi run mojo build mojo_gpu_demo.mojo -o mojo_demo
./mojo_demo
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `pixi: command not found` | `source ~/.bashrc` to pick up the exported `PATH` |
| `max package not found` | Check channel config with `pixi info` — the `max-nightly` channel must be present |
| `mojo: command not found` | Prefix every invocation with `pixi run`, e.g. `pixi run mojo --version` |
| GPU not detected | Confirm with `nvidia-smi`; the CPU-only chapters still work without one |

## 7. Keep your work

```bash
git init
git add *.mojo pixi.toml
git commit -m "Initial Mojo project"
```

With the toolchain confirmed, continue to [Part 0: Mojo Foundations](part0/01-variables-and-types.md).
