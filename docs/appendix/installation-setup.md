# Appendix A: Installation & Environment Setup

The full walkthrough — with expected-output benchmarks across T4 / A10 / RTX 3080 / A100 and a troubleshooting table — lives on the [Getting Started](../getting-started.md) page. This appendix mirrors that page's own four-stage shape (provision, install, run/build, preserve) as a compact reference, for whenever you just need the commands without the surrounding explanation.

## A.1 Provision a Linux GPU host

- An NVIDIA GPU with CUDA support (T4, A10, RTX-series, or newer) for the GPU chapters — the CPU-only chapters (Part 0.1–0.3, Part 6) run on any machine.
- Ubuntu 18.04+ or a similar Linux distribution.
- Python 3.8+ (Mojo's toolchain is distributed through the Python-based `pixi` package manager).

Check what's already on the machine before installing anything:

```bash
mojo --version 2>/dev/null && echo "Mojo already installed!" || echo "Mojo not found - proceed with installation"
pixi --version 2>/dev/null && echo "Pixi already installed!" || echo "Pixi not found - proceed with installation"
nvidia-smi 2>/dev/null && echo "NVIDIA GPU detected!" || echo "No NVIDIA GPU found or driver issues"
```

Any of these three environments works — pick whichever is already available:

```bash
# AWS: g4dn.xlarge (1x T4, 16GB), Deep Learning AMI (Ubuntu 20.04/22.04)
chmod 400 your_key.pem
ssh -i your_key.pem ubuntu@<EC2_PUBLIC_IP>

# Lambda Cloud: A10 (24GB), Quadro RTX 6000, or A100, Ubuntu 22.04 + Lambda Stack
chmod 400 your_key.pem
ssh -i your_key.pem ubuntu@<INSTANCE_PUBLIC_IP>

# Local Ubuntu box (any of the CPU-only chapters, or with a local NVIDIA GPU)
sudo apt update && sudo apt upgrade -y
```

```bash
sudo apt install -y \
    curl git unzip build-essential \
    libpython3-dev python3-pip pkg-config

# Confirm the OS can see the GPU before installing anything Mojo-specific --
# a driver problem here will look like a Mojo problem later otherwise.
nvidia-smi
nvcc --version
```

!!! danger "Cloud GPU disks are usually ephemeral"
    A cloud GPU instance's root disk rarely survives termination by default (Lambda Cloud instances have no persistent storage unless you explicitly attach one). Treat everything on the instance as temporary until it's backed up — Section A.4 below covers exactly how.

## A.2 Install Pixi and the pinned Mojo toolchain

[Pixi](https://pixi.sh) is Modular's package and environment manager for Mojo — it resolves the `max` (Modular Accelerated Xecution) toolchain, which bundles the Mojo compiler, against Modular's own conda channel. This is the one real, working install path this book's examples were run against; a plain `pip install mojo` is not how Mojo is distributed, and pinning by an ordinary package version number the way a pure-Python library would be pinned doesn't apply here — Pixi's lockfile is what makes an environment reproducible.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pixi --version
```

Create a project against Modular's channel and add the toolchain:

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

No `pixi login` step is required — this flow pulls the toolchain directly from the `max-nightly` channel. If `pixi run mojo --version` doesn't print a version string, `pixi info` will show whether the `max-nightly` channel actually resolved.

## A.3 Run and build

Any `.mojo` file in this book runs the same way, whether it's a single interpreted run or an ahead-of-time compiled binary — both should produce identical output on the same input, which is itself a useful smoke test:

```bash
# Interpreted run
pixi run mojo <filename>.mojo

# Compile to a standalone executable, then run the binary directly
pixi run mojo build <filename>.mojo -o <output_name>
./<output_name>
```

If the interpreted run and the compiled binary ever disagree on the same file, that's a signal to check for a stale build artifact or a mismatched environment before trusting either result. Mojo's pipeline is source → MLIR → optimization passes (vectorization, loop fusion) → LLVM IR → native code, which is why the compiled path can be faster but should never be *different* — it's the same optimizations applied ahead of time instead of at the point of running `mojo <file>.mojo` directly.

```bash
mojo --version 2>&1 | head -1
pixi run mojo --help
```

## A.4 Preserve cloud work

Treat a cloud instance's disk as temporary, whether or not the provider advertises persistence: commit source and environment metadata, then copy any uncommitted results off the instance before terminating it.

```bash
# 1. Commit everything version-controllable
git add .
git commit -m "Save Mojo work"

# 2. Archive the whole project directory, including files git doesn't track
tar -czf mojo-work.tar.gz .

# 3. Copy the archive to a machine that will still exist tomorrow
scp -i your_key.pem mojo-work.tar.gz you@your-local-machine:~/backups/
```

Verify both copies before terminating the instance, not after:

```bash
git status              # should report no pending changes
tar -tzf mojo-work.tar.gz | head -20   # confirm source and config files are actually inside
```

`git status` showing a clean tree only proves the *tracked* files are safe — build artifacts, datasets, and anything covered by `.gitignore` still need the `tar` step. Running both checks before terminating, rather than trusting that the commands above "should have worked," is the same discipline this book applies to every other claimed result: verify, don't assume.

See [Getting Started](../getting-started.md) for the expected-output benchmarks across T4 / A10 / RTX 3080 / A100, and its troubleshooting table for common `pixi`/`mojo` errors.
