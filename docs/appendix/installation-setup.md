# Appendix A: Installation & Environment Setup

The main [Getting Started](../getting-started.md) page establishes a local Mojo 1.0.0 environment. This appendix adds a reproducible Linux GPU path and makes each verification checkpoint explicit.

## A.1 Provision a Linux GPU host

Choose Ubuntu 22.04 or later and a supported accelerator. Persistent storage is worth configuring before installation because cloud GPU root disks are often ephemeral.

```bash
sudo apt update
sudo apt install -y curl git build-essential python3 python3-venv
nvidia-smi
```

**Manual worked example.** The package commands prepare the host; they do not prove the GPU works. `nvidia-smi` must separately show a supported model and driver. Record both values so a later launch failure can be compared against the Mojo compatibility table.

## A.2 Install the pinned compiler

Use an isolated environment and pin the release used to edit this book.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install "mojo==1.0.0"
mojo --version
```

**Manual worked example.** The expected last line starts with `Mojo 1.0.0`. The environment is reproducible because the package version is explicit; omitting the pin lets a future release change parser or library behavior.

## A.3 Run and build

Interpretation and ahead-of-time building should agree on the same deterministic smoke test.

```bash
mojo hello.mojo
mojo build hello.mojo -o hello
./hello
```

**Manual worked example.** For the `x*y+x` smoke test in Getting Started, both runs print 15. A mismatch means the two commands are resolving different environments or artifacts; fix that before benchmarking.

## A.4 Preserve cloud work

Treat an unattached instance disk as disposable. Commit source and environment metadata, then copy any uncommitted results before termination.

```bash
git add .
git commit -m "Save Mojo 1.0 work"
tar -czf mojo-work.tar.gz .
scp mojo-work.tar.gz <local-destination>
```

**Manual worked example.** `git status` should be clean after the commit. The archive provides a second recovery path; list it with `tar -tzf mojo-work.tar.gz` and confirm that source and environment files are present before terminating the instance.
