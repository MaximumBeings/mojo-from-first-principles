# Appendix A: Installation & Environment Setup

The full walkthrough — launching a GPU instance, installing Pixi and Mojo, running your first kernel, and troubleshooting — lives on the [Getting Started](../getting-started.md) page. This appendix collects the two alternate provisioning scripts used to prepare this book's examples, for reference.

## A.1 Lambda Cloud, minimal (workshop-style)

```bash
# SECTION A -- On your Lambda GPU instance (Ubuntu 22.04 + Lambda Stack)
sudo apt update && sudo apt install -y \
    curl git unzip build-essential libpython3-dev python3-pip pkg-config

curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# SECTION B -- Mojo environment via Modular + Conda-Forge channels
pixi init quickstart \
  -c https://conda.modular.com/max-nightly/ \
  -c conda-forge \
  && cd quickstart

pixi add max
pixi run mojo --version

echo 'def main(): print("Hello, Mojo!")' > hello.mojo
pixi run mojo hello.mojo

# Confirm CUDA
nvidia-smi
nvcc --version
```

This flow uses Pixi's native support for custom Conda channels and Modular's `max-nightly` channel directly — no `pixi login` step required.

## A.2 Lambda Cloud, full workshop / hackathon flow

For a from-scratch instance including launch and lifecycle notes:

```bash
# 1. Launch at https://cloud.lambdalabs.com
#    GPU: A10 (24GB), Quadro RTX 6000, or A100
#    Image: Ubuntu 22.04 + Lambda Stack
#    Storage: attach a persistent filesystem (>= 100GB) if you want data to survive termination

# 2. SSH in
chmod 400 your_key.pem
ssh -i your_key.pem ubuntu@<INSTANCE_PUBLIC_IP>

# 3. System packages + Pixi (same as A.1 Section A/B above)

# 4. Verify GPU + CUDA
nvidia-smi
nvcc --version
```

!!! danger "Ephemeral by default"
    Without an attached filesystem, a Lambda instance's storage is destroyed on termination with no recovery. There is no "Stop" — only **Restart** (reboots, keeps data while the instance still exists) and **Terminate** (deletes the instance *and* its ephemeral storage). Back up before terminating:
    ```bash
    tar -czvf backup.tar.gz ~/quickstart/
    scp -i your_key.pem ubuntu@<INSTANCE_PUBLIC_IP>:~/backup.tar.gz .
    ```
    Keep `mojoproject.toml`, source, and data under version control, and for repeatable setups, turn this script into `setup_mojo.sh` and run it on instance launch.

## A.3 Quick reference: running this book's examples

```bash
# Any .mojo file in this book runs the same way:
pixi run mojo <filename>.mojo

# Build a standalone executable:
pixi run mojo build <filename>.mojo -o <output_name>
./<output_name>
```

See [Getting Started](../getting-started.md) for the expected-output benchmarks across T4 / A10 / RTX 3080 / A100, and the troubleshooting table for common `pixi`/`mojo` errors.
