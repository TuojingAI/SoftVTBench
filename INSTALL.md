# Installation

This repo follows the upstream OpenPI layout and uses `uv`.

## Requirements

- Python 3.11
- CUDA-capable JAX environment for training
- `uv`
- `modelscope` for dataset download
- `ffmpeg` for frame checks and simulator debug video export
- A simulator evaluation host with Isaac Sim and the matching policy client checkout

## Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

`modelscope` is included in the project dependencies. If you run the downloader outside the repo environment, install it manually:

```bash
pip install modelscope
```

For manual Python execution, export:

```bash
export PYTHON=$(pwd)/.venv/bin/python
export PYTHONPATH=$(pwd)/src:$(pwd)/packages/openpi-client/src:${PYTHONPATH:-}
```

## Cache Locations

Use repo-local caches for reproducible runs:

```bash
export HF_LEROBOT_HOME=$(pwd)/.cache/lerobot
export HF_HOME=$(pwd)/.cache/huggingface
export HF_DATASETS_CACHE=$(pwd)/.cache/huggingface/datasets
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi
```

These directories are ignored by Git.

## Base Checkpoints

The training wrappers expect base checkpoint paths:

```bash
export PI0_BASE=/path/to/pi0_base/params
export PI05_BASE=/path/to/pi05_base/params
```

You can also rely on the default OpenPI `gs://openpi-assets/...` paths when your environment can access them.
