<div align="center">

<h1>SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for Physically Constrained Robotic Manipulation of Deformable Objects</h1>

Bowen Jing<sup>1,*</sup>, Mingxin Wang<sup>1,2,*</sup>, Ruiyang Hao<sup>3</sup>, Chenchen Ge<sup>1,4</sup>,
Hanwen Shen<sup>5</sup>, Junjie He<sup>6</sup>, Yang Cui<sup>7</sup>, Yiming Hou<sup>1,4</sup>,
Weitao Zhou<sup>2,8,‡</sup>, Jiawei Wang<sup>8</sup>, Minglei Li<sup>8</sup>,
Dandan Zhang<sup>9</sup>, Ding Zhao<sup>10</sup>, Houde Liu<sup>2</sup>, Xiaofan Li<sup>11</sup>,
Si Liu<sup>12</sup>, Ping Luo<sup>13</sup>, Haibao Yu<sup>1,13,‡</sup>

<sup>1</sup>Tuojing Intelligence &nbsp;·&nbsp; <sup>2</sup>Tsinghua University &nbsp;·&nbsp;
<sup>3</sup>King's College London &nbsp;·&nbsp; <sup>4</sup>Southeast University &nbsp;·&nbsp;
<sup>5</sup>Stevens Institute of Technology &nbsp;·&nbsp;
<sup>6</sup>The Hong Kong University of Science and Technology (GZ) &nbsp;·&nbsp;
<sup>7</sup>The University of Manchester &nbsp;·&nbsp; <sup>8</sup>Simple AI &nbsp;·&nbsp;
<sup>9</sup>Imperial College London &nbsp;·&nbsp; <sup>10</sup>Carnegie Mellon University &nbsp;·&nbsp;
<sup>11</sup>Zhejiang University &nbsp;·&nbsp;
<sup>12</sup>Beihang University &nbsp;·&nbsp; <sup>13</sup>The University of Hong Kong

<sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>‡</sup> Corresponding author

[![SoftVTBench](https://img.shields.io/badge/Arxiv-Paper-red)](https://arxiv.org/abs/2607.04234)&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/Arthur12137/SoftVTBench)&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-ModelScope-blue)](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench)&nbsp;
[![Homepage](https://img.shields.io/badge/Project-Website-cyan)](https://softvtbench.github.io/)&nbsp;

</div>

## News

- **July 5, 2026:** The [SoftVTBench paper](https://arxiv.org/abs/2607.04234) was released on arXiv.
- **July 2026:** The [project website](https://softvtbench.github.io/) and dataset mirrors on [Hugging Face](https://huggingface.co/datasets/Arthur12137/SoftVTBench) and [ModelScope](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench) are online.
- **July 2026:** The initial π<sub>0.5</sub> training and closed-loop evaluation code is released.

## Table of Contents

- [Introduction](#introduction)
- [Benchmark Design](#benchmark-design)
- [Task Suites](#task-suites)
- [Data and Asset Downloads](#data-and-asset-downloads)
- [v0.1 Scope and Release Status](#v01-scope-and-release-status)
- [Getting Started](#getting-started)
- [Benchmark Results](#benchmark-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction

SoftVTBench is a contact-rich, visuo-tactile benchmark for evaluating whether a robot can complete a deformable-object manipulation task **without unsafe physical interaction**. It separates task completion from physical safety, exposing rollouts that reach the goal but drop, slip, or over-compress the object.

<div align="center"><b>Overview of SoftVTBench.</b>
<img src="https://softvtbench.github.io/assets/teaser.png" alt="Figure 1: Overview of SoftVTBench">
<p>The current public release comprises four task suites, 1,628 demonstrations, 33 assets, and diverse tabletop scenes. Its safe-interaction envelope distinguishes grasps that are too loose, physically safe, or excessively deforming.</p>
</div>
<br>

Deformable-object manipulation requires more than reaching a target pose. A policy must regulate contact tightly enough to prevent slippage, while avoiding forces that cause excessive deformation. SoftVTBench makes this distinction explicit with:

- **Four matched task suites** spanning rigid/deformable objects and object/spatial variation.
- **Synchronized multimodal observations**: third-person RGB, wrist RGB, bilateral tactile RGB, tactile marker motion, proprioception, and language instructions.
- **Physics-grounded safety evaluation** using policy-hidden finite-element-method (FEM) states.
- **Two public outcomes**: Goal Success and Safety Success.
- **π<sub>0.5</sub> baselines** for vision-only and visuo-tactile policies under the same task protocol.

## Benchmark Design

SoftVTBench is built in Isaac Sim with FEM-simulated deformable objects and a Franka Panda arm equipped with GelSight Mini tactile sensors. Policy observations are synchronized at 20 Hz; privileged FEM states are used only by the evaluator and are never provided to the policy.

<p align="center">
  <img src="https://softvtbench.github.io/assets/method.png" alt="SoftVTBench benchmark design" width="100%">
</p>

The benchmark reports:

- **Goal Success:** the rollout satisfies the task goal.
- **Safety Success:** a goal-completing rollout also avoids dropping the object and keeps peak object deformation below its calibrated, object-specific threshold.

Safety Success is reported for deformable suites. NoDrop is part of the safety definition but is not reported as a separate headline metric.

## Task Suites

SoftVTBench follows a matched 2 × 2 design:

| Suite | Object type | Variation axis | Purpose |
| --- | --- | --- | --- |
| **Object-Rigid** | Rigid | Object identity | Measures object-centric manipulation competence without a deformation constraint. |
| **Spatial-Rigid** | Rigid | Spatial layout | Measures robustness to changing positions and layouts. |
| **Object-Soft** | Deformable | Object identity and compliance | Requires grasp-and-place completion while avoiding drop and excessive deformation. |
| **Spatial-Soft** | Deformable | Spatial layout | Adds spatial variation and visually matched distractors to safe deformable-object manipulation. |

Videos and qualitative examples for all four suites are available on the [project website](https://softvtbench.github.io/#suites).

## Data and Asset Downloads

> **Repository policy:** this GitHub repository contains source code, configurations, tests, and release metadata. Benchmark datasets, HDF5 files, videos, USD scene assets, deformable-object assets, tactile runtime assets, paper PDFs, and model checkpoints are hosted externally and should not be committed to GitHub.

| Property | Value |
| --- | --- |
| Task suites | 4 |
| Demonstration episodes | 1,628 currently hosted |
| Assets | 33 |
| Recording rate | 20 Hz |
| Core metrics | Goal Success / Safety Success |

The current release contains 500 Object-Soft, 500 Spatial-Soft, 421 Object-Rigid, and 207 Spatial-Rigid demonstrations. Each episode contains synchronized visual, tactile, proprioceptive, action, and task streams; the soft suites additionally contain evaluator-only safety information.

### SoftVTBench Data and Deformable Assets

SoftVTBench benchmark data, evaluation scenes, and deformable-object assets are hosted in the SoftVTBench dataset repository. They are not stored in this GitHub repository.

- [Hugging Face: `Arthur12137/SoftVTBench`](https://huggingface.co/datasets/Arthur12137/SoftVTBench)
- [ModelScope: `Arthur12137/SoftVTBench`](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench)

Both mirrors preserve the same top-level folder names.

| Component | Hosted folder | Required for |
| --- | --- | --- |
| Object-Soft demonstrations | [`object-soft/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/object-soft) | Object-centric soft training and evaluation |
| Spatial-Soft demonstrations | [`spatial-soft/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/spatial-soft) | Spatial soft training and evaluation |
| Object-Rigid demonstrations | [`object-rigid/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/object-rigid) | Object-centric rigid training and evaluation |
| Spatial-Rigid demonstrations | [`spatial-rigid/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/spatial-rigid) | Spatial rigid training and evaluation |
| Evaluation scene and USD assets | [`eval-assets/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/eval-assets) | Closed-loop evaluation; use `eval-assets/USD` as the runtime USD directory |
| Soft-object source assets | [`soft-assets/`](https://huggingface.co/datasets/Arthur12137/SoftVTBench/tree/main/soft-assets) | Optional asset authoring and benchmark extension |
| π<sub>0.5</sub> base checkpoint | [`gs://openpi-assets/checkpoints/pi05_base`](https://github.com/Physical-Intelligence/openpi#model-checkpoints) | Initialization for π<sub>0.5</sub> fine-tuning |

### Hugging Face

Download the complete data and asset bundle:

```bash
pip install -U huggingface_hub
export SOFTVTBENCH_DATA=/path/to/SoftVTBench_data
hf download Arthur12137/SoftVTBench \
  --repo-type dataset \
  --local-dir "$SOFTVTBENCH_DATA"
```

Or download only the components you need:

```bash
# One training/evaluation suite
hf download Arthur12137/SoftVTBench \
  --repo-type dataset \
  --include 'object-soft/*' \
  --local-dir "$SOFTVTBENCH_DATA"

# USD assets required for closed-loop evaluation
hf download Arthur12137/SoftVTBench \
  --repo-type dataset \
  --include 'eval-assets/*' \
  --local-dir "$SOFTVTBENCH_DATA"

# Optional soft-object source assets for asset authoring
hf download Arthur12137/SoftVTBench \
  --repo-type dataset \
  --include 'soft-assets/*' \
  --local-dir "$SOFTVTBENCH_DATA"
```

### ModelScope

For users in mainland China, download the same release from ModelScope:

```bash
pip install -U modelscope
modelscope download --dataset Arthur12137/SoftVTBench \
  --local_dir /path/to/SoftVTBench_data
```

### Tabero Upstream Assets

The simulator implementation in this repository is adapted from [NathanWu7/Tabero](https://github.com/NathanWu7/Tabero). Tabero's original LIBERO, robot, and tactile assets are not redistributed by SoftVTBench. Clone or visit the Tabero repository first and follow its README for the upstream asset layout and download instructions:

```bash
git clone https://github.com/NathanWu7/Tabero.git /path/to/Tabero
```

Tabero's setup currently points users to the following external asset repositories:

- [NathanWu7/Isaaclab_Libero](https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero) for the original LIBERO `assembled_hdf5/`, `USD/`, replay, and video assets.
- [china-sae-robotics/Tactile_Manipulation_Dataset](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) for Franka/GelSight robot USDs, textures, and tactile calibration files.

These Tabero upstream assets are separate from the SoftVTBench data and deformable assets hosted under `Arthur12137/SoftVTBench` above.

### Franka and GelSight Runtime Assets

The tactile simulator's robot USDs, textures, and calibration files are hosted separately by the upstream tactile-simulation project:

```bash
pip install -U huggingface_hub
export SOFTVTBENCH_DATA=/path/to/SoftVTBench_data
hf download china-sae-robotics/Tactile_Manipulation_Dataset \
  --repo-type dataset \
  --local-dir "$SOFTVTBENCH_DATA/tactile-runtime-assets"

ln -sfn "$SOFTVTBENCH_DATA/tactile-runtime-assets/assets/data" \
  SoftVTBench/source/tac_manip/tac_manip/assets/data
```

### Expected Local Layout

Keep data and assets outside the Git repository:

```text
/path/to/SoftVTBench_data/
├── object-soft/
├── spatial-soft/
├── object-rigid/
├── spatial-rigid/
├── eval-assets/
│   └── USD/
├── soft-assets/
└── tactile-runtime-assets/
    └── assets/
        └── data/
```

The public launchers use the following path convention:

```bash
export SOFTVTBENCH_DATA=/path/to/SoftVTBench_data
export RAW_ROOT="$SOFTVTBENCH_DATA/object-soft"       # select one task suite
export SOFTVT_EVAL_USD_DIR="$SOFTVTBENCH_DATA/eval-assets/USD"
export BASE_PARAMS=/path/to/pi05_base/params           # downloaded by OpenPI
```

The four benchmark folders are used for training and reference/evaluation data. `eval-assets/` and the separately downloaded tactile runtime assets are additionally required for closed-loop tactile evaluation, while `soft-assets/` is optional unless creating or extending assets. Dataset and asset components may carry different terms; consult the hosted dataset cards and per-asset notices before redistribution.

## v0.1 Scope and Release Status

The initial code release intentionally keeps the public workflow compact:

| Component | v0.1 scope |
| --- | --- |
| Model | π<sub>0.5</sub> only |
| Policy inputs | Vision-only or vision + tactile |
| Training | Dataset conversion → normalization statistics → LoRA training |
| Evaluation | Closed-loop evaluation on all four suites |
| Outputs | Goal Success for all suites; Safety Success and deformation summaries for soft suites |

Current release status:

| Resource | Status | Location |
| --- | --- | --- |
| Paper | Available | [arXiv:2607.04234](https://arxiv.org/abs/2607.04234) |
| Project website | Available | [softvtbench.github.io](https://softvtbench.github.io/) |
| Benchmark data and asset bundles | Available | [Hugging Face](https://huggingface.co/datasets/Arthur12137/SoftVTBench) · [ModelScope](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench) |
| Franka/GelSight runtime assets | Available upstream | [Tactile Manipulation Dataset](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset) |
| π<sub>0.5</sub> base checkpoint | Available upstream | [OpenPI model checkpoints](https://github.com/Physical-Intelligence/openpi#model-checkpoints) |
| Training and evaluation code | Available | This repository |
| SoftVTBench reference checkpoints | Planned | External release; not stored in GitHub |

## Getting Started

SoftVTBench uses two separate Python environments because the released simulator stack uses Python 3.10, while OpenPI requires Python 3.11. All public environment names use the **SoftVTBench** prefix; the simulator environment is `softvtbench-eval`.

### 1. Clone the Repository

```bash
git clone https://github.com/TuojingAI/SoftVTBench.git
cd SoftVTBench
export SOFTVTBENCH_ROOT="$PWD"
```

### 2. Install the Simulator and Evaluation Environment

The `softvtbench-eval` environment contains Isaac Sim 4.5, Isaac Lab 2.1.1, the SoftVTBench Isaac Lab extension, and the lightweight OpenPI client used by the simulator. The Isaac Lab commit below is the version recorded by the released environment lock.

```bash
conda env create -f environment.yml
conda activate softvtbench-eval
python -m pip install --upgrade pip

# Isaac Sim 4.5 and the remaining pinned simulator dependencies.
python -m pip install -r requirements.txt \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Isaac Lab 2.1.1, pinned to the commit used by the release environment.
export ISAACLAB_ROOT=/path/to/IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
git -C "$ISAACLAB_ROOT" checkout 90b79bb2d44feb8d833f260f2bf37da3487180ba
python -m pip install -e "$ISAACLAB_ROOT/source/isaaclab"
python -m pip install -e "$ISAACLAB_ROOT/source/isaaclab_assets"
python -m pip install -e "$ISAACLAB_ROOT/source/isaaclab_rl"
python -m pip install -e "$ISAACLAB_ROOT/source/isaaclab_tasks"

# SoftVTBench simulator extension and evaluation-side policy client.
python -m pip install -e SoftVTBench/source/tac_manip
python -m pip install -e openpi/upstream/packages/openpi-client
export SOFTVTBENCH_PYTHON="$(command -v python)"
```

Before the first non-interactive Isaac Sim import, review and accept NVIDIA's
Isaac Sim EULA, then set `OMNI_KIT_ACCEPT_EULA=YES`. Closed-loop evaluation also
uses the `ffmpeg` command to encode rollout videos; install it with your system
or Conda package manager and make sure it is available on `PATH`.

`requirements.lock.txt` is the tested pip environment snapshot retained for auditing; use `requirements.txt` for installation. When an exact Linux-64 Conda base is required, replace `conda env create -f environment.yml` with `conda create -n softvtbench-eval --file conda-linux-64.lock`, then continue with the same pip and editable-install steps.

### 3. Install the OpenPI Training and Policy-Server Environment

The `softvtbench-openpi` training environment is stored by uv at `openpi/upstream/.venv`. Its checked-in `uv.lock` is the authoritative dependency lock for π<sub>0.5</sub> training and serving; uv installs Python 3.11 if it is not already available.

```bash
python -m pip install --upgrade uv
cd openpi/upstream
uv python install 3.11
uv sync --frozen --python 3.11
cd "$SOFTVTBENCH_ROOT"

export OPENPI_PYTHON="$SOFTVTBENCH_ROOT/openpi/upstream/.venv/bin/python"
"$OPENPI_PYTHON" -c 'import jax, flax, openpi'
```

Download one benchmark suite, the required assets, and the π<sub>0.5</sub> base checkpoint using the paths in [Data and Asset Downloads](#data-and-asset-downloads). Then run the read-only preflight before training:

```bash
export SOFTVTBENCH_DATA=/path/to/SoftVTBench_data
"$SOFTVTBENCH_PYTHON" tools/doctor.py \
  --mode train \
  --suite object-soft \
  --data-root "$SOFTVTBENCH_DATA/object-soft" \
  --openpi-python "$OPENPI_PYTHON"
```

Before closed-loop soft-suite evaluation, validate both environments, the trained checkpoint, the evaluation USD bundle, and the released safety thresholds:

```bash
"$SOFTVTBENCH_PYTHON" tools/doctor.py \
  --mode eval \
  --suite object-soft \
  --data-root "$SOFTVTBENCH_DATA/object-soft" \
  --eval-assets "$SOFTVTBENCH_DATA/eval-assets" \
  --runtime-assets "$SOFTVTBENCH_DATA/tactile-runtime-assets/assets/data" \
  --checkpoint /path/to/checkpoint/step \
  --thresholds "$SOFTVTBENCH_ROOT/configs/safety_thresholds.json" \
  --softvtbench-python "$SOFTVTBENCH_PYTHON" \
  --openpi-python "$OPENPI_PYTHON"
```

### 4. Train π<sub>0.5</sub>

Run the complete π<sub>0.5</sub> training pipeline:

```bash
export OPENPI_PYTHON=/path/to/openpi-venv/bin/python
export RAW_ROOT=/path/to/SoftVTBench_data/object-soft
export BASE_PARAMS=/path/to/pi05_base/params

SUITE=object-soft MODALITY=tactile PHASE=all \
  bash openpi/scripts/train_softvtbench.sh
```

The released training launchers default to 8 GPUs (`FSDP_DEVICES=8`) with a global batch size of 256. Other resource settings can be supplied through `FSDP_DEVICES` and `BATCH_SIZE`; single-GPU training has not been validated for v0.1.

Set `PHASE=convert`, `PHASE=stats`, `PHASE=train`, or `PHASE=all` to run one training stage or the full pipeline. Supported suites are `object-soft`, `spatial-soft`, `object-rigid`, and `spatial-rigid`; set `MODALITY=vision` for the vision-only baseline.

### 5. Run Closed-Loop Evaluation

Run closed-loop evaluation:

```bash
OPENPI_PYTHON=/path/to/openpi-venv/bin/python \
SOFTVTBENCH_PYTHON=/path/to/softvtbench-eval/bin/python \
COLLECTION_ROOT=/path/to/SoftVTBench_data/object-soft \
SOFTVT_EVAL_USD_DIR=/path/to/SoftVTBench_data/eval-assets/USD \
SAFETY_THRESHOLDS="$PWD/configs/safety_thresholds.json" \
CKPT=/path/to/checkpoint/step \
SUITE=object-soft MODALITY=tactile N=50 \
  bash openpi/scripts/evaluate_softvtbench.sh
```

All four suites default to `N=50` evaluation episodes per task. With the default ten-task subset, a complete suite evaluation therefore contains 500 rollouts. Set both `N=1` and `TASKS_STR=0` explicitly for a single-rollout installation or chain smoke check; `N=1` alone still evaluates one episode for each of the ten default tasks.

## Benchmark Results

We compare a vision-only π<sub>0.5</sub> policy (VO) with a visuo-tactile π<sub>0.5</sub> policy (VT) under matched task and physics conditions.

| Suite | Policy | Goal Success ↑ | Safety Success ↑ |
| --- | --- | ---: | ---: |
| Object-Rigid | VO | **38.8%** | N/A |
| Object-Rigid | VT | 32.4% | N/A |
| Spatial-Rigid | VO | 56.4% | N/A |
| Spatial-Rigid | VT | **63.4%** | N/A |
| Object-Soft | VO | 70.4% | 21.4% |
| Object-Soft | VT | **71.8%** | **35.6%** |
| Spatial-Soft | VO | 74.2% | 32.6% |
| Spatial-Soft | VT | **84.2%** | **44.6%** |

<p align="center">
  <img src="https://softvtbench.github.io/assets/goal_vs_safety.png" alt="Goal Success and Safety Success on SoftVTBench" width="92%">
</p>

Tactile input does not provide a uniform gain on rigid tasks, but it consistently improves safe performance on the deformable suites. On Object-Soft, VT raises Safety Success from 21.4% to 35.6% while Goal Success remains comparable; on Spatial-Soft, it improves both Goal Success and Safety Success. See the [paper](https://arxiv.org/abs/2607.04234) for the full protocol, deformation statistics, and analysis.

## Citation

If you find SoftVTBench useful, please consider citing our paper:

```bibtex
@article{jing2026softvtbench,
  title   = {SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for Physically Constrained Robotic Manipulation of Deformable Objects},
  author  = {Jing, Bowen and Wang, Mingxin and Hao, Ruiyang and Ge, Chenchen and Shen, Hanwen and He, Junjie and Cui, Yang and Hou, Yiming and Zhou, Weitao and Wang, Jiawei and Li, Minglei and Zhang, Dandan and Zhao, Ding and Liu, Houde and Li, Xiaofan and Liu, Si and Luo, Ping and Yu, Haibao},
  journal = {arXiv preprint arXiv:2607.04234},
  year    = {2026}
}
```

For questions or release issues, please open a [GitHub issue](https://github.com/TuojingAI/SoftVTBench/issues).

## Acknowledgements

SoftVTBench builds on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [OpenPI](https://github.com/Physical-Intelligence/openpi). Our simulator implementation was developed with reference to [Tabero](https://github.com/NathanWu7/Tabero). We thank the authors and maintainers of these projects for their excellent work.

Third-party license details are retained in `THIRD_PARTY_NOTICES` and the license files bundled with the corresponding source components.
