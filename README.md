<div align="center">
<h1>SoftVTBench: A Deformation-Aware Visuo-Tactile Dataset and Benchmark for Deformable-Object Manipulation</h1>

[Bowen Jing](https://arthur12137.com/)<sup>1,\*</sup>, Mingxin Wang<sup>1,2,\*</sup>, [Ruiyang Hao](https://ry-hao.top/)<sup>3</sup>, Chenchen Ge<sup>1,4</sup>, Hanwen Shen<sup>5</sup>, Junjie He<sup>6</sup>, Yang Cui<sup>7</sup>, Yiming Hou<sup>1,4</sup>, <br> Weitao Zhou<sup>2,8,‡</sup>, Jiawei Wang<sup>8</sup>, Minglei Li<sup>8</sup>, Dandan Zhang<sup>9</sup>, Ding Zhao<sup>10</sup>, Houde Liu<sup>2</sup>, Xiaofan Li<sup>11</sup>, Si Liu<sup>12</sup>, Ping Luo<sup>13</sup>, [Haibao Yu](https://scholar.google.com/citations?user=JW4F5HoAAAAJ)<sup>1,13,‡</sup>

<sup>1</sup> Tuojing Intelligence, <sup>2</sup> Tsinghua University, <sup>3</sup> King's College London, <sup>4</sup> Southeast University, <br>
<sup>5</sup> Stevens Institute of Technology, <sup>6</sup> The Hong Kong University of Science and Technology (Guangzhou), <br>
<sup>7</sup> University of Manchester, <sup>8</sup> Simple AI, <sup>9</sup> Imperial College London, <sup>10</sup> Carnegie Mellon University, <br>
<sup>11</sup> Zhejiang University, <sup>12</sup> Beihang University, <sup>13</sup> The University of Hong Kong

<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>‡</sup> Corresponding author

<strong>ECCV 2026 Workshop Oral</strong>

[![arXiv](https://img.shields.io/badge/arXiv-2607.04234-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.04234)&nbsp;
[![Project Page](https://img.shields.io/badge/Project-Website-00b3b3?logo=githubpages&logoColor=white)](https://softvtbench.github.io/)&nbsp;
[![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Arthur12137/SoftVTBench)&nbsp;
[![ModelScope](https://img.shields.io/badge/Dataset-ModelScope-624aff?logo=alibabacloud&logoColor=white)](https://modelscope.cn/datasets/Arthur12137/SoftVTBench)&nbsp;
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/TuojingAI/SoftVTBench)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-4c9a2a?logo=apache&logoColor=white)](LICENSE)&nbsp;
[![Stars](https://img.shields.io/github/stars/TuojingAI/SoftVTBench?style=social)](https://github.com/TuojingAI/SoftVTBench)

</div>

## News

- **`Aug. 2026`:** SoftVTBench is accepted to an **ECCV 2026 Workshop** and will be presented as an **oral**.
- **`Aug. 2026`:** Dataset expanded to **4,000 demonstrations** over 40 tasks and 50+ assets, and the evaluation stack is re-released with deterministic replay and fail-closed episode receipts.
- **`Jul. 5th, 2026`:** We released our paper on [arXiv](https://arxiv.org/abs/2607.04234).
- **`Jul. 2026`:** The [project website](https://softvtbench.github.io/) and the dataset mirrors on [Hugging Face](https://huggingface.co/datasets/Arthur12137/SoftVTBench) and [ModelScope](https://modelscope.cn/datasets/Arthur12137/SoftVTBench) went online.
- **`Jul. 2026`:** We released the initial π<sub>0.5</sub> training and closed-loop evaluation code. ☕️

## Table of Contents

- [Introduction](#introduction)
- [Task Suites](#task-suites)
- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [Running the Benchmark](#running-the-benchmark)
- [Repository Structure](#repository-structure)
- [Benchmark Results](#benchmark-results)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction

Physical interaction quality is central to deformable-object manipulation, yet most benchmarks evaluate task success alone. A policy may complete the task while allowing the object to slip — or by crushing it. Task success sees neither. We introduce **SoftVTBench**, a visuo-tactile dataset and benchmark for physical-interaction-aware deformable-object manipulation. It contains 4,000 expert demonstrations and more than 50 assets, including volumetric deformable objects and visually matched rigid twins. At 20 Hz, each episode synchronizes multi-view RGB, dual-finger tactile RGB and marker motion, proprioception, language, and both binary and continuous gripper actions, alongside evaluator-only finite-element (FEM) states the policy never sees. On top of it we define the **Deformation-aware Success Rate (DSR)**, which credits a rollout only when the task is completed *and* peak normalized deformation stays within a per-object tolerance calibrated before any policy is trained.

<div align="center"><b>Overview of SoftVTBench.</b>
<img src="https://softvtbench.github.io/assets/fig1_overview.png" alt="Overview of SoftVTBench" />
Between a grasp too loose to hold and one tight enough to crush lies a narrow safe window, and only touch observes it from the inside.
</div>
<br>

## Task Suites

We propose a matched 2×2 design over object type (deformable vs. rigid twin) and variation axis (object identity vs. spatial layout), with one shared pick-and-place skill throughout.

| Suite | Scene / Goal | Object | #Tasks | #Demos | ID Eval | OOD | Size |
|---|---|---|:--:|:--:|:--:|:--:|:--:|
| `object-soft` | floor → basket | FEM soft body | 10 | 1,000 | 500 | 9 | 2.0 GB |
| `spatial-soft` | kitchen table → plate | FEM soft body | 10 | 1,000 | 500 | 9 | 2.5 GB |
| `object-rigid` | floor → basket | rigid twin | 10 | 1,000 | 500 | — | 975 MB |
| `spatial-rigid` | kitchen table → plate | rigid twin | 10 | 1,000 | 500 | — | 1.8 GB |
| **Total** | — | — | **40** | **4,000** | **2,000** | — | **7.3 GB** |

<div align="center"><b>Pipeline of SoftVTBench Construction and Evaluation.</b>
<img src="https://softvtbench.github.io/assets/fig2_pipeline.png" alt="SoftVTBench construction and evaluation pipeline" />
</div>
<br>

## Installation

### Requirements

| | Version | Notes |
|---|---|---|
| OS | Linux | Isaac Sim is not supported on macOS or Windows WSL |
| Python | 3.10 | pinned by the Isaac Sim 4.5 runtime |
| Isaac Sim | `4.5.0.0` | must match `config/physics.yaml` |
| Isaac Lab | `0.41.3` | must match `config/physics.yaml` |
| GPU | NVIDIA, ≥ 16 GB VRAM | PhysX 5 GPU FEM is required for the soft suites |
| Disk | ≥ 20 GB | 7.3 GB dataset + assets + rollout videos |

> Formal runs **fail closed** when the installed simulator packages disagree with `config/physics.yaml`. This is intentional: a number produced on a different physics build is not comparable.

### Step 1 — Install Isaac Sim and Isaac Lab

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and pin the two versions above. Verify before continuing:

```bash
python -c "import isaacsim, isaaclab; print(isaacsim.__version__, isaaclab.__version__)"
```

All later steps run **inside this environment**.

### Step 2 — Clone the repository

Simulator assets (USD meshes, GelSight calibration tables) are tracked with Git LFS, so install it before cloning — otherwise the assets arrive as text stubs and Isaac will fail to load the scene:

```bash
git lfs install
git clone https://github.com/TuojingAI/SoftVTBench.git
cd SoftVTBench
```

Confirm the assets were materialised (the file should be ~52 MB, not ~130 bytes):

```bash
ls -lh source/tac_manip/tac_manip/assets/data/Sensors/GelSight_Mini/Gelpad_extremely_high_res.usd
```

### Step 3 — Install the two packages

SoftVTBench ships two pip packages with different roles:

```bash
python -m pip install -e .                  # softvtbench — rollout engine, metrics, receipts
python -m pip install -e source/tac_manip   # tac_manip  — Isaac Lab extension: envs, sensors, assets
```

`tac_manip` must live under `source/` because Isaac Lab discovers extensions by that layout; `softvtbench` uses a standard `src/` layout and never imports Isaac at module level, so metrics and receipts can be exercised without a GPU.

### Step 4 — Install the tactile backend

GPU Taxim needs a `torch-scatter` wheel built for the exact PyTorch/CUDA pair in the Isaac environment:

```bash
pip install torch-scatter -f "https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html"
```

### Step 5 — Verify

```bash
python -c "import softvtbench, tac_manip; print('ok')"
python tools/audit_repository.py
```

The audit checks configuration validity, asset manifests, and that no private absolute path leaked into the release. It prints `OK: ...; N checks` on success.

<details>
<summary><b>Troubleshooting</b></summary>

| Symptom | Cause and fix |
|---|---|
| `USD file ... could not be opened` | Git LFS was not installed before cloning. Run `git lfs install && git lfs pull`. |
| Formal run aborts with a version mismatch | The installed `isaacsim`/`isaaclab` differ from `config/physics.yaml`. Reinstall the pinned versions rather than editing the config. |
| `PhysX CUDA error 700` | More than one Isaac shard per GPU. Run a single shard per device. |
| `ModuleNotFoundError: torch_scatter` | The wheel does not match the environment's torch/CUDA build; reinstall with the index URL in Step 4. |
| Deformable objects fall through the floor | The GPU does not support PhysX 5 GPU FEM, or the run was launched on CPU physics. |

</details>

## Dataset Download

The four subsets total about 7.3 GB and are mirrored byte-identically on Hugging Face and ModelScope; use whichever is faster from your network.

```bash
export SOFTVT_DATA_ROOT=/path/to/softvtbench/data
```

```bash
# Option A — Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Arthur12137/SoftVTBench \
  --repo-type dataset --local-dir "$SOFTVT_DATA_ROOT"
```

```bash
# Option B — ModelScope
pip install -U modelscope
modelscope download --dataset Arthur12137/SoftVTBench \
  --local_dir "$SOFTVT_DATA_ROOT"
```

To pull a single suite instead of all four:

```bash
huggingface-cli download Arthur12137/SoftVTBench \
  --repo-type dataset --include "object-soft/*" --local-dir "$SOFTVT_DATA_ROOT"
```

Each subset is laid out as follows. `manifest.jsonl` carries one line per demo with its task id, language instruction, sample count and file paths.

```text
<subset>/
├── manifest.jsonl
├── assemble_summary.json
└── libero_{spatial,object}/
    └── libero_{spatial,object}_task{0..9}/
        ├── replayed_demos/<task>_<language>_replayed_demo.hdf5
        └── video_datasets/<task>/
            ├── videos/demo_<i>_{agentview,eye_in_hand}_rgb.mp4
            └── tactile_outputs/demo_<i>_gsmini_{left,right}_markers_rgb.mp4
```

Evaluation USD assets are published separately as `Arthur12137/SoftVTBench-archive` on both hubs. See [Assets and data](docs/assets-and-data.md) for the bundle table and checksum verification, and [Data format](docs/data-format.md) for the full HDF5 schema.

### Configure the storage roots

No source file embeds a machine-specific path — every root comes from the environment:

```bash
cp env.example .env.softvtbench
$EDITOR .env.softvtbench
source .env.softvtbench
```

| Variable | Contents |
|---|---|
| `SOFTVTBENCH_ROOT` | this checkout |
| `SOFTVT_DATA_ROOT` | recorded HDF5 demonstrations, grouped by suite and task |
| `SOFTVT_ASSET_ROOT` | evaluation USD assets |
| `SOFTVT_RIGID_STAGING_ROOT` | rigid object staging assets |
| `SOFTVT_SPATIAL_SOFT_STAGING_ROOT` | spatial-soft staging assets |
| `SOFTVT_RESULT_ROOT` | where results, receipts and logs are written |

The evaluator resolves each demo from the selected suite's `data_dir` and `data_subdir` fields in `config/suites/*.yaml`, so the directory names under `SOFTVT_DATA_ROOT` must match those fields.

## Running the Benchmark

### In-distribution (500 episodes)

10 tasks × 50 demonstrations:

```bash
bash scripts/eval_stage.sh \
  object_soft \
  object_soft/pi05_full_vt_c \
  50 \
  "$SOFTVT_RESULT_ROOT/id_object_soft_pi05_full_vt" \
  9021 \
  "0 1 2 3 4 5 6 7 8 9" \
  0
```

Positional arguments: suite · policy · demos per task · output directory · seed · task ids · GPU id.

### Out-of-distribution (4,500 episodes)

Append the condition file for the nine-condition matrix — three levels each of lighting, mass and Young's modulus, one factor at a time — and write to a **separate** output directory:

```bash
bash scripts/eval_stage.sh \
  object_soft object_soft/pi05_full_vt_c 50 \
  "$SOFTVT_RESULT_ROOT/ood_object_soft_pi05_full_vt" \
  9021 "0 1 2 3 4 5 6 7 8 9" 0 \
  config/ood/formal_n50/conditions_9.txt
```

Each OOD episode reuses the task, initial state and seed of its in-distribution reference, so the two directories form a paired comparison.

### Output

Every run writes `results.jsonl` (one row per episode), `summary.json`, and a per-episode receipt recording the Git commit, dirty state, configuration fingerprint, resolved physics parameters and the applied OOD condition. A run whose receipt fails its contract aborts rather than producing a number.

Run **at most one formal Isaac shard per GPU** — concurrent shards trigger `PhysX CUDA error 700`.

To score a policy of your own, see [Custom policies](docs/custom-policy.md); read the [Reproducibility checklist](docs/reproducibility.md) before reporting numbers.

## Repository Structure

```text
SoftVTBench/
├── config/            suite definitions, object cards, physics pins, OOD protocol
├── docs/              protocol, data format, reproducibility, custom policies
├── scripts/           shell entry points for evaluation
├── source/tac_manip/  Isaac Lab extension — environments, tactile sensors, USD assets
├── src/softvtbench/   rollout engine, metrics, policy clients, receipts
└── tools/             repository and release audits
```

## Benchmark Results

Diffusion Policy, π<sub>0.5</sub>, and FastWAM under paired vision-only (VO) and visuo-tactile (VT) inputs, on identical episodes and seeds. Main results are shown in the table below:

| Models | Input | Object-Soft TSR | Object-Soft DSR | Spatial-Soft TSR | Spatial-Soft DSR |
|---|---|:--:|:--:|:--:|:--:|
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | VO | 37.4 | 33.6 | 15.6 | 13.4 |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | VT | 40.0 | 30.4 | 33.0 | 25.0 |
| [π<sub>0.5</sub>](https://github.com/Physical-Intelligence/openpi) | VO | 41.6 | 38.4 | 26.0 | 22.6 |
| [π<sub>0.5</sub>](https://github.com/Physical-Intelligence/openpi) | VT | 41.4 | 35.0 | 27.6 | 22.0 |
| FastWAM | VO | **62.0** | **58.0** | 37.0 | 36.6 |
| FastWAM | VT | 57.6 | 54.4 | **56.4** | **56.0** |

**DSR falls below TSR in all twelve configurations**, accounting for 0.7–24% of each configuration's successes — an exact count of the rollouts that reached the target by mishandling the object. It also flips rankings: for Diffusion Policy, TSR puts VT above VO on Object-Soft (40.0 vs. 37.4), while DSR reverses it (30.4 vs. 33.6). Under distribution shift, visuo-tactile variants win all six task-success comparisons and five of six on DSR, whereas in distribution the same comparison is split. Making touch available is not the same as using it.

More discussions and analysis are provided in [paper](https://arxiv.org/abs/2607.04234).

## Contact

If you have any questions, please open an [issue](https://github.com/TuojingAI/SoftVTBench/issues) or contact [Bowen Jing](https://arthur12137.com/).

## Acknowledgement

This work is partly built upon [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [TacEx](https://github.com/TimSchneider42/tacex), [Taxim](https://github.com/Robo-Touch/Taxim), [FOTS](https://github.com/Rancho-zhao/FOTS), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [openpi](https://github.com/Physical-Intelligence/openpi). Thanks them for their great works!

## Citation

If you find SoftVTBench is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{jing2026softvtbench,
  title         = {SoftVTBench: A Deformation-Aware Visuo-Tactile Dataset and
                   Benchmark for Deformable-Object Manipulation},
  author        = {Jing, Bowen and Wang, Mingxin and Hao, Ruiyang and Ge, Chenchen and
                   Shen, Hanwen and He, Junjie and Cui, Yang and Hou, Yiming and
                   Zhou, Weitao and Wang, Jiawei and Li, Minglei and Zhang, Dandan and
                   Zhao, Ding and Liu, Houde and Li, Xiaofan and Liu, Si and
                   Luo, Ping and Yu, Haibao},
  year          = {2026},
  eprint        = {2607.04234},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2607.04234}
}
```
