# SoftVTBench

SoftVTBench is a reproducible benchmark for vision-only (VO) and
vision-tactile (VT) manipulation of rigid and deformable objects. This
repository owns the simulator, benchmark suites, ID/OOD protocol, rollout
engine, metrics and result receipts. Training code and model checkpoints are
owned by the companion **SoftVTBench-Models** repository.

The public layout intentionally follows one rule: benchmark code never vendors
or reimplements a model backend. See [Architecture](docs/architecture.md) for
the boundary between the two repositories.

## What is included

- Four 10-task suites: `object_rigid`, `spatial_rigid`, `object_soft` and
  `spatial_soft`.
- The formal N=50 ID protocol and nine-condition OOD matrix for the two soft
  suites.
- Deterministic policy sampling, environment restoration and fail-closed
  episode receipts.
- OpenPI, remote-worker and replay benchmark clients.
- The `tac_manip` Isaac Lab extension and its required simulator assets.

Large demonstrations, checkpoints and generated results are not stored in Git.
Their expected layout is documented in [Assets and data](docs/assets-and-data.md).

> **Artifacts are not published yet.** The demonstrations, evaluation USD assets
> and reference checkpoints have no download location as of this release, so the
> repository can be inspected, audited and unit-tested but cannot execute an
> episode. See [Assets and data](docs/assets-and-data.md) for the bundle table
> that will carry the URLs.

To evaluate a policy of your own, see [Custom policies](docs/custom-policy.md).

## Repository map

```text
SoftVTBench/
├── config/                 suites, objects, physics and OOD protocol
├── scripts/                stable shell entry points
├── source/tac_manip/       Isaac Lab extension and simulator assets
├── src/softvtbench/        rollout, metrics, clients and receipts
├── tests/                  dependency-light characterization tests
└── tools/                  repository and release audits
```

## Installation

The formal simulator stack is Linux, Python 3.11, Isaac Sim 5.1 and Isaac Lab
2.3. GPU Taxim additionally needs a `torch-scatter` wheel built for the exact
PyTorch/CUDA pair in that environment.

Simulator assets are stored in Git LFS, so install it once before cloning:

```bash
git lfs install
```

Clone both repositories as siblings, then install the benchmark and extension
inside the Isaac environment:

```bash
git clone <softvtbench-url> SoftVTBench
git clone <softvtbench-models-url> SoftVTBench-Models

python -m pip install -e SoftVTBench
python -m pip install -e SoftVTBench/source/tac_manip
```

Copy the environment templates from both repositories, replace their checkout,
data, asset, checkpoint and result paths, then source them. No source file needs
a machine-specific edit.

```bash
cp SoftVTBench/env.example .env.softvtbench
cp SoftVTBench-Models/env.example .env.softvtbench-models
$EDITOR .env.softvtbench
$EDITOR .env.softvtbench-models
source .env.softvtbench
source .env.softvtbench-models
cd SoftVTBench
python tools/audit_repository.py \
  --models-root "$SOFTVT_MODELS_ROOT"
```

Install the selected model backend in its own environment as described by
SoftVTBench-Models. The evaluator communicates with heavy backends through a
local worker, so JAX and PyTorch stacks do not have to coexist with Isaac.

## Formal evaluation

The formal clean source was derived from release commit
`ed472c477df92ccc456cb6426621f3974221d6ac`. Commit both new repositories and
set `SOFTVT_REQUIRE_CLEAN_RELEASE=1` for release runs; every result records both
Git commits, dirty state, configuration fingerprint and an episode receipt.

ID N=50 (10 tasks x 50 demos = 500 episodes):

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

OOD N=50 (10 tasks x 50 demos x 9 conditions = 4,500 episodes):

```bash
bash scripts/eval_stage.sh \
  object_soft \
  object_soft/pi05_full_vt_c \
  50 \
  "$SOFTVT_RESULT_ROOT/ood_object_soft_pi05_full_vt" \
  9021 \
  "0 1 2 3 4 5 6 7 8 9" \
  0 \
  config/ood/formal_n50/conditions_9.txt
```

Use different output directories for ID and OOD. Run at most one formal Isaac
shard per GPU; concurrent shards previously triggered PhysX CUDA error 700.
The clean protocol sets `total_width_tighten_m=0` for both VO and VT, so older
VO OOD results with a 0.6 mm tightening offset must not be merged with these
results.

## Verification

The fast checks do not start Isaac:

```bash
python -m unittest discover -s tests -v
python tools/audit_repository.py --models-root ../SoftVTBench-Models
```

The audit rejects non-empty duplicate files, private absolute paths (including
bare IP literals used as hosts), generated artifacts, invalid configuration,
missing formal policies and cross-repository ownership violations. See
[Reproducibility](docs/reproducibility.md) before reporting formal numbers and
follow the [release process](docs/release-process.md) before publishing.

## Documentation

| Document | Purpose |
|---|---|
| [Architecture](docs/architecture.md) | the two-repository ownership boundary |
| [Protocol](docs/protocol.md) | suites, action contract, OOD matrix, determinism |
| [Custom policies](docs/custom-policy.md) | evaluating a policy you wrote yourself |
| [Assets and data](docs/assets-and-data.md) | artifact bundles and directory layout |
| [Data format](docs/data-format.md) | HDF5 trajectory schema |
| [Reproducibility](docs/reproducibility.md) | checklist before reporting numbers |
| [Release process](docs/release-process.md) | publishing a version |
| [Audit](docs/audit.md) | source audit and refactoring record |

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing a protocol or formal
configuration. Project-authored code is Apache-2.0; vendored code and assets
retain their original terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
