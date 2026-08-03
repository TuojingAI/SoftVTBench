# Evaluating your own policy

SoftVTBench evaluates any policy that implements four operations. Nothing in the
rollout, metric or receipt layer is specific to the four reference families, and
adding a policy never requires editing `rollout.py`, `runner.py` or a suite
config.

There are two integration paths. Pick the in-process one when your model can be
imported inside the Isaac Sim environment, and the worker one when it cannot.

## The policy contract

```python
class Policy(abc.ABC):
    modality: str = "vo"          # "vo" | "vt"

    def reset(self, *, episode_seed: int | None = None) -> None: ...
    def observe(self, env_obs: dict) -> None: ...
    def predict(self) -> np.ndarray: ...       # (k, 7)
```

- `reset` runs once at the start of every episode and must clear **all** internal
  state: temporal buffers, quaternion branch tracking, sampler RNG. `episode_seed`
  is derived from suite, task and demo (see [protocol.md](protocol.md)); seed your
  sampler from it so a rerun reproduces the same rollout.
- `observe` is called once per environment step. The policy owns its own history;
  the benchmark does not stack frames for you.
- `predict` returns a `(k, 7)` chunk of **absolute target-next** actions,
  `[x, y, z, ax, ay, az, gripper]`, with the rotation as an axis-angle. `k` is the
  number of control steps executed per inference and must match the execution
  profile the policy is registered under: `chunked_30x10` requires exactly 10,
  `native_env_steps` uses the policy's own chunk length.

`set_language(text)` is optional. Policies without a language input may omit it;
the worker returns `"ok"` for compatibility.

Gripper decoding (Schmitt thresholds, continuous vs. relative aperture, hold
steps) lives outside the policy in `gripper_execution`, driven by
`config/policy_protocols.yaml`. Do not implement it inside `predict`.

## Path A: in-process policy

Subclass `Policy` and register it. Registration is by name and must be unique.

```python
# my_policy.py
import numpy as np
from softvtbench.evaluation.policies.base import Policy, register
from softvtbench.evaluation.preprocessing.obs_build import build_proprio
from softvtbench.evaluation.preprocessing.rotation import QuatBranchTracker


@register("my_backend")
class MyPolicy(Policy):
    def __init__(self, *, ckpt_path: str, modality: str = "vo", **_):
        self.modality = modality
        self.model = load_my_model(ckpt_path)
        self._tracker = QuatBranchTracker()
        self._history = []

    def reset(self, *, episode_seed: int | None = None) -> None:
        self._history.clear()
        self._tracker.reset()
        self.model.seed(episode_seed)

    def observe(self, env_obs: dict) -> None:
        self._history.append(build_proprio(env_obs, self._tracker))

    def predict(self) -> np.ndarray:
        return self.model.act(self._history)[-10:]        # (10, 7)
```

Reuse `build_proprio`, `build_marker` and `QuatBranchTracker` from
`softvtbench.evaluation.preprocessing`. They implement the exact proprioception
layout, tactile marker packing and quaternion branch continuity used to record
the demonstrations; reimplementing them is the most common source of a silent
train/eval mismatch.

Import the module before the runner builds a policy so the decorator has run, then
declare it in a registry entry:

```yaml
- id: object_soft/my_policy_vt
  suite: object_soft
  backend: my_backend
  modality: vt
  execution_profile: continuous_chunked
  ckpt_path: ${SOFTVT_CHECKPOINT_ROOT}/object_soft/my_model/latest.pt
```

Fields other than `id`, `suite`, `backend`, `modality` and `execution_profile` are
passed to your constructor as keyword arguments.

## Path B: out-of-process worker

Isaac Sim pins its own CUDA/Python stack. When your model cannot coexist with it,
run the adapter in its own environment behind the bundled worker and let the
benchmark talk to it over loopback HTTP:

```bash
python -m softvtbench_models.worker \
  --backend my_backend --port 9103 \
  --kwargs '{"ckpt_path": "/checkpoints/my_model/latest.pt", "modality": "vt"}'
```

The benchmark side then uses the `remote` client, which forwards `ping`, `reset`,
`observe`, `predict` and `set_language` unchanged. The wire format is pickle over
HTTP on `127.0.0.1`.

This boundary exists to isolate dependency stacks. It is **not** a security
boundary: it accepts pickled payloads and must never be bound to a non-loopback
interface. See [SECURITY.md](../SECURITY.md).

## Before you report numbers

A run is only comparable to the published results if it satisfies the protocol,
not merely if it completes:

1. Use the unmodified suite, object, physics and OOD configs. Any change to task
   geometry, thresholds, seeds or conditions is a different benchmark.
2. Run the full matrix: 10 tasks x 50 demos = 500 ID episodes per suite/policy,
   and x 9 conditions = 4,500 OOD episodes.
3. Keep every `episode_receipts/` file. Aggregation fails closed when a receipt is
   missing, its hash differs, a contract check failed, or a
   `(condition, task, episode)` identity repeats.
4. Report the deformation distribution (`d_peak` min/median/p95/max) alongside
   success. Success alone hides the safety-goal trade-off the benchmark exists to
   measure.

Work through [reproducibility.md](reproducibility.md) before treating a run as
formal.
