# Evaluation protocol

## Suites

The repository defines four suites with ten tasks each. The formal N=50 OOD
matrix applies to `object_soft` and `spatial_soft`. Rigid suites remain available
for ID experiments and data generation.

Each task uses demos `demo_0` through `demo_49` when `episodes=50` and
`demo_offset=0`. ID and OOD use the same demonstrations and policy sampling seed
for paired comparison.

## Action and control

Policies predict a 7-D absolute target-next action:

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper_position]
```

π0.5 and FastWAM use the `chunked_30x10` protocol: at most 30 inferences with ten
executed control steps per inference. Diffusion Policy uses its native eight
action steps. Continuous-gripper thresholds and FastWAM relative-aperture
calibration are centralized in `config/policy_protocols.yaml`.

The goal success criterion requires eight consecutive successful simulation
steps. Deformation is reported as a distribution (`d_peak` minimum, median,
p95 and maximum); no unfinalized safety threshold is converted into a success
label.

## OOD matrix

The formal condition file contains nine rows:

- lighting: bright mild/strong and dark strong;
- mass: heavy mild/moderate/strong;
- Young's modulus: softer mild/strong and harder strong.

Each episode restores authored physics and lighting before applying its
condition, preventing state leakage or compounding material edits. VO and VT use
`total_width_tighten_m=0` in this clean release.

## Determinism and receipts

Environment seeds are derived from suite, task and demo. Policy inference seeds
exclude the condition label, so ID and OOD receive the same sampling noise.
OpenPI resets JAX's request RNG from a required uint32 seed; DP seeds the first
denoising inference of each episode.

Every result row references a content-hashed episode receipt. Aggregation fails
when a receipt is missing, its hash differs, a contract check failed, or a
`(condition, task, episode)` identity is duplicated.

