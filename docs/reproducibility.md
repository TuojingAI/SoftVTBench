# Reproducibility checklist

Use this checklist before treating an experiment as formal.

1. Checkout committed, clean revisions of SoftVTBench and SoftVTBench-Models.
2. Run both repository audits and all dependency-light tests.
3. Verify the checkpoint and statistics files against the release artifact
   checksums supplied with the model distribution.
4. Source environment files; do not edit YAML or Python paths per machine.
5. Confirm the scene receipt reports the Python, Isaac Sim and Isaac Lab
   versions declared in `config/physics.yaml` and the
   `lower_limit_only` soft-gripper constraint.
6. Set `SOFTVT_REQUIRE_CLEAN_RELEASE=1`.
7. Use a new output directory and one formal Isaac shard per GPU.
8. Run ID and OOD into different directories.
9. Confirm expected counts: 500 ID rows or 4,500 OOD rows per policy/suite.
10. Preserve `fingerprint.txt`, `stage_summary.json`, `results.jsonl` and all
   `episode_receipts/` files together.
11. Never combine legacy VO OOD runs containing the 0.6 mm tightening offset
    with this release.

The source/config fingerprint is independent of filesystem enumeration order.
Receipts include benchmark and model Git revisions, dirty flags, dataset
identity, policy configuration, condition identity and runtime contract checks.

Changing a path alias is allowed only when the underlying artifact checksum is
unchanged. Changing a suite value, policy transform, controller threshold,
condition file, seed scheme, checkpoint or normalization statistics creates a
new experimental protocol and requires a version bump.
