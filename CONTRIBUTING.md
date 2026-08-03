# Contributing

Changes should preserve the benchmark contract and keep the two-repository
ownership boundary explicit.

Before opening a change:

1. Add or update a characterization test for observable behavior.
2. Run `python -m unittest discover -s tests -v`.
3. Run `python tools/audit_repository.py --models-root ../SoftVTBench-Models`.
4. Document protocol-affecting changes in `docs/protocol.md` and bump the
   release schema/version when old and new results cannot be combined.

Do not add checkpoints, datasets, generated results, private paths or another
copy of a model backend. Avoid a new abstraction until at least two real call
sites share the same stable behavior. Changes to task geometry, physics,
controller thresholds, preprocessing, seeds, metrics or OOD definitions require
explicit before/after evidence and must never be presented as a refactor only.

