# Release process

1. Confirm third-party source and binary-asset redistribution terms.
2. Run dependency-light tests in both repositories.
3. Regenerate both checksum manifests:

   ```bash
   python tools/update_manifests.py --models-root ../SoftVTBench-Models
   ```

4. Run the cross-repository audit and review every change to formal config.
5. Commit each repository, rerun the audit from clean checkouts and tag the same
   semantic version in both repositories.
6. Publish datasets/checkpoints to both the Hugging Face and ModelScope mirrors
   with their own SHA-256 manifest, then fill the repository IDs in
   [assets-and-data.md](assets-and-data.md) and delete its "not yet published"
   notice. **Until this step lands, an external user cannot run a single
   episode.** Two requirements:
   - Upload byte-identical archives to both mirrors, so one `SHA256SUMS`
     validates either download.
   - While packaging the demonstrations, rewrite the HDF5 `task` attribute to
     `SoftVT-Soft-Collection-v0`. The recorded files still carry the pre-release
     collection identifier, and [data-format.md](data-format.md) documents the
     new value.
7. Execute one ID smoke episode and one OOD smoke episode before launching the
   full N=50 matrix.
8. Verify the published clone from a clean machine: `git lfs install`, clone both
   repositories, download one bundle, check its SHA-256 and replay one episode.

Do not hand-edit `MANIFEST.sha256`. The manifest excludes itself, Git metadata
and ignored local build/cache directories. A release is invalid if a declared
path is missing, an undeclared path is present or a digest differs.

