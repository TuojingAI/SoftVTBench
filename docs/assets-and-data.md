# Assets and data

## Obtaining the artifacts

> **Not yet published.** The demonstrations, evaluation USD assets and reference
> checkpoints are not in Git and, as of this release, are not yet downloadable.
> Until the bundles below are uploaded and the URLs filled in, this repository
> can be inspected, audited and unit-tested, but **no episode can be run**.
>
> Maintainers: publishing these bundles is step 6 of
> [release-process.md](release-process.md). Fill in the repository IDs below,
> record the SHA-256 of each archive, and delete this notice.

Each bundle is mirrored on Hugging Face and ModelScope. The two mirrors carry
byte-identical archives; use whichever is faster from your network.

| Bundle | Contents | Hugging Face | ModelScope |
|---|---|---|---|
| demonstrations | 500 ID demonstrations per soft suite, HDF5 | `TBD` | `TBD` |
| assets | evaluation USD assets and staging objects | `TBD` | `TBD` |
| checkpoints | the 16 formal policy checkpoints and normalization statistics | `TBD` | `TBD` |

```bash
# Hugging Face
huggingface-cli download <hf-repo-id> --repo-type dataset --local-dir "$SOFTVT_DATA_ROOT"

# ModelScope
modelscope download --dataset <modelscope-repo-id> --local_dir "$SOFTVT_DATA_ROOT"
```

Every bundle ships its own `SHA256SUMS`. Verify before use:

```bash
sha256sum -c SHA256SUMS
```

A checksum mismatch invalidates any number produced from that artifact. Do not
proceed past a failed check. Because the two mirrors are byte-identical, the same
`SHA256SUMS` must validate a download from either one; a mirror-dependent digest
means one of them is stale.

Checkpoints are covered by the terms of their base models, not by this
repository's license. Read `THIRD_PARTY_NOTICES.md` in the companion
SoftVTBench-Models checkout before redistributing weights.

## Configuring the roots

Source code does not embed private storage paths. Configure these roots:

| Variable | Contents |
|---|---|
| `SOFTVT_DATA_ROOT` | recorded HDF5 demonstrations grouped by suite/task |
| `SOFTVT_ASSET_ROOT` | LIBERO/SoftVTBench USD assets used during evaluation |
| `SOFTVT_RIGID_STAGING_ROOT` | rigid object staging assets |
| `SOFTVT_SPATIAL_SOFT_STAGING_ROOT` | spatial-soft staging assets |
| `SOFTVT_CHECKPOINT_ROOT` | model artifacts described by the companion registry |

The evaluator resolves a demo from the selected suite's `data_dir` and
`data_subdir` fields. With the default suite configs, the demonstration layout
is:

```text
$SOFTVT_DATA_ROOT/
├── object_soft/libero_object_task0/.../demo_0
├── object_soft/libero_object_task9/.../demo_49
├── spatial_soft/libero_spatial_task0/.../demo_0
└── spatial_soft/libero_spatial_task9/.../demo_49
```

See [data-format.md](data-format.md) for the HDF5 trajectory schema. Published
artifact bundles should include a SHA-256 manifest. Do not silently substitute
or rename trained artifacts without recording an alias and verifying its hash.

