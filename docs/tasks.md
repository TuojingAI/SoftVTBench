# SoftTacWorld Tasks

Task subset JSON files are provided in two locations:

```text
configs/tasks/
examples/tabero/config/
```

The `examples/tabero/config/` path is kept for compatibility with the legacy converters. `configs/tasks/` is the public-facing configuration path.

## Reference Training Subset

The full object+spatial reference subset is:

```text
configs/tasks/softtacworld_object_spatial_success_only_20260614.json
```

It covers:

```text
libero_object task IDs: 0-9
libero_spatial task IDs: 0-9
```

## Quick-Eval Subset

The older quick-eval subset is:

```text
configs/tasks/softtacworld_tasks.json
```

It covers:

```text
libero_object task IDs: 0, 1, 2, 3, 5, 6, 7, 8, 9
```

Use the full object+spatial file for systematic evaluation of checkpoints trained on object+spatial data.
