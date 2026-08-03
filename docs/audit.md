# Source audit and refactoring record

## Audited baselines

- Anonymous release snapshot: 1,060 files, 27 MiB.
- Formal clean evaluation source: release commit
  `ed472c477df92ccc456cb6426621f3974221d6ac`.
- Simulator extension/assets: the local SoftVTBench source used by the formal
  evaluation stack.

The anonymous snapshot's own manifest and dependency-light tests were checked
before restructuring. Its 49 tests produced 43 passes; six collection errors
were missing optional `torch`/`cv2` dependencies on the audit Mac, not assertion
failures.

## Findings

| Category | Evidence | Resolution |
|---|---|---|
| Exact duplication | 147 non-empty hash groups, 322 files, 3,124,620 redundant bytes | New repositories have zero non-legal exact duplicates internally and across the boundary. |
| OpenPI copied twice | 145 shared paths between evaluation and training; 143 byte-identical, only two differed | One OpenPI tree remains, owned by SoftVTBench-Models; benchmark keeps only its client. |
| Configuration sprawl | 100 evaluator policy entries mixed formal, debug, binary, chunk and stale checkpoint variants | Formal registry reduced to the 16 published N=50 policies; execution profiles live once in the benchmark. |
| Hard-coded locations | 41 files matched private/absolute path patterns, including LIBERO JSON and checkpoint/run paths | Formal source/config/scripts use environment variables and stable artifact aliases. |
| Dead or misleading files | Old examples, scratch configs, generic deployment scripts, kitchen/PushT/RoboMimic workflows, stale nested repository metadata | Confirmed non-formal files were removed from the new staging repositories. |
| Mixed responsibilities | Model servers/adapters and a full OpenPI checkout were embedded beside evaluation code | Backends and workers moved to SoftVTBench-Models; tasks, rollout, metrics and receipts remain in SoftVTBench. |
| Unsafe cleanup | Queue/stage tooling used process-name cleanup and a checkpoint view could be overwritten | Each stage records and stops only its own PIDs; an existing checkpoint view is refused. |
| Incomplete formal release checks | Remote clean tree lacked a reliable release descriptor/formal condition gate in the inspected state | `release.json`, policy/config fingerprints, two-repository clean checks and fail-closed receipts are required. |
| Missing FastWAM configs | The formal launcher derived four soft data config names that were absent | All eight suite x modality config names now exist; invalid interpolation lists were corrected without changing values. |
| Template residue | Isaac extension metadata/UI example and unused UR10e config remained from a template | Metadata now describes SoftVTBench; unused template UI/robot code was removed. |

## Refactoring rules used

1. Preserve task values, action semantics, controller thresholds, seed behavior,
   checkpoint content and metric definitions.
2. Allow repository layout, environment-variable names and stable artifact
   aliases to change.
3. Characterize behavior before deduplicating shared math or preprocessing.
4. Keep vendored backend licenses and source headers.
5. Prefer deletion over an abstraction for one unused call path.
6. Keep the receipt and formal runner conservative even when they are long;
   splitting sealed contract logic without behavioral coverage would add risk.

The resulting audit command performs more than 120 structural, configuration
and contract checks across both repositories, including
AST/JSON/TOML/YAML/shell parsing, formal matrix validation, private-path
detection and content-hash deduplication.
