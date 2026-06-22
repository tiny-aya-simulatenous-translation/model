# Diagnosis -> Recovery table

The encoded playbook used by `tpu-diagnoser` to classify a failure
and propose patches. Source spec: `../SPEC.md`.

Match priority: top-to-bottom. First regex hit wins.

| # | Symptom (regex on tmux log) | Root cause | Patch | Tier | Source |
|---|---|---|---|---|---|
| 1 | `Failed to deserialize executable: UNIMPLEMENTED` | XLA persistent cache broken on TPU v4 + torch_xla 2.9 | Remove `XLA_PERSISTENT_CACHE_PATH` env from startup_script.sh + _remote_redeploy.sh | T2 | pytorch/xla #8930, #9094 |
| 2 | `ValueError.*Layer \d+ has mismatched keys` | scan_layers `_ensure_same_structure` rejects LoRA[0:33]+FullFT[34:35] split | `use_scan_layers: false` in YAML config | T2 | torch_xla scan_layers.py source |
| 3 | `AssertionError.*FakeTensor.*aten\.index_select` | `is_layer_pure=True` + position-embedding gather | Drop `is_layer_pure=True` from scan_utils.py call site | T2 | PyTorch #105485 |
| 4 | `TypeError.*unexpected keyword.*attention_mask` | scan kwargs binding | Already fixed by KwargBoundLayer (no-op) | -- | local |
| 5 | `RESOURCE_EXHAUSTED` OR `OOM` OR `exit code 137` | per-chip HBM exceeded | Halve `batch_size` OR `depth_chunk_size` in YAML | T2 | std PT/XLA |
| 6 | TPU duty=0% AND HBM>50% AND wall>30 min AND no `step=` | Compile hung / executor stuck | Kill, dump `met.metrics_report()`, add `python -u`, ensure `XLA_PERSISTENT_CACHE_PATH` is unset | T2 | NCCL #112518 |
| 7 | `gcloud ssh.*Connection refused` | VM SSH unreachable | ESCALATE -- do not auto-recreate QR | T3 | GCP Spot docs |
| 8 | `kernel panic` OR `Bus error` in tmux log | VM-level corruption | ESCALATE -- do not auto-recreate QR | T3 | GCP Spot docs |
| 9 | N worker PIDs all unreachable for 3+ consecutive polls (`N=1` on v6e-8, `N=4` on legacy v4-32, `N=8` on v6e-64) | TPU VM(s) down | ESCALATE -- do not auto-recreate QR | T3 | GCP Spot docs |
| 10 | Same `classification` as previous iteration | Doom loop circuit breaker | ESCALATE | T4 | LangChain self-heal April 2026 |
| 11 | User selects "Abort+Diag" or "Pause" at check-in | User decision | ESCALATE | T4 | -- |
| 12 | `compilation_cause_count` rising AND no error AND elapsed < 30 min | Normal compile (sharding distribution) | Recommend "continue" at check-in | T0 | our prior runs + #8612 |
| 13 | `compilation_cause_count` rising AND no error AND elapsed > 60 min | Possibly dynamic shapes (recompiles per step) | Recommend "abort+diag" at check-in; offer `XLA_IR_DEBUG=1` next iter | T2 | PT/XLA debug docs |
| 14 | `could not read Username for 'https://github.com'` OR `Repository not found` | Fresh TPU VM lacks private GitHub credentials | Relaunch with `REPO_TARBALL_GS_URI=gs://...` so startup fetches code from GCS | T2 | opt-4-depth32 startup |
| 15 | `ImportError.*libpython3\.12\.so\.1\.0` | torch_xla cannot find uv-managed CPython shared library | Export uv CPython lib dir in `LD_LIBRARY_PATH` via `_remote_redeploy.sh` / startup fallback | T2 | opt-4-depth32 redeploy |

## Output schema (`tpu-diagnoser` returns this)

```json
{
  "classification": "xla-cache | scan-structure | fakeTensor | oom | compile-stall | git-auth | libpython | t3-corruption | repeat | unknown",
  "matched_signature": "exact regex/string that matched",
  "matched_table_row": 1,
  "recommended_patches": [
    {"file": "path/to/file", "kind": "remove-env|drop-flag|edit-yaml|halve-batch", "details": "..."}
  ],
  "tier": "T0|T1|T2|T3|T4",
  "confidence": 0.0,
  "should_retry": true,
  "reason": "human-readable summary"
}
```

## How to extend

1. Add a row to the table above with: regex, root cause, patch, tier, source.
2. Update `tpu-diagnoser.md` to recognize the new signature.
3. If the patch is structural, add a corresponding entry to `tpu-orchestrate/SKILL.md`.
4. Run a smoke test that reproduces the symptom + verifies the patch.
