# Check-in protocol

At each scheduled check-in (T+15/30/45/60/90 min wall), the orchestrator
MUST stop the loop and present a structured snapshot via `AskUser` with
exactly 4 options. The user's choice is the only thing that resumes
the loop.

## Snapshot format (each AskUser body)

```
== TPU production/canary status @ T+<elapsed> min ==

wandb:
  state           = <running|crashed|finished>
  runtime_min     = <float>
  last_step       = <int>
  heartbeat_age_s = <float>

system metrics (per chip):
  TPU duty cycle  = [c0, c1, c2, c3] %
  HBM utilization = [c0, c1, c2, c3] %
  host RSS        = <float> GiB

compile signals:
  compilation_cause_count = <int>
  first_step_eta          = <compiling|stalled|progressing|past>

last 30 lines of /tmp/train.log (worker 0):
  ...

watchdog verdict: <compiling|stalled|crashed|progressing|success>
```

## The 4 options (AskUser format)

```
1. [question] TPU check-in @ T+<elapsed> min. Continue, abort,
   adjust, or pause?
   [topic] Checkin-action
   [option] Continue (schedule next checkin at T+<next>)
   [option] Abort + diagnose (kill process, run met.metrics_report, escalate)
   [option] Adjust + continue (you provide a patch instruction; I redeploy)
   [option] Pause loop (orchestrator stops; you take manual control)
```

## Behavior per option

### 1. Continue
- Schedule next check-in at the next threshold (T+30 if currently at T+15, etc.)
- After T+90 -> indefinite pause; user must explicitly resume
- Loop returns to WATCH state

### 2. Abort + diagnose
- SSH to all 4 workers, run `tmux kill-session -t training`
- Kill any orphaned `python` processes
- Read tail of `/tmp/train.log` and the `met.metrics_report()` if available
- Hand off to `tpu-diagnoser` for full classification
- Report diagnosis JSON to user, then ESCALATE (orchestrator stops)

### 3. Adjust + continue
- Prompt user for the patch instruction (free text)
- Apply patch via `Edit` tool to the indicated file
- Run `_remote_redeploy.sh` to push code + restart tmux
- Reset T=0 marker (new check-in clock starts)
- Loop returns to WATCH state

### 4. Pause loop
- Stop polling
- Leave the running TPU process untouched (user may want to ssh in manually)
- Update PROGRESS.md with "Paused at T+<elapsed>, user took manual control"
- Orchestrator session goes idle until user gives next instruction

## State persistence (`_artifacts/orch_state.json`)

After every check-in, write:

```json
{
  "deploy_t0_ts": "2026-05-06T05:00:00Z",
  "last_checkin_min": 30,
  "last_checkin_action": "continue",
  "iteration": 1,
  "last_classification": "compile-stall",
  "consecutive_same_classification": 1,
  "checkins_done": [15, 30],
  "next_checkin_min": 45
}
```

This file is the source of truth so we never double-trigger a check-in
across orchestrator turns.
