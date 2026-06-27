# TPU capacity log -- observed queue times + autonomous fallback policy

## 2026-05-13 update

Production topology has pivoted to **single-host v6e-8 spot in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`, profile shorthand `v6e-8-eu`). Iter
24h completed 5000/5000 baseline steps on this profile and uploaded
the final canonical checkpoint; `opt-prod5k` then completed 5000/5000
optimized steps with checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`.
Phase 4 now uses the same v6e-8 EU profile, with private-repo-safe
startup via `REPO_TARBALL_GS_URI`. v4-32 spot in `us-central2-b` is
now legacy. The fallback policy in section 1 below is preserved as the
canonical autonomous decision tree for new capacity attempts, but
day-to-day operation starts at the v6e-8 EU profile unless the user
explicitly asks for another topology.

**Purpose:** Record real queue-wait durations so future sessions can
make smart autonomous decisions about which TRC slice to try. Updated
after every QR submission attempt.

The authoritative TRC quota table lives in
[`docs/tpu-trc-allocation.md`](./tpu-trc-allocation.md); this file
augments it with *observed* capacity behaviour.

---

## 1. Autonomous fallback policy

When a session needs to spin up a TPU and finds the first choice
queued, it should follow this decision tree:

```text
1. Try on-demand v4-64 in us-central2-b (TRC email recommends on-demand first)
   Wait up to 10 min.
   If ACTIVE -> proceed.
   If still WAITING_FOR_RESOURCES after 10 min:

2. Try spot v4-32 in us-central2-b (same zone; zero infra change)
   Wait up to 10 min.
   If ACTIVE -> proceed.
   If still WAITING_FOR_RESOURCES after 10 min:

3. Try spot v5e-64 in europe-west4-b (biggest spot grant; needs
   v5e-tuned config + v2-alpha-tpuv5-lite runtime)
   Wait up to 15 min (different region, may have more slack).
   If ACTIVE -> proceed.
   If still WAITING_FOR_RESOURCES after 15 min:

4. Try spot v5e-64 in us-central1-a (same chip family, US zone)
   Wait up to 10 min.
   If ACTIVE -> proceed.
   If still WAITING_FOR_RESOURCES:

5. Try spot v6e-64 in europe-west4-a or us-east1-d (newest gen,
   less battle-tested in this codebase; needs v2-alpha-tpuv6e runtime)
   Wait up to 15 min.

6. All failed -> stop and ask the user. Do NOT auto-delete QRs
   the user might want to keep queued.
```

**Important rules:**
- Never submit more than one QR at a time. A queued QR still
  consumes quota; two parallel QRs for the same type will both
  block.
- Delete the failed QR before submitting the next one.
- Record every attempt in section 2 below (timestamp, profile, wait
  duration, outcome).
- After 3 consecutive failures in the same day, stop and ask
  the user rather than burning the retry budget.

## 2. Observed queue durations

| Date | Time (UTC) | Profile | Tier | Wait | Outcome | Notes |
|---|---|---|---|---|---|---|
| 2026-05-03 | ~12:00 | v4-64 on-demand (us-central2-b) | on-demand | <5 min | ACTIVE | Original probe run; QR provisioned quickly. |
| 2026-05-03 | ~14:00 | v4-64 on-demand (us-central2-b) | on-demand | ~0 | ACTIVE | Real composite fsdpv2_lora compile; QR already up from earlier. |
| 2026-05-05 | 09:23 | v4-32 spot (us-central2-b) | spot | 17+ min | WAITING | Cancelled after 17 min; no progress. Spot v4 in this zone is contended. |
| 2026-05-05 | 09:46 | v4-64 on-demand (us-central2-b) | on-demand | 10+ min | WAITING/CANCELLED | Cancelled after 10 min per fallback policy timeout; on-demand v4 is also genuinely contended in this zone today. |
| 2026-05-05 | 12:45 | v5e-64 spot (europe-west4-b) | spot | 2 min PROVISIONING | FAILED | `IN_USE_ADDRESSES limit (8)` -- regional IP quota too small for 8-host v5e-64 with external IPs. Not a TPU capacity issue. Mitigations: (a) request IP quota bump, (b) use `--internal-ips` + Private Google Access + GCS-mirrored dataset. |
| 2026-05-05 | 13:02 | v6e-64 spot (us-east1-d) | spot | 5 min PROVISIONING | FAILED | Code 13 "an internal error has occurred" -- same failure pattern as v5e-64 ew4b (PROVISIONING -> SUSPENDING -> FAILED in <5 min). 8-host slice + 8 IP cap = identical IP quota gate. |
| 2026-05-05 | 13:42 | v4-32 spot (us-central2-b) | spot | 3.5 min PROVISIONING | ACTIVE | Retry of the 09:23 attempt with same config + same QR submission. Spot pool cleared in the intervening hours; ACTIVE reached at 13:51 UTC. First successful canary launch of the day. |
| 2026-05-08 | -- | v4-32 spot (us-central2-b) | spot | hours | SUSPENDED | v4 spot pool reclaimed by other TRC users; QR has been SUSPENDED for several hours with no recovery. Pivoted away. |
| 2026-05-08 | -- | v6e-8 spot (europe-west4-a) | spot | ~4:44 | ACTIVE | Single-host (8 chips), 32 GiB HBM/chip. ACTIVE within 5 min from QR submission. Iter 13b (run `zd42n7di`) reached 20 steps + canonical save in 23.3 min wall. |
| 2026-05-09 | 15:25 | v6e-8 spot (europe-west4-a) | spot | ~3 min to ACTIVE | ACTIVE / COMPLETED | Iter 24h production retry. QR reached ACTIVE at 15:28 UTC; run `7rrjupc7` completed 5000/5000 steps at 2026-05-10T01:47:24Z and uploaded `step_005000_final` (8 objects, 2.37 GiB). |
| 2026-05-13 | ~11:30 | v6e-8 spot (europe-west4-a) | spot | immediate preemptions / capacity retry | FAILED | Phase 4 `opt-4-depth32` first attempts hit spot preemption and then QR capacity error code 8. User chose to keep retrying same zone. |
| 2026-05-13 | ~12:00 | v6e-8 spot (europe-west4-a) | spot | ~few min to ACTIVE | ACTIVE / COMPLETED | Relaunched Phase 4 with `REPO_TARBALL_GS_URI=gs://tinyaya-stage2-tpu/code/phase4-depth32-20260513T120005Z.tar.gz`; startup avoided private GitHub clone and W&B `i15igq8d` completed 300/300 steps with exit 0. |

*(Update this table after every attempt.)*

## 3. Per-profile heuristics

Based on the limited data so far:

| Profile | Expected wait | Confidence | Notes |
|---|---|---|---|
| v4-64 on-demand (uc2b) | 0-15 min | LOW (1 success, 1 pending) | Succeeds when no other TRC user in this zone has the on-demand quota occupied. |
| v6e-8 spot (ew4a) | 0-10 min when capacity is available; retry after preemption/capacity code 8 | HIGH | Current validated production/optimization profile. One host = one external IP, fits quota, no multi-host rendezvous. |
| v4-32 spot (uc2b) | 0-20 min, hour-by-hour | MEDIUM (1 fail at 09:23, 1 success at 13:42) | Legacy fallback. Spot pool clears on a sub-hour timescale. Cancel-and-retry within a few hours is a valid strategy. 4 hosts = 4 IPs, fits the 8-IP regional cap easily. |
| v5e-64 spot (ew4b) | 2 min to PROVISIONING then FAILED | LOW (1 IP-quota fail) | Hits `IN_USE_ADDRESSES` regional quota (8) because 8-host slice needs 8 external IPs. Use `--internal-ips` (requires Private Google Access on subnet + GCS-mirrored dataset) or request IP quota bump. |
| v5e-64 spot (uc1a) | unknown | NONE | Never tried. |
| v6e-64 spot (ue1d) | 5 min PROVISIONING then FAILED | LOW (1 IP-quota fail) | Same root cause as v5e-64: 8-host slice + regional 8-IP cap. Code 13 ("internal error") instead of explicit IP-quota message, but identical failure pattern and timing. |

**Actionable insight:** v5e-64 in europe-west4-b is the strongest
fallback candidate because (a) 64 chips is the biggest spot grant we
have, (b) europe region may have lower TRC competition than us-central2,
(c) the canary config is already retuned for v5e HBM/chip constraints.
Try it THIRD (after on-demand v4 and spot v4).

## 4. Session checklist for autonomous QR launch

Before submitting any QR, a droid session must:

- [ ] Read this file (docs/tpu-capacity-log.md) for the latest
  observed wait times.
- [ ] Read docs/tpu-trc-allocation.md for the quota table.
- [ ] Check gcloud compute tpus queued-resources list to confirm no
  existing QR is consuming quota.
- [ ] Pick the first profile from the decision tree in section 1.
- [ ] Submit the QR, start a 10-min poller.
- [ ] If ACTIVE: proceed to probe + canary.
- [ ] If still WAITING: delete the QR, advance to the next profile,
  update section 2 with the observed wait.
- [ ] After 3 failures: stop and ask the user.

## 5. Cross-references

- docs/tpu-trc-allocation.md -- authoritative TRC quota table.
- scripts/tpu/launch_spot.sh -- the TRC_PROFILE-aware launch wrapper.
- configs/stage2_tpu_canary_v4_spot.yaml -- v4-32 spot canary.
- configs/stage2_tpu_canary_v5e_spot.yaml -- v5e-64 spot canary.
- configs/stage2_tpu_canary.yaml -- original v5litepod-16 canary
  (no longer matches any quota we have; kept for reference).
- .factory/memories.md -- decision entry for the fallback policy.

## 6. Maintenance

Prune entries older than 90 days from section 2 during the monthly
archive-progress cycle. The heuristics in section 3 should be updated
whenever a new profile is tried (success or failure).

## 7. Known issues + workarounds

### 7.1 `IN_USE_ADDRESSES limit` on v5e/v6e slices

**Symptom:** QR transitions `WAITING_FOR_RESOURCES` -> `PROVISIONING`
-> `SUSPENDING` -> `FAILED` within 2 minutes with error
`You have reached IN_USE_ADDRESSES limit. [EID: ...]`.

**Cause:** Regional Compute Engine `IN_USE_ADDRESSES` quota is **8**
in every region we have TPU quota in (europe-west4, us-central1,
us-central2). v5litepod-64 (8 hosts) and v6e-64 (8 hosts) want one
external IP per host = 8 IPs total, hitting the limit exactly.
v4-64 (4 hosts? or 8 hosts depending on topology) is right at the
edge. Even momentary other usage (a Cloud Build worker, a Compute
VM) pushes us over.

**Workaround A (recommended):** Use `INTERNAL_IPS=1` with
`launch_qr.sh` (requires the `--internal-ips` flag plumbing added
2026-05-05). Prerequisites:
  1. Private Google Access enabled on `default` subnet in the target
     region (`gcloud compute networks subnets update default
     --region=<R> --enable-private-ip-google-access`).
  2. Dataset mirrored to `gs://tinyaya-stage2-tpu/encoded/` so the
     startup script can pull it via GCS instead of HF Hub.
  3. Code tarball already lives in GCS (already done: `code/tinyaya-repo-hot.tar.gz`).

**Workaround B:** Request a regional IP quota bump from GCP support
(typical turnaround 1-3 business days).

**Workaround C:** Pick a smaller slice (v4-32 = 4 hosts = 4 IPs;
stays under quota) at the cost of less data parallelism.

### 7.2 GCP quota increase request (drafted 2026-05-05)

A quota-increase email was drafted for the user to send to GCP
(both via the IAM & Admin console and to
`cloud-tpu-trc-support@google.com`). Request: raise
`IN_USE_ADDRESSES` from 8 to 32 in europe-west4, us-central1,
us-east1, and us-central2. Draft is at
`_artifacts/gcp-quota-increase-request.md`. Until the quota is
bumped, v5e-64 and v6e-64 slices remain blocked; v4-32 and v4-64
stay viable (4 hosts = 4 IPs = fits cap).

## 8. Compile time observations

Recording how long XLA compile took on each successful run, so the
fallback policy can budget realistic wall time before declaring a
canary failed.

| Date | Slice | Strategy | scan_layers state | Forward step 1 | Backward step 1 | Notes |
|---|---|---|---|---|---|---|
| 2026-05-03 | v4-64 on-demand uc2b | fsdpv2_lora | (probe only, no training) | n/a | n/a | Per `.factory/memories.md` 2026-05-03 entries; 25+ min compile feared, never completed mid-session. |
| 2026-05-05 | v4-32 spot uc2b | fsdpv2_lora | TypeError -> manual loop | ~20 min | **~4h 25min** | scan_layers fallback was triggered on every layer call (40+ TypeError lines per step). Backward unrolled 36 CohereDecoderLayer + 6 MoshiDecoderLayer instances into HLO; XLA optimisation passes spent multi-hour wall time before producing executable. Process died at 8h 24min wall (likely supervisor timeout or OOM-kill); supervisor loop restarted from scratch with no XLA cache reuse. |

**Lesson:** the documented "25+ min compile" budget assumes
`scan_layers` is taking the fast path. With the manual-loop fallback
the compile is 5-10x longer. Until the `_KwargBoundLayer` patch in
`.factory/memories.md` 2026-05-05 "scan_layers TypeError" is applied,
do not budget less than 5h wall time for first-canary compile.
