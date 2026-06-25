# TRC TPU allocation -- authoritative record

## 2026-05-13 update

Topology pivot: the **validated production path is now single-host TPU
v6e-8 in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`, profile shorthand `v6e-8-eu` in
`scripts/tpu/launch_spot.sh`). Iter 24h completed 5000/5000 baseline
steps on this profile and uploaded
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`.
The optimized `opt-prod5k` run then completed 5000/5000 steps with
checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`.
Phase 4 activation/depth sweeps now start from the same v6e-8 EU
profile; `opt-4-depth32` completed its 300-step gate on this profile
with exit 0.
The v4-32 spot in `us-central2-b` ran iter 1-11 and is now legacy.
The TRC allocation table below has not changed -- but the **selected
zone / tier for daily operation** has pivoted to v6e-8 spot in
`europe-west4-a`. The allocation table, decision tree, and policy text
below remain authoritative for the full quota grant; they are simply
augmented by the production-validated choice to run on v6e-8 spot.
v6e-64 in `europe-west4-a` is the next scale-up target once spot
capacity allows.

**Status:** Active
**Captured:** 2026-05-05
**Recipient:** `mayankbhaskar007@gmail.com`
**GCP project:** `ml-pipelines-315702`
**Grant duration:** 90 days (free Cloud TPU usage; non-TPU GCP services
still billed; Google may reclaim capacity at any time)
**Source:** TRC welcome email from `trc-support@google.com` (subject
"You have access to free Cloud TPUs"), pasted by the user on
2026-05-05 and archived verbatim in this document.

This file is the **single source of truth** for which TPU types,
zones, and tiers we are entitled to. The older table in
`docs/tpu-launch-plan.md` §2 is **SUPERSEDED** by this document; that
section now points here.

---

## 1. Allocation table

| Quantity | TPU type | Zone | Tier |
|---:|---|---|---|
| 32 chips | Cloud TPU v4 | `us-central2-b` | **spot** |
| 32 chips | Cloud TPU v4 | `us-central2-b` | **on-demand** |
| 64 chips | Cloud TPU v5e | `europe-west4-b` | spot |
| 64 chips | Cloud TPU v5e | `us-central1-a` | spot |
| 64 chips | Cloud TPU v6e | `europe-west4-a` | spot |
| 64 chips | Cloud TPU v6e | `us-east1-d` | spot |

The dual v4 lines in `us-central2-b` are independent quotas: 32 chips
of on-demand AND 32 chips of spot in the same zone.
`v6e-8-eu` is a smaller single-host slice requested against the v6e
spot capacity in `europe-west4-a`; it is not a separate TRC grant row.

### Profile shorthands

`scripts/tpu/launch_spot.sh` accepts the following `TRC_PROFILE`
values, each mapping to a single row above:

| `TRC_PROFILE` | TPU type passed to `gcloud` | Zone | Notes |
|---|---|---|---|
| `v6e-8-eu` (current production profile) | `v6e-8` | `europe-west4-a` | Single-host 8-chip slice validated by iter 24h and `opt-prod5k`; fits external-IP quota and avoids multi-host rendezvous. |
| `v4-32-uc2b` | `v4-32` | `us-central2-b` | Same zone as on-demand v4 quota; smallest legacy v4 fallback. |
| `v5e-64-ew4b` | `v5litepod-64` | `europe-west4-b` | Largest v5e slice; matches v5litepod-* canary tuning. |
| `v5e-64-uc1a` | `v5litepod-64` | `us-central1-a` | Same chip family as `v5e-64-ew4b`, US zone. |
| `v6e-64-ew4a` | `v6e-64` | `europe-west4-a` | Newest gen; needs `v2-alpha-tpuv6e` runtime. |
| `v6e-64-ue1d` | `v6e-64` | `us-east1-d` | Same as above, US zone. |

The on-demand v4 quota does not need a profile here -- it is already
the default of `scripts/tpu/launch_qr.sh` (no `--spot`).

## 2. Important caveats (verbatim from the email)

> This free 90-day trial is only available for new Cloud TPUs you
> create in the zones listed above. To avoid charges, please be sure
> to create your Cloud TPUs in the appropriate zone.

> While your Cloud TPUs are free, you'll still be charged for the
> rest of the GCP services you use. If you have a new account, Google
> Cloud's $300 USD introductory credit may completely offset these
> costs, and you can minimize costs even more by utilizing the new
> Cloud TPU VM architecture.

> Please note that demand for Cloud TPUs is high, so we can't
> guarantee you'll get to use all of your TPU quota. Google reserves
> the right to reclaim TRC quota and TRC Cloud TPU capacity at any
> time.

> If you have access to both on-demand and preemptible quotas, we
> recommend preferring on-demand and falling back to preemptible
> if/when on-demand is not available.

> If you have access to v2-8 and/or v3-8 quotas, please be aware
> that these individual devices cannot be used in pod configurations.
> *(Not applicable to this grant -- no v2/v3 quotas.)*

> If you encounter an error message indicating that your quota is
> exhausted, confirm that you have deleted any unused Cloud TPUs
> and/or Queued Resources that may still be consuming quota.

## 3. How to pick a zone (decision tree)

```text
Is the on-demand v4 quota in us-central2-b free right now?
+-- Yes -> use the on-demand path (existing launch_qr.sh defaults; no --spot)
+-- No  -> launch_spot.sh; pick TRC_PROFILE based on goal:
          +-- "I want minimal infra change" .................. v4-32-uc2b (default)
          +-- "I want maximum throughput on v5e family" ...... v5e-64-ew4b
          +-- "I am US-based and want low latency" ........... v5e-64-uc1a OR v6e-64-ue1d
          +-- "I want the validated production path" ......... v6e-8-eu
          +-- "I want to test newest multi-host hardware" .... v6e-64-ew4a OR v6e-64-ue1d
          +-- "Probe / smoke test only, smallest cost" ....... v4-32-uc2b (still smallest in this grant)
```

**Operational default for this repo's active TPU work:** `v6e-8-eu`. It uses one
host with eight chips, fits the current external-IP quota, and is the
only profile that has completed both the iter 24h baseline and
`opt-prod5k`. The script-level fallback default remains
`v4-32-uc2b` for historical compatibility, but active production and
optimization commands set `TRC_PROFILE=v6e-8-eu` explicitly.

## 4. Program requirements (we accepted these)

Per the welcome email, TRC participants are expected to:

- Share TRC-supported research with the world (peer-reviewed
  publications, open-source code, blog posts, or other means).
- Share detailed feedback with Google to help improve the TRC
  program and the underlying Cloud TPU platform.
- Conduct research in accordance with the Google AI Principles.
- Accept Google's Terms and Conditions.
- Acknowledge that the participant's information will be used
  in accordance with Google's Privacy Policy.

Acknowledged. Outputs from this repo (code, blog post, eval results)
are intended to satisfy the publication requirement.

## 5. Support

- Email: `trc-support@google.com`.
- Discord: `#tpu-research-cloud` channel on the Google Developer
  Community Discord server.
- Recommended reading: PyTorch/XLA performance debugging blog series
  (Parts I-III) -- already linked from `docs/tpu-changes.md`.

## 6. Footer (verbatim from the email)

> Google LLC 1600 Amphitheatre Parkway, Mountain View, CA 94043
>
> This email was sent to `mayankbhaskar007@gmail.com` to update you
> about important information regarding your Google Cloud Platform
> account.

## 7. Cross-references

- `docs/tpu-launch-plan.md` §2 -- now redirects here. The May-2026
  table in that document was a draft and is **SUPERSEDED**.
- `.factory/memories.md` -- "TRC allocation captured" decision entry
  (2026-05-05) records the supersedure.
- `scripts/tpu/launch_spot.sh` -- the launch wrapper that materialises
  the `TRC_PROFILE` shorthands above into `gcloud` flags.
- `configs/stage2_tpu_canary_v4_spot.yaml` and
  `configs/stage2_tpu_v4_spot.yaml` -- the v4-32 spot retunes of the
  canary and production configs.
