#!/usr/bin/env python3
"""Promote a W&B sweep winner's hyperparameters into a training config.

WHY THIS EXISTS
---------------
After the proxy sweep (``sweeps/sweep_stage2.yaml``) picks a recipe, the
winning hyperparameters have to land in the production config
(``configs/tpu/stage2_tpu_v6e_v2.yaml``) before the 15k release run. Doing
that by hand is error-prone: the swept *flat* names (``lr_lora``, ``lora_r``,
``lora_alpha_mult`` ...) map onto *nested* config sections, and a hand-edit
easily clobbers the file's block comments or sets the wrong section.

This helper patches **only** the swept keys, **in place**, preserving every
comment and the rest of the file. It uses the *same* flat-name -> section
mapping as ``scripts/train_hierarchical.py`` so the promoted config is
identical to what the sweep agent actually ran.

Two modes:
  * ``--sweep <entity>/<project>/<sweep_id>`` — pull the best run from W&B
    (lowest ``--metric``, default ``val/audio_loss``) and read its config.
  * ``--set k=v ...`` — set values explicitly (when you picked by eye).

This script is backend-agnostic and imports neither torch nor torch_xla.
``wandb`` is imported lazily, only when ``--sweep`` is used.
"""

from __future__ import annotations

import argparse
import re
import sys

# Flat swept-param name -> (config section, nested key). Mirrors the override
# mapping in scripts/train_hierarchical.py::load_config + main(). lora_alpha_mult
# is special: lora.alpha = lora_alpha_mult * lora_r (handled below).
PARAM_MAP: dict[str, tuple[str, str]] = {
    "lr_lora": ("optim", "lr_lora"),
    "lr_depth": ("optim", "lr_depth"),
    "text_weight": ("loss", "text_weight"),
    "warmup_steps": ("train", "warmup_steps"),
    "weight_decay": ("train", "weight_decay"),
    "max_steps": ("train", "max_steps"),
    "val_every": ("logging", "val_every"),
    "val_on_tpu": ("logging", "val_on_tpu"),
    "lora_r": ("lora", "r"),
    "lora_dropout": ("lora", "dropout"),  # Phase E regularization knob
    # lora_alpha_mult -> lora.alpha (= mult * r); resolved in resolve_updates().
}


def _fmt(value) -> str:
    """Render a Python value as the scalar text to write into the YAML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # compact but unambiguous (keep sci-notation small LRs readable)
        return repr(value)
    return str(value)


def set_yaml_scalar_inplace(lines: list[str], section: str, key: str, value: str) -> list[str]:
    """Set ``section.key: value`` in ``lines``, preserving comments + layout.

    Replaces the value of an existing ``key:`` within ``section`` (keeping its
    indentation and any trailing ``# inline comment``). If the key is absent
    from the section, inserts ``  key: value`` right after the section header.

    Args:
        lines: File lines (each ending in ``\\n``).
        section: Top-level mapping name (column 0, e.g. ``optim``).
        key: Nested key under that section.
        value: Pre-formatted scalar text to write.

    Returns:
        The mutated ``lines`` list (also mutated in place).

    Raises:
        KeyError: If ``section`` is not a top-level mapping in the file.
    """
    sec_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    top_re = re.compile(r"^\S")
    # locate the section header
    start = next((i for i, ln in enumerate(lines) if sec_re.match(ln)), None)
    if start is None:
        raise KeyError(f"section '{section}:' not found in config")
    # section body runs until the next column-0 line
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if top_re.match(lines[i]):
            end = i
            break
    key_re = re.compile(rf"^(\s+){re.escape(key)}:\s*([^#\n]*?)(\s*#.*)?$")
    for i in range(start + 1, end):
        m = key_re.match(lines[i])
        if m:
            indent, _old, comment = m.group(1), m.group(2), (m.group(3) or "")
            lines[i] = f"{indent}{key}: {value}{comment}\n"
            return lines
    # not present -> insert just after the header (2-space indent convention)
    lines.insert(start + 1, f"  {key}: {value}\n")
    return lines


def resolve_updates(hp: dict) -> list[tuple[str, str, str]]:
    """Turn a {flat_param: value} dict into [(section, key, formatted_value)].

    Applies the lora_alpha_mult -> lora.alpha = mult * lora_r rule. Unknown
    params are ignored with a warning (the sweep may log extra config keys).
    """
    updates: list[tuple[str, str, str]] = []
    for name, (sec, key) in PARAM_MAP.items():
        if name in hp and hp[name] is not None:
            updates.append((sec, key, _fmt(hp[name])))
    # lora.alpha = lora_alpha_mult * lora_r
    if hp.get("lora_alpha_mult") is not None:
        if hp.get("lora_r") is None:
            print("WARN: --set lora_alpha_mult given without lora_r; "
                  "skipping lora.alpha (cannot compute mult * r).", file=sys.stderr)
        else:
            updates.append(("lora", "alpha", _fmt(int(hp["lora_alpha_mult"]) * int(hp["lora_r"]))))
    known = set(PARAM_MAP) | {"lora_alpha_mult"}
    for extra in sorted(set(hp) - known):
        print(f"note: ignoring non-swept config key '{extra}'", file=sys.stderr)
    return updates


def best_run_config(sweep_path: str, metric: str) -> dict:
    """Fetch the best (minimal-metric) run's config from a W&B sweep.

    Warns if the best run failed the ``sweep/text_ok`` health flag.
    """
    import wandb  # lazy: only needed for --sweep

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = [r for r in sweep.runs if metric in r.summary]
    if not runs:
        raise SystemExit(f"no finished runs in {sweep_path} have metric '{metric}'")
    best = min(runs, key=lambda r: r.summary[metric])
    text_ok = best.summary.get("sweep/text_ok")
    if text_ok is not None and float(text_ok) < 1.0:
        print(f"WARN: best run {best.id} has sweep/text_ok={text_ok} (<1) — "
              "text stream may not have learned. Review before promoting.",
              file=sys.stderr)
    print(f"best run: {best.id} ({metric}={best.summary[metric]:.4f}) — {best.name}")
    # run.config holds the CLI-arg values wandb passed for swept params
    return {k: v for k, v in best.config.items() if k in (set(PARAM_MAP) | {"lora_alpha_mult"})}


def parse_set(pairs: list[str]) -> dict:
    """Parse ``--set k=v`` pairs into a typed dict (int/float/bool/str)."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got '{pair}'")
        k, v = pair.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config YAML to patch in place")
    p.add_argument("--sweep", help="W&B sweep path <entity>/<project>/<sweep_id>")
    p.add_argument("--metric", default="val/composite",
                   help="metric to minimise when picking the winner (default val/composite; "
                        "v0.3 Phase D. Use val/audio_loss for the legacy v2 sweep)")
    p.add_argument("--set", nargs="*", default=[], dest="set_pairs",
                   help="explicit key=value HPs (flat swept names)")
    p.add_argument("--dry-run", action="store_true", help="print changes, don't write")
    args = p.parse_args()

    if not args.sweep and not args.set_pairs:
        raise SystemExit("provide --sweep <path> or --set k=v ...")

    hp: dict = {}
    if args.sweep:
        hp.update(best_run_config(args.sweep, args.metric))
    if args.set_pairs:
        hp.update(parse_set(args.set_pairs))  # explicit --set overrides the pull

    updates = resolve_updates(hp)
    if not updates:
        raise SystemExit("no recognised swept params to promote")

    with open(args.config) as f:
        lines = f.readlines()
    for sec, key, val in updates:
        set_yaml_scalar_inplace(lines, sec, key, val)
        print(f"  {sec}.{key} = {val}")

    if args.dry_run:
        print(f"[dry-run] {args.config} unchanged.")
        return
    with open(args.config, "w") as f:
        f.writelines(lines)
    print(f"promoted {len(updates)} hyperparameters into {args.config}")


if __name__ == "__main__":
    main()
