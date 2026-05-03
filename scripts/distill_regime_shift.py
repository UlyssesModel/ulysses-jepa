"""Distill ADR-003 regime-shift streams via the gemma4:31b teacher.

Reads no input data — generates synthetic streams via
``experiments.regime_generator``, then runs them through
``ulysses_jepa.distill.label_with_teacher`` to produce gold narrations
from the teacher LLM. The resulting ``DistillationItem``s carry the
``gold_regime`` metadata the harness needs for the ``regime_correct``
metric.

The output .pt file mixes both regimes (stationary and shift); the
harness consumes it directly. Per-regime breakdowns are recoverable from
``item.metadata["regime_kind"]``.

Example:

    python scripts/distill_regime_shift.py \\
        --output data/distilled_regime_shift.pt \\
        --n-streams-per-regime 5 \\
        --teacher-base-url http://127.0.0.1:11434 \\
        --teacher-model gemma4:31b
"""
from __future__ import annotations

import argparse
import os

from experiments.regime_generator import (
    DEFAULT_N,
    DEFAULT_WINDOWS,
    generate_set,
)
from ulysses_jepa.distill import label_with_teacher, save_distilled
from ulysses_jepa.injection import ScottyClient, ScottyConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n-streams-per-regime", type=int, default=5,
                   help="Number of independently-seeded streams per regime")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--windows-per-stream", type=int, default=DEFAULT_WINDOWS)
    p.add_argument("--base-seed-stationary", type=int, default=1000)
    p.add_argument("--base-seed-shift", type=int, default=2000)
    p.add_argument("--teacher-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--teacher-model", default="gemma4:31b")
    args = p.parse_args()

    stationary_streams, stationary_meta = generate_set(
        regime="stationary",
        n_streams=args.n_streams_per_regime,
        base_seed=args.base_seed_stationary,
        n=args.n,
        windows=args.windows_per_stream,
    )
    shift_streams, shift_meta = generate_set(
        regime="shift",
        n_streams=args.n_streams_per_regime,
        base_seed=args.base_seed_shift,
        n=args.n,
        windows=args.windows_per_stream,
    )
    streams = stationary_streams + shift_streams
    meta_list = stationary_meta + shift_meta
    print(f"[distill-regime-shift] {len(streams)} streams "
          f"({args.n_streams_per_regime} stationary + {args.n_streams_per_regime} shift)")

    teacher = ScottyClient(ScottyConfig(
        base_url=args.teacher_base_url, model=args.teacher_model
    ))
    items = label_with_teacher(teacher, streams, show_progress=True)
    for item, meta in zip(items, meta_list):
        item.metadata = meta

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_distilled(items, args.output)
    print(f"[distill-regime-shift] wrote {len(items)} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
