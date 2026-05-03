"""Synthetic tensor-stream generator for the ADR-003 narrow experiment.

Produces deterministic streams in two regimes:

* ``stationary`` — N(0, σ) draws throughout. Labelled ``gold_regime="calm"``.
* ``shift``      — N(0, σ) draws for the first half of the stream and
                   N(-3σ, σ) draws for the second half (a 3σ downward
                   mean shift at the midpoint, per ADR-003). Labelled
                   ``gold_regime="crash"``.

Tensor shape matches the existing distillation flow (``scripts/distill_teacher.py::synthetic_streams``):
each window is a square ``n × n`` float32 tensor at log-return scale
(σ = 0.003 by default).

This module is **pure**: no LLM calls, no I/O, no network. Labelling
(producing ``target_text`` from a teacher LLM) is a separate operational
step — see ``scripts/distill_regime_shift.py`` and the
``distill-regime-shift`` Makefile target.

Regime labels (``"calm"``, ``"crash"``) are drawn from
``eval/hmm_baseline.py::GaussianHMM.state_names`` so the existing
``regime_correct`` string-match in ``eval/harness.py`` works unchanged.
"""
from __future__ import annotations

from typing import Literal

import torch


Regime = Literal["stationary", "shift"]

# Match the scale used by `scripts/distill_teacher.py::synthetic_streams`.
DEFAULT_N: int = 32
DEFAULT_WINDOWS: int = 4
DEFAULT_SIGMA: float = 0.003

# HMM-vocabulary regime tags, consistent with `eval/hmm_baseline.py`.
_REGIME_TO_GOLD: dict[str, str] = {
    "stationary": "calm",
    "shift": "crash",
}


def generate_stream(
    regime: Regime,
    seed: int,
    n: int = DEFAULT_N,
    windows: int = DEFAULT_WINDOWS,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[list[torch.Tensor], dict]:
    """Generate one synthetic stream + per-stream metadata.

    Returns ``(stream, metadata)`` where ``stream`` is a list of ``windows``
    tensors of shape ``(n, n)`` and ``metadata`` carries:

    * ``gold_regime`` — HMM state name (``"calm"`` or ``"crash"``).
    * ``regime_kind`` — the input ``regime`` arg, for traceability.
    * ``shift_index`` — window index of the breakpoint, or ``None`` for
      stationary streams.
    * ``seed`` — the seed used, so a stream can be reproduced exactly.
    """
    if regime not in _REGIME_TO_GOLD:
        raise ValueError(
            f"unknown regime {regime!r}; expected one of {sorted(_REGIME_TO_GOLD)}"
        )
    if windows < 2 and regime == "shift":
        raise ValueError(f"shift streams need windows >= 2, got {windows}")

    gen = torch.Generator().manual_seed(seed)
    base = [
        torch.randn(n, n, generator=gen, dtype=torch.float32) * sigma
        for _ in range(windows)
    ]

    if regime == "stationary":
        stream = base
        shift_index: int | None = None
    else:
        shift_index = windows // 2
        post_mu = -3.0 * sigma
        stream = [
            (t + post_mu) if i >= shift_index else t for i, t in enumerate(base)
        ]

    metadata = {
        "gold_regime": _REGIME_TO_GOLD[regime],
        "regime_kind": regime,
        "shift_index": shift_index,
        "seed": seed,
    }
    return stream, metadata


def generate_set(
    regime: Regime,
    n_streams: int,
    base_seed: int = 1000,
    n: int = DEFAULT_N,
    windows: int = DEFAULT_WINDOWS,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[list[list[torch.Tensor]], list[dict]]:
    """Generate ``n_streams`` independently-seeded streams of one regime.

    Returns ``(streams, metadata_list)`` so the streams list can be passed
    straight to ``ulysses_jepa.distill.label_with_teacher`` while the
    metadata list is reattached to the resulting ``DistillationItem``s
    by the caller (see ``scripts/distill_regime_shift.py``).
    """
    if n_streams < 1:
        raise ValueError(f"n_streams must be >= 1, got {n_streams}")
    streams: list[list[torch.Tensor]] = []
    meta: list[dict] = []
    for i in range(n_streams):
        s, m = generate_stream(
            regime, seed=base_seed + i, n=n, windows=windows, sigma=sigma
        )
        streams.append(s)
        meta.append(m)
    return streams, meta
