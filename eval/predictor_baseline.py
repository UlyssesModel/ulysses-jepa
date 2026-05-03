"""Pipeline E — Forward-Entropy-Benchmark predictor as an eval baseline.

Wraps Forward-Entropy-Benchmark's EntropyPredictor (a `(1-D obs, horizon)
-> (K,) entropy-forecast` contract) into the EvalRecord shape used by the
harness for the existing A/B/C/D pipelines.

Per `DECISIONS.md` D-012 and ADR-002, EntropyPredictor and KirkClient are
*different* abstractions that don't compose by inheritance. Pipeline E
therefore consumes the same KirkOutput stream the other pipelines see —
it just extracts the entropy time series from `kos` and feeds it through
the predictor's own `(observations, horizon)` contract for a separate
forecast-quality reading.

The predictor abstraction lives in a sibling repo without packaging;
imports are deferred and guarded so this module loads cleanly even when
Forward-Entropy-Benchmark isn't on `sys.path`. Callers that hand in a
predictor instance opt into the dependency.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from ulysses_jepa.interfaces import KirkOutput

from .metrics import EvalRecord


PIPELINE_E_NAME = "E_predictor_baseline"


def kirk_outputs_to_observations(kos: Sequence[KirkOutput]) -> np.ndarray:
    """Pull the entropy scalar from each KirkOutput as a 1-D obs prefix.

    EntropyPredictor.predict() consumes a 1-D float array. We treat the
    Kirk-stream's entropy time series as the observation prefix — the
    predictor forecasts entropy at `t + horizon` from it. Other fields
    in KirkOutput (Array, Vector) are out of scope for this contract;
    the projection adapter remains the consumer for those.
    """
    if not kos:
        return np.empty(0, dtype=np.float64)
    vals: list[float] = []
    for ko in kos:
        e = ko.entropy
        if torch.is_tensor(e):
            e = e.detach().cpu()
            if torch.is_complex(e):
                e = e.real
            vals.append(float(e.item()))
        else:
            vals.append(float(e))
    return np.asarray(vals, dtype=np.float64)


def score_with_predictor(
    predictor: Any,
    kos: Sequence[KirkOutput],
    horizon: int = 1,
) -> EvalRecord:
    """Run an EntropyPredictor on a Kirk-output stream and return one EvalRecord.

    Args:
        predictor: an instance whose `.predict(observations, horizon)`
            returns a `(K,)` numpy vector — the EntropyPredictor contract
            from Forward-Entropy-Benchmark. Duck-typed; we don't import
            the ABC here so this module stays loadable on hosts where the
            predictor repo isn't wired up.
        kos: stream of KirkOutputs from a `KirkClient.infer_stream()` call.
        horizon: forecast horizon to ask the predictor for.

    The returned record uses `pipeline = "E_predictor_baseline"`.
    `rouge_l` / `regime_correct` are `None` because this is a numerical
    forecast, not a narration or single-label classification — the
    notes field carries the predictor's last-observed entropy, the
    forecast mean, and the `IS_PRODUCTION_KIRK` flag for traceability.
    """
    obs = kirk_outputs_to_observations(kos)
    t0 = time.perf_counter()
    pred = np.asarray(predictor.predict(obs, horizon), dtype=np.float64)
    t1 = time.perf_counter()

    last_obs = float(obs[-1]) if obs.size else float("nan")
    forecast_mean = float(pred.mean()) if pred.size else float("nan")
    is_prod = bool(getattr(type(predictor), "IS_PRODUCTION_KIRK", False))

    return EvalRecord(
        pipeline=PIPELINE_E_NAME,
        input_token_count=int(obs.size),
        output_token_count=int(pred.size),
        prefill_latency_ms=(t1 - t0) * 1000,
        decode_latency_ms=0.0,
        end_to_end_latency_ms=(t1 - t0) * 1000,
        output_text="",
        rouge_l=None,
        regime_correct=None,
        cost_usd=0.0,
        notes=(
            f"EntropyPredictor.predict horizon={horizon} "
            f"K={pred.size} last_obs={last_obs:.4f} "
            f"forecast_mean={forecast_mean:.4f} "
            f"is_production_kirk={is_prod}"
        ),
    )
