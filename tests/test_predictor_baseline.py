"""Tests for Pipeline E — Forward-Entropy-Benchmark predictor as eval baseline.

Two layers:
  - hermetic unit tests against a hand-rolled duck-typed stub (no FE-Bench
    import needed; always run)
  - one smoke test that actually imports the predictor abstraction from
    Forward-Entropy-Benchmark and exercises a no-skill baseline; skips
    cleanly when the sibling repo (or its transitive STAC-ML mirror) isn't
    available — see HANDOFF.md "Cross-repo dependencies"
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.harness import EvalHarness, HarnessConfig
from eval.predictor_baseline import (
    PIPELINE_E_NAME,
    kirk_outputs_to_observations,
    score_with_predictor,
)
from ulysses_jepa.distill import DistillationItem
from ulysses_jepa.kirk_client import StubKirkClient


class _FakeEntropyPredictor:
    """Duck-typed EntropyPredictor stub. IS_PRODUCTION_KIRK=False (research tier)."""

    IS_PRODUCTION_KIRK = False

    def __init__(self, K: int = 5):
        self._K = K
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def predict(self, observations: np.ndarray, horizon: int) -> np.ndarray:
        self.calls.append((np.asarray(observations).shape, horizon))
        if observations.size == 0:
            return np.zeros(self._K, dtype=np.float32)
        return np.full(self._K, float(observations.mean()), dtype=np.float32)


def test_kirk_outputs_to_observations_extracts_entropy_scalars():
    kirk = StubKirkClient(n=16)
    kos = [kirk.infer(torch.randn(16, 16)) for _ in range(4)]
    obs = kirk_outputs_to_observations(kos)
    assert obs.shape == (4,)
    assert obs.dtype == np.float64
    # StubKirkClient produces entropy in roughly the Uhura-documented band
    assert np.all(obs >= 0.0) and np.all(obs <= 30.0)


def test_kirk_outputs_to_observations_empty_stream():
    obs = kirk_outputs_to_observations([])
    assert obs.shape == (0,)
    assert obs.dtype == np.float64


def test_score_with_predictor_returns_pipeline_e_record():
    kirk = StubKirkClient(n=16)
    kos = [kirk.infer(torch.randn(16, 16)) for _ in range(5)]
    pred = _FakeEntropyPredictor(K=5)

    rec = score_with_predictor(pred, kos, horizon=2)

    assert rec.pipeline == PIPELINE_E_NAME
    assert rec.input_token_count == 5
    assert rec.output_token_count == 5
    assert rec.rouge_l is None
    assert rec.regime_correct is None
    assert rec.cost_usd == 0.0
    assert rec.end_to_end_latency_ms >= 0.0
    assert "horizon=2" in rec.notes
    assert "K=5" in rec.notes
    assert "is_production_kirk=False" in rec.notes
    assert pred.calls == [((5,), 2)]


def test_score_with_predictor_handles_empty_stream():
    pred = _FakeEntropyPredictor(K=3)
    rec = score_with_predictor(pred, [], horizon=1)
    assert rec.input_token_count == 0
    assert rec.output_token_count == 3
    assert "K=3" in rec.notes
    assert pred.calls == [((0,), 1)]


def test_score_with_predictor_records_is_production_kirk_true():
    """A production-tier predictor (IS_PRODUCTION_KIRK=True) is surfaced in notes."""

    class _ProdLikePredictor:
        IS_PRODUCTION_KIRK = True

        def predict(self, observations, horizon):
            return np.zeros(2, dtype=np.float32)

    rec = score_with_predictor(_ProdLikePredictor(), [], horizon=1)
    assert "is_production_kirk=True" in rec.notes


def test_harness_runs_pipeline_e_when_flag_and_predictor_set():
    """End-to-end: HarnessConfig(run_pipeline_e=True) + predictor → an E record."""
    kirk = StubKirkClient(n=16)
    pred = _FakeEntropyPredictor(K=5)
    harness = EvalHarness(
        kirk=kirk,
        predictor=pred,
        config=HarnessConfig(
            run_pipeline_a=False,
            run_pipeline_b=False,
            run_pipeline_c=False,
            run_pipeline_e=True,
            predictor_horizon=3,
        ),
    )
    item = DistillationItem(
        tensors=[torch.randn(16, 16) * 0.01 for _ in range(4)],
        target_text="unused for pipeline E",
    )
    records = harness.evaluate_one(item)
    e_records = [r for r in records if r.pipeline == PIPELINE_E_NAME]
    assert len(e_records) == 1
    assert "horizon=3" in e_records[0].notes
    assert pred.calls == [((4,), 3)]


def test_harness_skips_pipeline_e_when_predictor_missing():
    """Flag without a predictor instance → no E record (silently skipped)."""
    kirk = StubKirkClient(n=16)
    harness = EvalHarness(
        kirk=kirk,
        config=HarnessConfig(
            run_pipeline_a=False,
            run_pipeline_b=False,
            run_pipeline_c=False,
            run_pipeline_e=True,
        ),
    )
    item = DistillationItem(
        tensors=[torch.randn(16, 16) * 0.01 for _ in range(2)],
        target_text="unused",
    )
    records = harness.evaluate_one(item)
    assert all(r.pipeline != PIPELINE_E_NAME for r in records)


def test_real_predictor_abstraction_loadable_when_repo_present():
    """Smoke test: if Forward-Entropy-Benchmark is fully wired, import + run it.

    Skips on hosts that don't have the predictor repo (or its transitive
    STAC-ML-Markets-Inference-Models mirror) — that's the expected case
    in CI today. See HANDOFF.md "Cross-repo dependencies" for the wiring
    instructions and DECISIONS.md D-012 for the why.
    """
    try:
        from entropy_predictor import (  # type: ignore[import-not-found]
            BaselineRecentMeanEntropyPredictor,
            EntropyPredictor,
        )
    except Exception as e:
        pytest.skip(
            f"Forward-Entropy-Benchmark not importable ({type(e).__name__}: {e}); "
            "see HANDOFF.md \"Cross-repo dependencies\"."
        )

    p = BaselineRecentMeanEntropyPredictor(window=10, K=5)
    assert isinstance(p, EntropyPredictor)
    assert p.K == 5
    rng = np.random.default_rng(0)
    out = p.predict(rng.standard_normal(50), horizon=1)
    assert out.shape == (5,)
    assert np.isfinite(out).all()

    # And the integration with score_with_predictor works end-to-end.
    kirk = StubKirkClient(n=16)
    kos = [kirk.infer(torch.randn(16, 16)) for _ in range(8)]
    rec = score_with_predictor(p, kos, horizon=1)
    assert rec.pipeline == PIPELINE_E_NAME
    assert rec.output_token_count == 5
    assert "is_production_kirk=False" in rec.notes
