"""Tests for `experiments/regime_generator.py`.

Covers:
  - Output shape (square n×n tensors, expected stream length).
  - Metadata correctness (gold_regime in HMM vocabulary, shift_index
    points at the right window).
  - Determinism: same seed → identical tensors; different seed → different.
  - 3σ shift is detectable: post-shift window means significantly below
    pre-shift means (validates the shift is actually applied).
  - Stationary streams have no detectable shift.
  - Argument validation.
"""
from __future__ import annotations

import pytest
import torch

from experiments.regime_generator import (
    DEFAULT_N,
    DEFAULT_SIGMA,
    DEFAULT_WINDOWS,
    generate_set,
    generate_stream,
)


# ---------------------------------------------------------------------------
# Shape & metadata
# ---------------------------------------------------------------------------


def test_stationary_stream_shape_and_metadata():
    stream, meta = generate_stream("stationary", seed=42)
    assert len(stream) == DEFAULT_WINDOWS
    for t in stream:
        assert t.shape == (DEFAULT_N, DEFAULT_N)
        assert t.dtype == torch.float32
    assert meta["gold_regime"] == "calm"
    assert meta["regime_kind"] == "stationary"
    assert meta["shift_index"] is None
    assert meta["seed"] == 42


def test_shift_stream_shape_and_metadata():
    stream, meta = generate_stream("shift", seed=7, windows=6)
    assert len(stream) == 6
    for t in stream:
        assert t.shape == (DEFAULT_N, DEFAULT_N)
    assert meta["gold_regime"] == "crash"
    assert meta["regime_kind"] == "shift"
    assert meta["shift_index"] == 3  # windows // 2
    assert meta["seed"] == 7


def test_custom_n_and_windows():
    stream, _ = generate_stream("stationary", seed=0, n=16, windows=8)
    assert len(stream) == 8
    assert stream[0].shape == (16, 16)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_tensors():
    s1, _ = generate_stream("stationary", seed=99)
    s2, _ = generate_stream("stationary", seed=99)
    for a, b in zip(s1, s2):
        assert torch.equal(a, b)


def test_different_seeds_produce_different_tensors():
    s1, _ = generate_stream("stationary", seed=1)
    s2, _ = generate_stream("stationary", seed=2)
    # At least one window must differ; equality across all windows would
    # mean the seed isn't being threaded into the RNG.
    assert any(not torch.equal(a, b) for a, b in zip(s1, s2))


# ---------------------------------------------------------------------------
# Shift is actually applied
# ---------------------------------------------------------------------------


def test_shift_stream_has_detectable_mean_drop():
    """Post-shift windows should have means clearly below pre-shift means.

    With a 3σ shift on n×n=1024 samples per window, the expected per-window
    mean drops by 3σ ≈ 0.009 while the SE on each window mean is
    σ/√1024 ≈ 9.4e-5, so the gap is ~100 SEs — trivially detectable.
    """
    stream, meta = generate_stream("shift", seed=123, windows=8)
    breakpoint = meta["shift_index"]
    pre_means = torch.tensor([t.mean().item() for t in stream[:breakpoint]])
    post_means = torch.tensor([t.mean().item() for t in stream[breakpoint:]])
    gap = pre_means.mean().item() - post_means.mean().item()
    # 3σ shift = 3 * 0.003 = 0.009; allow generous slack for sampling noise.
    assert gap > 2.0 * DEFAULT_SIGMA, (
        f"expected mean gap > 2σ, got {gap:.5f} (pre={pre_means.mean():.5f}, "
        f"post={post_means.mean():.5f})"
    )


def test_stationary_stream_has_no_shift():
    """Stationary streams should not show a midpoint mean break."""
    stream, _ = generate_stream("stationary", seed=456, windows=8)
    half = len(stream) // 2
    pre_means = torch.tensor([t.mean().item() for t in stream[:half]])
    post_means = torch.tensor([t.mean().item() for t in stream[half:]])
    gap = abs(pre_means.mean().item() - post_means.mean().item())
    # No injected shift; gap should be well below 1σ.
    assert gap < DEFAULT_SIGMA, f"unexpected mean gap in stationary stream: {gap}"


# ---------------------------------------------------------------------------
# generate_set
# ---------------------------------------------------------------------------


def test_generate_set_returns_parallel_lists():
    streams, meta = generate_set("shift", n_streams=5)
    assert len(streams) == 5
    assert len(meta) == 5
    seeds = [m["seed"] for m in meta]
    # Distinct seeds, monotonically increasing from base_seed.
    assert seeds == sorted(seeds)
    assert len(set(seeds)) == 5


def test_generate_set_streams_are_independent():
    streams, _ = generate_set("stationary", n_streams=3, base_seed=2000)
    # Two streams with adjacent seeds should not share tensors.
    assert not torch.equal(streams[0][0], streams[1][0])
    assert not torch.equal(streams[1][0], streams[2][0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_regime_raises():
    with pytest.raises(ValueError, match="unknown regime"):
        generate_stream("nonsense", seed=0)  # type: ignore[arg-type]


def test_shift_with_too_few_windows_raises():
    with pytest.raises(ValueError, match="windows >= 2"):
        generate_stream("shift", seed=0, windows=1)


def test_generate_set_zero_streams_raises():
    with pytest.raises(ValueError, match="n_streams"):
        generate_set("stationary", n_streams=0)
