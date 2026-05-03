# ADR-002 — KirkClient and EntropyPredictor as parallel abstractions

**Status:** Accepted
**Date:** 2026-05-02
**Author:** John Edge
**Supersedes:** none
**Related:** [DECISIONS.md D-006](../../DECISIONS.md), [D-007](../../DECISIONS.md), [D-012](../../DECISIONS.md)
**Pinned upstream:** Forward-Entropy-Benchmark @ `e2732baf07b55aad32fec635d4c4fef9759518e9`

## Context

`ulysses-jepa` defines `KirkClient` (in `src/ulysses_jepa/interfaces.py`)
and ships three concrete implementations in `src/ulysses_jepa/kirk_client.py`:
`StubKirkClient`, `KirkPipelineClient`, `KirkSubprocessClient`.

A sibling repo, `Forward-Entropy-Benchmark`, defines `EntropyPredictor`
(in `scripts/entropy_predictor.py`) and ships several concrete
implementations: `BaselineRecentMeanEntropyPredictor`, `MatsushitaPredictor`,
`KirkEntropyPredictor` (research-tier), `ParquetKirkPredictor`,
`TiberiusKirkPredictor`, `KirkEntropyFromParquetPredictor`.

At a glance these look like duplicate abstractions — both deal with
"things that produce Kirk-shaped numbers." A naïve consolidation pass
would try to retire one and have the other adopt its callers. **Don't.**

## Decision

Keep both abstractions side by side. They answer different questions
and live at different points in the pipeline.

| | `KirkClient.infer` | `EntropyPredictor.predict` |
|---|---|---|
| Owner repo | `ulysses-jepa` | `Forward-Entropy-Benchmark` |
| Input | `torch.Tensor` (N×N feature matrix) | `np.ndarray` (1-D observation prefix) + `horizon: int` |
| Output | `KirkOutput` — `layer2_input`, `layer2_reconstruction` (Array), `layer2_marginals` (Vector), `entropy` (Scalar), plus `mode`/`timestamp_ns` | `np.ndarray` of shape `(K,)` — forward-entropy estimate at `t + horizon` |
| Job | Drive the projection adapter (Kirk Layer-2 trio → LLM embedding sequence; D-006) | Forecast forward entropy on a 1-D series; benchmark scoring |
| Statefulness | `active_inference` modes update model state across calls | Stateless per `predict` call |

The `EntropyPredictor` abstraction has no layer-2 reconstruction surface.
`predict()` returns only `(K,)` entropy. There is no path from it to the
projection adapter's input contract — the adapter consumes Array + Vector,
which `EntropyPredictor` does not expose. Trying to merge would either
delete the adapter's input source (broken) or force an upstream extension
(out of scope; cross-repo PR with no clear owner).

## Why we don't merge

1. **Information loss.** Pulling `KirkClient` callers onto
   `EntropyPredictor` would lose the Array (32×32 reconstruction) and
   Vector (64-wide marginals) the adapter needs. The math in
   `src/ulysses_jepa/adapter.py` consumes those tensors directly.
2. **Different input shapes.** `KirkClient` consumes a 2-D tensor (the
   N×N feature matrix Uhura emits). `EntropyPredictor` consumes a 1-D
   observation prefix. They aren't the same data wearing different hats.
3. **No upstream owner.** No repo currently owns a unified abstraction.
   `KirkClient` is local to `ulysses-jepa` (D-007). `EntropyPredictor`
   is local to `Forward-Entropy-Benchmark`. Forcing a merge picks a
   winner where there isn't one.

## How they compose

They compose by being used at different points:

- `src/ulysses_jepa/*` consumes `KirkClient` to drive the adapter.
- `eval/predictor_baseline.py` (Pipeline E) consumes the same
  `KirkClient` stream and additionally wraps `EntropyPredictor` to
  produce a forecast-quality reading on the entropy time series. The
  predictor and the adapter see the same source data; they just project
  it through different abstractions.

This is intentional: the eval harness already runs four parallel
pipelines (A/B/C/D); E is a fifth orthogonal reading. No abstraction
sharing needed.

## What would make us revisit

Revisit consolidation if Forward-Entropy-Benchmark grows a method that
exposes the layer-2 trio. Concretely, something like:

```python
class TiberiusKirkPredictor(EntropyPredictor):
    def infer_layer2(self, tensor: np.ndarray) -> dict:
        # Returns {"reconstruction": (n, n), "marginals": (2n,), "entropy": ()}
        ...
```

If that lands, `ulysses-jepa` could replace its three `KirkClient`
implementations with thin wrappers over `infer_layer2` and let the
predictor repo own the production paths. Until then, the two
abstractions are parallel and that is correct.

## References

- Forward-Entropy-Benchmark `scripts/entropy_predictor.py` @ `e2732baf0` —
  `EntropyPredictor` ABC, `predict(obs, horizon) -> (K,)`.
- Forward-Entropy-Benchmark `scripts/parquet_kirk.py`,
  `scripts/tiberius_client.py`, `scripts/quantbot_predictor.py` @
  `e2732baf0` — concrete production-tier predictors with
  `IS_PRODUCTION_KIRK = True`.
- `src/ulysses_jepa/interfaces.py` — `KirkClient` Protocol +
  `KirkOutput` dataclass + `KirkMode` enum.
- `src/ulysses_jepa/kirk_client.py` — `StubKirkClient`,
  `KirkPipelineClient`, `KirkSubprocessClient`.
- `eval/predictor_baseline.py` — Pipeline E integration.
- [DECISIONS.md D-012](../../DECISIONS.md) — short-form record of this
  decision.
- [HANDOFF.md "Cross-repo dependencies"](../../HANDOFF.md) — wiring
  instructions for the predictor repo dependency.
