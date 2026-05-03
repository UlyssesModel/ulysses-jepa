# ADR-003 — Narrow First Experiment for Kirk Embedding Value

**Status:** Accepted
**Date:** 2026-05-03
**Author:** John Edge
**Related:** [D-012](../../DECISIONS.md), [ADR-002](0002-kirk-client-vs-entropy-predictor.md)

## Context

We have a working eval harness (Pipelines A–E) running real Gemma against
synthetic-but-real-cost numbers, with `rouge_l` and `regime_correct`
metrics already in `eval/metrics.py::EvalRecord`. The central hypothesis
is that Kirk latent embeddings (Pipeline C) deliver meaningful quality
advantages over Kirk-compressed text (Pipeline B) — especially on
non-stationary / regime-shifting data — while preserving cost wins.

The broader ambition is a phased program: factorial DOE, cross-domain
generalization, production stress, full experimentation platform (Ray,
MLflow, statistical analysis). Building that platform before we have one
clean falsifiable result on the core claim risks premature optimization
and weeks of wasted effort.

D-012 and ADR-002 establish that KirkClient and EntropyPredictor are
parallel abstractions; this ADR is downstream of that distinction.

## Decision

Run a **minimal 2 × 3 × 5 experiment** using the *existing harness*:

- **Data regime** (2 levels): stationary vs strong regime shift
- **Target model tier** (3 levels): frontier / mid / small (specifically
  Gemma 4 31B / Gemma 4 26B / Gemma 3 12B as already pulled on scotty-gpu)
- **Replications** (5 per cell): 30 cells total, 4 pipeline records per
  cell (B, C, D, E), 120 EvalRecord rows total

The teacher LLM that produces gold narrations during distillation is
fixed (Gemma 4 31B); it is orthogonal to the target-model factor and
not varied here.

**Strong regime shift** is defined as a 3σ jump in the underlying
state-space mean at the midpoint of each stream, with the
`gold_regime` metadata field flipping from one HMM state name to
another at the breakpoint. Mean-shift only — vol-regime and
correlation-breakdown shifts deferred to Phase 2.

**Primary success criterion:** Pipeline C shows statistically
significant improvement over Pipeline B on the strong-shift cells, on
both `rouge_l` (vs the gemma4:31b teacher narrations) and
`regime_correct` (string-match of HMM state names against
`gold_regime`), without material regression in cost-per-query.

If successful → green-light broader Phase 1 factorial + experimentation
platform investment per the program proposal.

If inconclusive or negative → pivot the projection adapter design or
re-examine whether embedding injection adds anything over compressed
text for this domain.

## Rationale

- Fastest path to a defensible number on the load-bearing claim
- Uses existing infrastructure (no new orchestration code)
- Limits blast radius if the embedding path underperforms
- Preserves the bigger vision without building toward it yet
- Includes D and E baselines for free (already in the harness),
  giving a richer comparison than B-vs-C alone without expanding
  the factor design

## Deferred — revisited only on positive result

- Full factorial DOE with interaction analysis (5 factors)
- Experiment orchestrator + Ray Tune + MLflow/W&B integration
- Cross-domain generalization suite (robotics, sensor, code-evolution)
- Production stress + long-horizon agentic loops
- Vol-regime / correlation-breakdown shift types
- LLM-as-judge metric beyond ROUGE-L
- Human eval

## Consequences

- **Positive result**: justifies deeper investment, customer-facing
  artifacts, and the broader experimental program.
- **Negative result**: early signal to refine the projection adapter
  or adjust expectations on embedding injection. Pipeline B remains
  the production winner, deck pivots to "compressed-text + air-gap"
  rather than "embedding-injection + air-gap."

## Open in implementation

Two mechanics decisions deferred to the regime-generator PR rather than
relitigated here:

- **Sweep axis fit**: whether to extend `SweepGrid` with `regime` /
  `tier` / `seed` axes, or run the existing sweep multiple times from a
  Makefile target with per-(regime, tier) `--distilled` and
  `--llm-config` flags. Conservative default is the latter — matches
  "uses existing infrastructure (no new orchestration code)."
- **Replication granularity**: whether one replication = one fresh
  seeded stream, or one fresh seeded eval set of N items. Either is
  acceptable; locked when the generator code is written.

Both are textually scoped, so they can't sneak in as bigger changes
later — the ADR pins the universe of acceptable answers.

## Implementation plan

1. Land this ADR on `main` (one commit, no code).
2. Add `experiments/regime_generator.py` — single module producing
   `DistillationItem` streams in stationary and 3σ-shift variants.
3. Add an experiment YAML at `configs/experiment_adr003.yaml`
   parameterizing the 2 × 3 sweep + 5 replications.
4. Run the sweep against scotty-gpu's three Gemma tiers.
5. Statistical analysis (paired t-test or Wilcoxon on B vs C per cell).
6. Publish results as ADR-004 (the experiment outcome) regardless of
   sign — positive or negative result both deserve a permanent record.
