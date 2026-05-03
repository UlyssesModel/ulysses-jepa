# Architectural decisions log

> One entry per decision. Each entry: **Decision · Why · Source.**
> Add new entries at the bottom; never edit historical ones — append a
> superseding entry instead.

## D-001 Project Array + Vector together; drop Scalar from LLM input
**Decided:** 2026-05-02

**What:** The projection adapter consumes Kirk's Array (N×N) and Vector (2N)
outputs, projects them as a sequence of (N+2) embedding vectors, and feeds
that to the LLM. The Scalar output is preserved as an anomaly gate but not
injected into the LLM.

**Why:** Array carries spatial/temporal structure; Vector carries
row+column marginal summaries. Together they form a multi-resolution
representation that maps naturally onto an LLM token sequence. Scalar is
one number — useful for routing/gating, too lossy to inject as a "thought."

**Source:** Kavara Data Science Guide, slides 13–14 (output types).
Conversation 2026-05-02.

## D-002 Token order: array rows first, then row-marginal summary, then col-marginal summary
**Decided:** 2026-05-02

**What:** Inside one Kirk window, the (N+2) projected vectors are
concatenated in this order: `[row_0, row_1, ..., row_{N-1}, row_summary,
col_summary]`.

**Why:** Gives the LLM a coherent reading order: the state itself, then a
summary of "what each row aggregates to," then "what each column aggregates
to." Aligned with how transformer attention naturally accumulates context
left-to-right.

**Source:** Architectural sketch 2026-05-02. Subject to validation when
training data is available.

## D-003 Real-valued projection by default
**Decided:** 2026-05-02

**What:** `AdapterConfig.use_complex` defaults to `False`. Real component
of complex inputs is used; imag is dropped.

**Why:** Per Ted's `kirk_data_description.md`, production Kirk uses
real-valued log-returns with imag uniformly zero. The complex128 container
is preserved as an optional second-channel research extension. No
information loss in dropping imag when imag=0. When the second-channel
extension is enabled, set `use_complex=True` and the adapter splits real/imag
and concatenates along the input dim — width doubles, info preserved.

**Source:** Uhura Confluence page (Ted-spec alignment table). Ted's
`kirk_data_description.md`.

## D-004 Two inference paths in v0.1 — embedding-injection cannot run on Ollama
**Decided:** 2026-05-02

**What:** Pipeline B (compressed-text via ScottyClient → Ollama) and
Pipeline C (embedding-injection via HF Transformers). Pipeline B is
production-shape; Pipeline C is the new build. Both run side by side in the
eval harness.

**Why:** Ollama's OpenAI-compatible `/v1/chat/completions` endpoint takes
tokens, not embeddings. Pure `inputs_embeds` injection requires direct model
access (vLLM `prompt_embeds` or HF Transformers). The cost-saving thesis
lives in Pipeline C; the production baseline is Pipeline B. The eval
harness measures both against the worst-case Pipeline A (raw text).

**Source:** Scotty repo README; OpenAI API spec. Conversation 2026-05-02.

## D-005 Frozen LLM, trainable adapter only — LLaVA-style distillation
**Decided:** 2026-05-02

**What:** During training the LLM weights are frozen (`requires_grad=False`).
Gradients flow through the LLM (no parameter updates) and into the
projection adapter, which updates against next-token cross-entropy on
teacher-generated narrations.

**Why:** Standard recipe in vision-language models. Adapter is small
(~5–50M parameters); training in hours on a single H100 is realistic.
Bypasses the cost and instability of co-training a 31B-parameter LLM.

**Source:** LLaVA / BLIP-2 papers; conversation 2026-05-02.

## D-006 Layer-2 Kirk preferred for projection (when available)
**Decided:** 2026-05-02

**What:** When Spencer's two-layer Kirk pipeline is wired (per the
"Aggregating array time series" pattern in the Data Science Guide slides
20–23), the projection adapter taps off the layer-2 outputs, not layer-1.

**Why:** Layer-2 is already a learned hierarchical representation —
Kirk has done its own attention-equivalent work. Projecting from layer-2
gives the LLM a richer starting point and sidesteps any criticism that
the projection is "just an embedding model."

**Source:** Kavara Data Science Guide, slides 20–23. Conversation
2026-05-02.

## D-007 Repo positioning: sibling of uhura / tiberius-openshift / scotty
**Decided:** 2026-05-02

**What:** ulysses-jepa is its own repo; consumes Uhura's tensor frames or
Kirk outputs via standard interfaces (Kafka or in-process); produces LLM
narrations or downstream completions.

**Why:** Matches the layered architecture in the Uhura Confluence page.
Each layer composes by Kafka topic / file handoff, no glue code. Allows
independent versioning, deployment, and testing.

**Source:** Uhura Confluence page (four-layer Kavara stack diagram).

## D-008 At N=32 the Kirk forward pass uses CPUBackend, not AMX
**Decided:** 2026-05-02

**What:** ts_sor_base-1's auto-routing logic puts N≤20 on FusedBackend (C/MKL,
zero-alloc, L1-resident), 21≤N≤500 on CPUBackend (NumPy/MKL AVX-512), and
N>500 on AMXBackend (PyTorch BF16 + oneDNN AMX). The kirk-pipeline Layer-2
dimension is fixed at N=32, which falls into the CPU bucket. **AMX does not
fire on the embedding-injection hot path.**

**Why this matters:** the Red Hat conference deck must be careful about the
AMX claim. Pipeline C's cost reduction comes from input-token compression
relative to Pipeline A and B — *not* from AMX-vs-GPU compute differences.
The AMX story is real but applies to Uhura's large-N sweeps (sp500/N=500+),
not to the Layer-2 embedding-injection path.

**Implication for the deck:** lead with input compression and the LLM-tier
shift as the ulysses-jepa thesis. Reference the AMX numbers only in the
context of Uhura's existing leaderboard, where AMX is real and measured.
Don't conflate them.

**Source:** `ts_sor_base-1 — Architecture & Backend Design` (PE/73531394).
`ts_sor_base-1 Implementation Status — 2026-04-19` (PE/79921153).
Conversation 2026-05-02.

## D-009 Use the `amx-stride2-32` venue policy on GNR+TDX
**Decided:** 2026-05-02

**What:** OpenShift Deployments set `OMP_NUM_THREADS=32`, `OMP_PLACES={0}:32:2`,
`OMP_PROC_BIND=close` in the container env. Mirrors the calibrated
`VENUE_POLICIES['gnr-tdx']` for both pinned and spread buckets after the
E2 calibration sweep.

**Why:** stride-2 32 threads beat every other tested config (lower-32,
upper-32, full-64-close, full-64-spread) at all N≥500 on Jarett's image
digest `4e9bc9e63f`. Caveat from the 2026-04-20 peer review: the win is
partly thread-count + turbo, not pure placement. The shipped policy is
empirically correct; the attribution is being re-validated via E6.

**Source:** `ts_sor_base-1 Implementation Status — 2026-04-19`,
"E2 results" section. Conversation 2026-05-02.

## D-010 JEPA framing as the external architectural narrative
**Decided:** 2026-05-02

**What:** External-facing description: "Ulysses is the JEPA encoder for
non-stationary, low-SNR time series; an LLM acts as the predictor consuming
the encoder's embedding." Drop "wormhole" terminology.

**Why:** Anchored in current literature (LeCun's JEPA work). Clean role
separation maps onto the Red Hat / Intel pitch (encoder on CPU, predictor
on GPU). Credibility with technical evaluators.

**Source:** Kavara × WMD doc; conversation 2026-05-02.

## D-011 Single-message stream batching: T = windows-per-batch
**Decided:** 2026-05-02

**What:** The streamer accumulates `windows_per_batch` (default 4) frames
from the input topic before running one inference call and emitting one
narration. No within-batch parallelism; cross-batch parallelism is via
multiple Deployment replicas reading different Kafka partitions.

**Why:** Mirrors the way Uhura's broadcaster emits one frame per cadence
tick. Batching at this layer keeps end-to-end latency bounded (4 windows ×
12s cadence = 48s narration latency for the 12s deployment) and keeps
the LLM call dense (more soft tokens per call → better GPU utilization).

**Source:** uhura streamer pattern; conversation 2026-05-02.

## D-012 EntropyPredictor and KirkClient are different abstractions
**Decided:** 2026-05-02

**What:** ulysses-jepa keeps `KirkClient` (and the `Stub` / `Pipeline` /
`Subprocess` implementations in `src/ulysses_jepa/kirk_client.py`) for the
projection-adapter input path. Forward-Entropy-Benchmark's
`EntropyPredictor` is a *separate* dependency, used only in `eval/`
(`Pipeline E`, see `eval/predictor_baseline.py`) as a forecast-quality
baseline alongside the HMM (`D_hmm_baseline`).

**Why:** an initial framing assumed these were duplicate abstractions
that could be consolidated. They aren't:

  - `KirkClient.infer(tensor: torch.Tensor) -> KirkOutput` — N×N tensor
    in, the full Layer-2 trio (Array, Vector, Scalar) out. This is the
    projection adapter's input contract per D-006.
  - `EntropyPredictor.predict(observations: np.ndarray, horizon: int) -> np.ndarray` —
    1-D observation prefix in, `(K,)` forward-entropy estimate out.
    Forward-entropy benchmark scoring.

Different inputs, different outputs, different jobs. Forcing them into
one abstraction would lose the projection-adapter's input contract
entirely (the predictor doesn't surface the layer-2 reconstruction the
adapter consumes). They compose by being used in different parts of the
pipeline, not by inheritance.

**D-007 is not superseded.** ulysses-jepa remains a sibling repo that
defines `KirkClient` locally because no upstream owns that abstraction
yet. Forward-Entropy-Benchmark owns the `EntropyPredictor` abstraction
because that's where it's authored and exercised.

**Future:** if Forward-Entropy-Benchmark grows a method on
`TiberiusKirkPredictor` (e.g. `infer_layer2(tensor) -> dict` with
reconstruction / marginals / entropy) that produces ulysses-jepa-compatible
output, revisit consolidation. ADR-002 captures the conceptual
distinction in more depth.

**Source:** conversation 2026-05-02. Forward-Entropy-Benchmark inspected
at commit `e2732baf07b55aad32fec635d4c4fef9759518e9` —
`EntropyPredictor.predict()` returns a `(K,)` entropy vector only; no
layer-2 reconstruction surface. Concrete implementations:
`KirkEntropyPredictor` (`IS_PRODUCTION_KIRK=False`), `ParquetKirkPredictor`
+ `TiberiusKirkPredictor` + `KirkEntropyFromParquetPredictor`
(`IS_PRODUCTION_KIRK=True`).
