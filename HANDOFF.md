# Handoff — what's stubbed, what's real, what's next

> **Status:** v0.1 build complete. 63 files, ~7K lines Python, 105 test
> cases passing locally. All Phase-0 questions resolved. Solo execution.

## What's real and runnable today

| Path | How to run | What it proves |
| --- | --- | --- |
| Unit tests (105 cases, CPU only) | `make test` | Adapter, kirk-client, pipelines, eval, distill, train, serve, prompts, logging, sweep, HMM all sound |
| Plumbing-validation demo | `make demo` | Distill → train → eval → sweep end-to-end against Stub Kirk + tiny LLM, emits `reports/demo.html` |
| Local Docker dev stack | `make compose-up` | Ollama + Redpanda + predictor running locally; smoke-test all 3 routes |
| Production OpenShift deploy | `make openshift-apply` | Namespace + Strimzi cluster + KServe InferenceService + Streamer + Grafana dashboard, dependency-ordered |
| Post-deploy health check | `make smoke-test` | All 3 V2 routes respond 200 with non-empty narrations |

## What's stubbed and needs filling in

### 1. Gemma 4 31B `hidden_size`

`5376` placeholder in `configs/adapter_default.yaml` and `configs/llm_gemma4.yaml`.
Auto-pin once Ollama has gemma4:31b pulled:

```bash
make pin-gemma
```

The script queries Ollama's `/api/show` endpoint, falls back to HuggingFace
config lookup if needed, updates both YAMLs in place, leaves `.bak` files.

### 2. `KirkPipelineClient._run_layer2` output schema

The integration test at `tests/test_kirk_pipeline_integration.py` is the
load-bearing safety net here. It runs only when `kirk_pipeline` is importable
in the test process — skipped on dev hosts, fires automatically on
tdx-amx-node-octo and IvorHQ.

When the wheel lands and the test runs, the most likely failure is the dict
shape returned by `KirkModelInterface.forward()` in `active_inference` mode.
Current code assumes `{"reconstruction": (n,n), "marginals": (2n,), "entropy": ()}`.
If the real return is a tuple, dataclass, or different keys, the test points
at exactly which line of `KirkPipelineClient._run_layer2` to fix. ~1 hour
of work.

### 3. Real distillation dataset

The training path runs end-to-end against synthetic data and a teacher LLM
(Scotty/Ollama works as the teacher in dev). For real numbers:

```bash
# Pull a trading day from the Quantbot reference set
python scripts/distill_teacher.py \
    --output data/distilled_train.pt \
    --uhura-frames-glob "data/uhura/2024-09-03/*.npz" \
    --teacher-base-url https://api.anthropic.com/v1 \
    --teacher-model claude-opus-4-6

# Tag with regime labels (gives the eval harness regime_correct values)
python scripts/label_regimes.py \
    --distilled data/distilled_train.pt \
    --out data/distilled_train_labeled.pt
```

Cost estimate: ~1K labeled streams at Opus pricing is ~$5–20.

### 4. KServe predictor entry point — DONE

`scripts/serve_kserve.py` is a real FastAPI server with three routes
(Pipelines A/B/C) and a `/metrics` endpoint. 16 test cases covering happy
paths and error paths.

### 5. Streaming Kafka consumer — DONE

`scripts/stream_consumer.py` consumes Uhura tensor frames, runs the
pipeline, publishes narrations to a downstream topic. Strimzi mTLS,
cooperative-sticky rebalancing, idempotent producer, structured logging,
Prometheus metrics on a separate port.

### 6. Container image build — TODO

`quay.io/kavara/ulysses-jepa:v0.1` is referenced in the OpenShift manifests
but not built yet. Build with:

```bash
make image && make push
```

Requires podman + quay.io credentials.

## Critical-path order of operations

1. **Pin Gemma 4 31B hidden dim** — `make pin-gemma`. Five-minute job.
2. **Validate the adapter trains end-to-end on stub data** —
   `make distill-stub && make train-stub && make eval-stub`. Exercises the
   gradient path through frozen LLM. Use a smaller LLM for first run if
   Gemma 4 31B isn't local yet.
3. **Wire `KirkPipelineClient` against the real wheel** on
   `tdx-amx-node-octo` or `IvorHQ`. First call surfaces any layer-2 output
   schema mismatch — fix in `_run_layer2`.
4. **Pull a real Uhura frame batch + distill against a frontier teacher** —
   1K labeled streams via `scripts/distill_teacher.py`.
5. **First real adapter training run** — `python scripts/train.py` with the
   real config. Validate on dev set via `eval/runner.py`. Target ROUGE-L
   within 0.15 of Pipeline B baseline at <50% of B's input token cost.
6. **Build + push the container image, deploy to OpenShift** —
   `make image && make push && make openshift-apply`. The Strimzi cluster
   manifest creates Kafka cluster from zero; smoke-test confirms the routes
   are live.

## The Phase-4 decision tree

After step 5 produces real numbers, three possible outcomes:

- **Pipeline C beats B by ≥3× cost at parity quality** — ship Pipeline C as
  the new default. Conference deck headline: cost win + IP-protection win.
- **Pipeline C beats B by 1.5–3×** — ship Pipeline C for the IP-protection
  story (the air-gappable form factor); modest cost headline.
- **Pipeline C ties or loses to B on cost-per-quality** — Pipeline C still
  ships if the customer's threat model demands the no-tokenization channel,
  otherwise harden Pipeline B for production. Conference deck pivots to
  "Pipeline B beats GPU baseline by 40× and runs in confidential compute."

ADR-001 has the full argument. Pipeline C is not the only reason this
project ships — Pipeline B already pays the bills.

## Things that should NOT change without re-deciding

Pinned by the architecture; updates need a new entry in `DECISIONS.md` and
re-training:

- `KirkOutput` contract (layer2_input, layer2_reconstruction, layer2_marginals, entropy)
- Adapter token order (rows, row-summary, col-summary)
- `n=32` Layer-2 dimension (kirk-pipeline-defined; not a free parameter)
- Real-valued default (`use_complex=False`)
- Frozen LLM + token-mask + CE loss form (LLaVA-style distillation)
- Three-pipelines-in-one-server (A/B/C all hit the same predictor)
- Versioned prompts in `prompts.py` (changing prompts requires version bump
  and re-train, not in-place edit)
