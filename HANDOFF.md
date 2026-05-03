# Handoff — what's stubbed, what's real, what's next

> **Status:** v0.1 build complete. 63 files, ~7K lines Python, 105 test
> cases passing locally. All Phase-0 questions resolved. Solo execution.

## Cross-repo / cross-host topology

Where work happens and which box owns which role:

| Host | Role |
| --- | --- |
| **IvorHQ** (WSL2 Ubuntu) | Primary dev box. `ulysses-jepa` working tree alongside its `Forward-Entropy-Benchmark` and `STAC-ML-Markets-Inference-Models` siblings. CPU-only test runs land here. |
| **scotty-gpu** (GCP us-central1-a, Tailscale `100.120.101.79`) | Gemma serving box. Ollama on `scotty-gpu:11434` hosts Gemma 4 31B + 26B and Gemma 3 12B. Teacher LLM for distillation; live target for `make pin-gemma`. |
| **ny5ulysses01** | Production TDX host. Owns the **Kirk model IP vault** (TDX domain on port 2250). This duty does **not** live on scotty-gpu — see `ts_sor_base-1` for the canonical wiring. |
| **tdx-amx-node-octo** | Production benchmark + integration-test target. Where `tests/test_kirk_pipeline_integration.py` actually fires (and where the `_run_layer2` schema check has teeth). |

## What's real and runnable today

| Path | How to run | What it proves |
| --- | --- | --- |
| Unit tests (105 cases, CPU only) | `make test` | Adapter, kirk-client, pipelines, eval, distill, train, serve, prompts, logging, sweep, HMM all sound |
| Plumbing-validation demo | `make demo` | Distill → train → eval → sweep end-to-end against Stub Kirk + tiny LLM, emits `reports/demo.html` |
| Local Docker dev stack | `make compose-up` | Ollama + Redpanda + predictor running locally; smoke-test all 3 routes |
| Production OpenShift deploy | `make openshift-apply` | Namespace + Strimzi cluster + KServe InferenceService + Streamer + Grafana dashboard, dependency-ordered |
| Post-deploy health check | `make smoke-test` | All 3 V2 routes respond 200 with non-empty narrations |

## What's stubbed and needs filling in

### 1. `KirkPipelineClient._run_layer2` output schema

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

### 2. Real distillation dataset

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

### 3. KServe predictor entry point — DONE

`scripts/serve_kserve.py` is a real FastAPI server with three routes
(Pipelines A/B/C) and a `/metrics` endpoint. 16 test cases covering happy
paths and error paths.

### 4. Streaming Kafka consumer — DONE

`scripts/stream_consumer.py` consumes Uhura tensor frames, runs the
pipeline, publishes narrations to a downstream topic. Strimzi mTLS,
cooperative-sticky rebalancing, idempotent producer, structured logging,
Prometheus metrics on a separate port.

### 5. Container image build — TODO

`quay.io/kavara/ulysses-jepa:v0.1` is referenced in the OpenShift manifests
but not built yet. Build with:

```bash
make image && make push
```

Requires podman + quay.io credentials.

## Critical-path order of operations

1. ~~**Pin Gemma 4 31B hidden dim**~~ — **DONE.** Confirmed `5376` against
   live Ollama on `scotty-gpu:11434` (commit `43a7da4`); both YAMLs match.
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

## Cross-repo dependencies

`ulysses-jepa` runs against a single sibling-repo dependency that doesn't
ship as a package yet:

### Forward-Entropy-Benchmark — needed for tests + Pipeline E

The `EntropyPredictor` abstraction (and its concrete implementations:
`KirkEntropyPredictor`, `ParquetKirkPredictor`, `TiberiusKirkPredictor`,
`KirkEntropyFromParquetPredictor`) lives in
[`UlyssesModel/Forward-Entropy-Benchmark`](https://github.com/UlyssesModel/Forward-Entropy-Benchmark).
It is consumed in `eval/predictor_baseline.py` (Pipeline E) and is
**different from `KirkClient`** — see `DECISIONS.md` D-012 and
`docs/adr/0002-kirk-client-vs-entropy-predictor.md`.

The predictor repo has no `pyproject.toml`, so we shim its `scripts/`
directory onto `sys.path`:

  - **Local pytest:** `conftest.py` at the repo root inserts
    `~/Forward-Entropy-Benchmark/scripts`. Clone the sibling repo
    next to `ulysses-jepa/`:

    ```bash
    git clone git@github.com:UlyssesModel/Forward-Entropy-Benchmark.git ~/Forward-Entropy-Benchmark
    git -C ~/Forward-Entropy-Benchmark checkout e2732baf07b55aad32fec635d4c4fef9759518e9
    ```

  - **Make targets:** `make test`, `make test-cov`, `make demo` all
    set `PYTHONPATH=src:.:$(FE_BENCH)`. Override `FE_BENCH` if your
    clone lives elsewhere.

  - **CI:** `.github/workflows/ci.yml` clones the predictor repo at the
    pinned commit and adds it to `PYTHONPATH`.

Tests that need the predictor are skip-guarded — running pytest without
the sibling repo present skips the Pipeline E tests cleanly rather than
erroring. The other 120 tests are predictor-independent.

**Transitive dep, FYI:** `Forward-Entropy-Benchmark/scripts/entropy_predictor.py`
top-level-imports `hankel_adapter` and `ulysses_predictor` from a third
sibling repo (`UlyssesModel/STAC-ML-Markets-Inference-Models`). Until
that import is made lazy upstream, exercising Pipeline E end-to-end
also requires:

```bash
git clone git@github.com:UlyssesModel/STAC-ML-Markets-Inference-Models.git \
    ~/STAC-ML-Markets-Inference-Models
```

The predictor's `_MIRROR_CANDIDATES` block resolves the import once
that clone is in place. Without it, our skip-guards fire and Pipeline E
silently skips — which is the desired CI behavior on hosts where the
STAC-ML mirror isn't provisioned.

**Cleanup path:** when Forward-Entropy-Benchmark grows a `pyproject.toml`,
drop `conftest.py`, drop `FE_BENCH` from the Makefile, drop the
`actions/checkout` step from CI, and add the package as a normal
dependency in `pyproject.toml`'s dev extras.

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
