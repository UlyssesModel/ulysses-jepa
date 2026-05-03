"""Pytest conftest: ensure Forward-Entropy-Benchmark is on sys.path.

The predictor abstraction lives in a sibling repo without packaging.
Until that repo grows a pyproject.toml, we shim its scripts dir onto
sys.path so the eval harness can import EntropyPredictor + concrete
implementations.

Cleanup path: when Forward-Entropy-Benchmark exposes a proper package,
drop this shim and add the package as a normal dependency in pyproject.toml.
"""
import os
import sys

_FE_BENCH = os.path.expanduser("~/Forward-Entropy-Benchmark/scripts")
if os.path.isdir(_FE_BENCH) and _FE_BENCH not in sys.path:
    sys.path.insert(0, _FE_BENCH)
