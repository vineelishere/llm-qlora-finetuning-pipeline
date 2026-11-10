"""
Unit tests for the pure measurement helpers. No torch, no model download.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from metrics import (  # noqa: E402
    compression_ratio_mb,
    folder_size_mb,
    percentile,
    summarize_latencies,
    tokens_per_second,
)


def test_percentile_matches_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0
    assert percentile(values, 50) == 30.0
    # rank = 0.95 * 4 = 3.8 -> between 40 and 50
    assert percentile(values, 95) == pytest.approx(48.0)


def test_percentile_handles_single_value_and_unsorted_input():
    assert percentile([7.0], 95) == 7.0
    assert percentile([30.0, 10.0, 20.0], 50) == 20.0


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_latencies_reports_spread_not_just_mean():
    summary = summarize_latencies([100.0, 120.0, 110.0, 900.0])

    assert summary["runs"] == 4
    assert summary["mean_ms"] == 307.5
    assert summary["min_ms"] == 100.0
    assert summary["max_ms"] == 900.0
    # The outlier must not move p50 much: p50 stays in the 110-120 band
    assert 100.0 < summary["p50_ms"] < 200.0
    assert summary["p95_ms"] > summary["p50_ms"]


def test_tokens_per_second():
    assert tokens_per_second(128, 2.0) == 64.0
    with pytest.raises(ValueError):
        tokens_per_second(10, 0)


def test_compression_ratio():
    assert compression_ratio_mb(2.96, 0.94) == pytest.approx(68.24, abs=0.01)
    with pytest.raises(ValueError):
        compression_ratio_mb(0, 1)


def test_folder_size_mb_sums_nested_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 1024 * 1024)          # 1 MB
    (tmp_path / "nested" / "b.bin").write_bytes(b"x" * 512 * 1024)  # 0.5 MB

    assert folder_size_mb(str(tmp_path)) == pytest.approx(1.5, abs=0.01)
