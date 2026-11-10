"""
Measurement helpers
-------------------
Small, dependency-free primitives shared by `evaluate.py` and
`benchmarks/benchmark_model.py`, so latency/size numbers are computed the same
way everywhere and can be unit-tested without torch.

Keeping these pure (no torch import) is deliberate: `tests/test_metrics.py`
runs in CI on a plain Python image.
"""

import os
import platform
import sys
from typing import Dict, Iterable, List, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """
    Linear-interpolated percentile (same definition as numpy's default).

    Args:
        values: Sample values (unsorted is fine).
        p: Percentile in [0, 100].

    Returns:
        The interpolated value at percentile ``p``.
    """
    if not values:
        raise ValueError("percentile() needs at least one value")
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be within [0, 100], got {p}")

    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low

    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_latencies(latencies_ms: Iterable[float]) -> Dict[str, float]:
    """
    Turn a list of per-request latencies into a reportable summary.

    p50/p95 are reported alongside the mean because a single slow first request
    skews the mean and hides the steady-state cost.
    """
    values: List[float] = [float(v) for v in latencies_ms]
    if not values:
        raise ValueError("summarize_latencies() needs at least one measurement")

    return {
        "runs": len(values),
        "mean_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def tokens_per_second(total_tokens: int, total_seconds: float) -> float:
    """Throughput of generated tokens (decode speed, excluding prompt prefill time)."""
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    return round(total_tokens / total_seconds, 2)


def folder_size_mb(folder_path: str) -> float:
    """Total on-disk size of a directory tree in MB (used for quantization comparisons)."""
    total_bytes = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            if os.path.exists(file_path):
                total_bytes += os.path.getsize(file_path)
    return round(total_bytes / (1024 * 1024), 2)


def compression_ratio_mb(reference_mb: float, candidate_mb: float) -> float:
    """Percentage size reduction of ``candidate`` against ``reference`` (positive = smaller)."""
    if reference_mb <= 0:
        raise ValueError("reference_mb must be positive")
    return round((reference_mb - candidate_mb) / reference_mb * 100, 2)


def peak_rss_mb() -> Optional[float]:
    """
    Resident memory of this process in MB, or None when psutil is unavailable.

    On CPU runs this is the practical proxy for "what does serving this model
    cost", which is the number that decides whether it fits the box.
    """
    try:
        import psutil  # imported lazily so the core pipeline has no extra dependency
    except ImportError:
        return None

    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def environment_fingerprint(device: Optional[str] = None, dtype: Optional[str] = None) -> Dict[str, object]:
    """
    Record the environment a measurement came from.

    Numbers without this are not reproducible - the same model can differ by
    several times in throughput between CPU and GPU, or between dtypes.
    """
    fingerprint: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": device or "unknown",
        "dtype": dtype,
    }

    try:
        import torch

        fingerprint["torch"] = torch.__version__
        fingerprint["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            fingerprint["gpu"] = torch.cuda.get_device_name(0)
        fingerprint["torch_threads"] = torch.get_num_threads()
    except ImportError:
        fingerprint["torch"] = None

    try:
        import transformers

        fingerprint["transformers"] = transformers.__version__
    except ImportError:
        fingerprint["transformers"] = None

    return fingerprint
