"""Comparators for atomized WebPPL evaluation.

The dataset shape: each atom returns a single answer (value / distribution
/ samples / record). The harness dispatches on `answer_shape` to a
comparator here.
"""

from __future__ import annotations

import json
import math
import re


# Shape constants — answer_shape values stored on each atom.
SHAPE_VALUE = "value"
SHAPE_DISTRIBUTION = "distribution"
SHAPE_SAMPLES = "samples"

# Canonical __kind tag emitted by the executor's serializer for distributions.
KIND_DISTRIBUTION = "distribution"


def is_record_shape(shape) -> bool:
    return isinstance(shape, dict) and "record" in shape


# ---------------------------------------------------------------------------
# Code string comparison (baseline)
# ---------------------------------------------------------------------------

def normalize_code(code: str) -> str:
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", " ", code).strip()
    code = re.sub(r";\s*$", "", code)
    return code


def code_exact_match(generated: str, ground_truth: str) -> bool:
    return normalize_code(generated) == normalize_code(ground_truth)


def code_jaccard(generated: str, ground_truth: str) -> float:
    g = set(normalize_code(generated).split())
    t = set(normalize_code(ground_truth).split())
    if not g and not t:
        return 1.0
    if not g or not t:
        return 0.0
    return len(g & t) / len(g | t)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _normalize_dist(serialized) -> dict | None:
    """Pull {canonical_val_key: prob} from a canonical distribution object.

    Input shape (cross-PPL canonical, emitted by `__serialize` via WebPPL's
    built-in `serializeDist`):
        {"__kind": "distribution", "support": [v0, ...], "probs": [p0, ...]}

    Keys in the returned mapping are `json.dumps(val, sort_keys=True)` so
    object support points with reordered fields collide into the same bin.
    """
    if not isinstance(serialized, dict):
        return None
    if serialized.get("__kind") != KIND_DISTRIBUTION:
        return None
    support = serialized.get("support") or []
    probs = serialized.get("probs") or []
    if len(support) != len(probs):
        return None
    out: dict = {}
    for v, p in zip(support, probs):
        if p is None:
            continue
        canonical = json.dumps(v, sort_keys=True)
        out[canonical] = out.get(canonical, 0.0) + float(p)
    total = sum(out.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def _kl(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    keys = set(p) | set(q)
    return sum(
        p.get(k, 0.0) * math.log(p.get(k, 0.0) / max(q.get(k, epsilon), epsilon))
        for k in keys
        if p.get(k, 0.0) > 0
    )


def _tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    raw = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    return min(1.0, max(0.0, raw))


def kl_divergence(p_ser, q_ser) -> float | None:
    p, q = _normalize_dist(p_ser), _normalize_dist(q_ser)
    return None if p is None or q is None else _kl(p, q)


def total_variation(p_ser, q_ser) -> float | None:
    p, q = _normalize_dist(p_ser), _normalize_dist(q_ser)
    return None if p is None or q is None else _tv(p, q)


# ---------------------------------------------------------------------------
# Empirical samples
# ---------------------------------------------------------------------------

def _sample_key(s):
    if isinstance(s, (list, dict)):
        return json.dumps(s, sort_keys=True)
    return s


def empirical_tv(gen_samples: list, gt_samples: list) -> float | None:
    """TV between two empirical histograms over arbitrary sample values."""
    if not gen_samples or not gt_samples:
        return None
    gen_hist: dict = {}
    gt_hist: dict = {}
    for s in gen_samples:
        gen_hist[_sample_key(s)] = gen_hist.get(_sample_key(s), 0) + 1
    for s in gt_samples:
        gt_hist[_sample_key(s)] = gt_hist.get(_sample_key(s), 0) + 1
    n_g, n_t = len(gen_samples), len(gt_samples)
    keys = set(gen_hist) | set(gt_hist)
    raw = 0.5 * sum(
        abs(gen_hist.get(k, 0) / n_g - gt_hist.get(k, 0) / n_t) for k in keys
    )
    return min(1.0, max(0.0, raw))


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------

def value_match(gen, gt, rtol: float = 0.05):
    exact = gen == gt
    approx = exact
    if not exact:
        try:
            ref = abs(float(gt))
            if ref < 1e-12:
                approx = abs(float(gen)) < rtol
            else:
                approx = abs(float(gen) - float(gt)) / ref <= rtol
        except (TypeError, ValueError):
            if isinstance(gen, list) and isinstance(gt, list) and len(gen) == len(gt):
                try:
                    diffs = [abs(float(g) - float(t)) for g, t in zip(gen, gt)]
                    refs = [max(abs(float(t)), 1e-12) for t in gt]
                    approx = all(d / r <= rtol for d, r in zip(diffs, refs))
                except (TypeError, ValueError):
                    approx = False
            else:
                approx = False
    return {"exact_match": bool(exact), "approx_match": bool(approx)}


# ---------------------------------------------------------------------------
# Shape-dispatched comparison
# ---------------------------------------------------------------------------

def _cmp_distribution(gen, gt) -> dict:
    if not (isinstance(gen, dict) and isinstance(gt, dict)
            and gen.get("__kind") == KIND_DISTRIBUTION
            and gt.get("__kind") == KIND_DISTRIBUTION):
        return {"shape": SHAPE_DISTRIBUTION, "ok": False, "error": "not a distribution",
                "gen_kind": gen.get("__kind") if isinstance(gen, dict) else None,
                "gt_kind": gt.get("__kind") if isinstance(gt, dict) else None}
    p = _normalize_dist(gen)
    q = _normalize_dist(gt)
    if p is None or q is None:
        return {"shape": SHAPE_DISTRIBUTION, "ok": False, "error": "empty distribution"}
    return {"shape": SHAPE_DISTRIBUTION, "kl": _kl(p, q), "tv": _tv(p, q)}


def _distribution_to_samples(d: dict, n: int = 200) -> list | None:
    """Coerce a serialized distribution into a sample list for histogram
    comparison with samples-shape atoms. Returns ~n samples drawn from
    the distribution's support according to its probs.
    """
    if not isinstance(d, dict) or d.get("__kind") != KIND_DISTRIBUTION:
        return None
    support = d.get("support") or []
    probs = d.get("probs") or []
    if not support or not probs:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    norm = [p / total for p in probs]
    # Deterministic expansion: each value gets round(p * n) copies.
    out = []
    for v, p in zip(support, norm):
        out.extend([v] * max(1, round(p * n)))
    return out[:max(n, len(support))]


def _cmp_samples(gen, gt) -> dict:
    # Allow comparison when one side produced a distribution and the
    # other a sample list — coerce by drawing from the distribution.
    if isinstance(gen, dict) and gen.get("__kind") == KIND_DISTRIBUTION:
        gen = _distribution_to_samples(gen) or gen
    if isinstance(gt, dict) and gt.get("__kind") == KIND_DISTRIBUTION:
        gt = _distribution_to_samples(gt) or gt
    if not isinstance(gen, list) or not isinstance(gt, list):
        return {"shape": SHAPE_SAMPLES, "ok": False, "error": "samples must be a list",
                "gen_type": type(gen).__name__, "gt_type": type(gt).__name__}
    return {"shape": SHAPE_SAMPLES, "n_gen": len(gen), "n_gt": len(gt),
            "tv": empirical_tv(gen, gt)}


def _cmp_value(gen, gt) -> dict:
    return {"shape": SHAPE_VALUE, **value_match(gen, gt)}


_LEAF_COMPARATORS = {
    SHAPE_VALUE: _cmp_value,
    SHAPE_DISTRIBUTION: _cmp_distribution,
    SHAPE_SAMPLES: _cmp_samples,
}


def compare_by_shape(gen, gt, shape) -> dict:
    """Recursively compare two answers under a given answer_shape."""
    if is_record_shape(shape):
        if not isinstance(gen, dict) or not isinstance(gt, dict):
            return {"shape": "record", "ok": False, "error": "non-record answer"}
        return {"shape": "record", "fields": {
            fname: compare_by_shape(gen.get(fname), gt.get(fname), fshape)
            for fname, fshape in shape["record"].items()
        }}
    cmp_fn = _LEAF_COMPARATORS.get(shape)
    if cmp_fn is None:
        return {"shape": str(shape), "ok": False, "error": "unknown shape"}
    return cmp_fn(gen, gt)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def collect_metrics(comparison: dict) -> dict:
    """Walk a comparison tree; return a flat dict like {'rain.tv': 0.05, ...}."""
    out: dict = {}

    def _walk(node, prefix: str):
        if not isinstance(node, dict):
            return
        shape = node.get("shape")
        if shape == "record":
            for fname, fnode in (node.get("fields") or {}).items():
                _walk(fnode, f"{prefix}{fname}.")
            return
        if shape == SHAPE_DISTRIBUTION:
            if node.get("kl") is not None:
                out[prefix + "kl"] = node["kl"]
            if node.get("tv") is not None:
                out[prefix + "tv"] = node["tv"]
        elif shape == SHAPE_SAMPLES:
            if node.get("tv") is not None:
                out[prefix + "tv"] = node["tv"]
        elif shape == SHAPE_VALUE:
            out[prefix + "exact"] = 1.0 if node.get("exact_match") else 0.0
            out[prefix + "approx"] = 1.0 if node.get("approx_match") else 0.0

    _walk(comparison, "")
    return out


def aggregate_metrics(metrics_per_atom: list[dict]) -> dict:
    """Bucket metric values across atoms by suffix (kl/tv/exact)."""
    kls, tvs, exacts = [], [], []
    for m in metrics_per_atom:
        for k, v in (m or {}).items():
            if v is None:
                continue
            if k.endswith("kl"):
                kls.append(v)
            elif k.endswith("tv"):
                tvs.append(v)
            elif k.endswith("exact"):
                exacts.append(v)

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "mean_kl": _mean(kls), "n_kl_metrics": len(kls),
        "mean_tv": _mean(tvs), "n_tv_metrics": len(tvs),
        "mean_value_exact": _mean(exacts), "n_exact_metrics": len(exacts),
    }
