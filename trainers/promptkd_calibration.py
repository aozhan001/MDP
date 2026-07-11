import json
import math
import os
import random
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


CALIBRATION_CACHE_VERSION = "v1"
CALIBRATION_TIE_EPS = 1e-8


@dataclass(frozen=True)
class CalibrationFallback:
    """Configured parameters used when automatic calibration cannot select one."""

    mtp_alpha: float
    dvp_alpha: float
    prior_gamma: float


def parse_batch_calibration(batch):
    """Return only unlabeled images from a training batch."""

    return batch["img"]


def stable_hash(value, length=12):
    """Build a short deterministic hash for cache metadata."""

    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _warn(warn_fn, message):
    if warn_fn is not None:
        warn_fn(message)


def _as_float_list(values):
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(v) for v in values]


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_alpha(value):
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


def _valid_gamma(value):
    return _is_finite_number(value) and float(value) >= 0.0


def _normalize_one_candidate_list(values, fallback, name, is_valid, warn_fn=None):
    try:
        raw_values = _as_float_list(values)
    except (TypeError, ValueError) as exc:
        raw_values = []
        _warn(warn_fn, f"{name} candidates could not be parsed ({exc}); using fallback {fallback}.")

    valid_values = []
    invalid_values = []
    for value in raw_values:
        if is_valid(value):
            valid_values.append(float(value))
        else:
            invalid_values.append(value)

    if invalid_values or not valid_values:
        _warn(
            warn_fn,
            f"{name} candidates are empty or invalid ({invalid_values}); using fallback {fallback}.",
        )
        if not is_valid(fallback):
            raise ValueError(f"{name} fallback value is invalid: {fallback}")
        valid_values = [float(fallback)]

    return sorted(set(valid_values))


def resolve_candidate_space(
    mtp_candidates,
    dvp_candidates,
    prior_candidates,
    fallback,
    mtp_enabled=True,
    dvp_enabled=True,
    prior_enabled=True,
    warn_fn=None,
):
    """Validate, de-duplicate, sort, and gate the calibration search candidates."""

    if not isinstance(fallback, CalibrationFallback):
        fallback = CalibrationFallback(*fallback)

    if mtp_enabled:
        mtp = _normalize_one_candidate_list(
            mtp_candidates,
            fallback.mtp_alpha,
            "MTP_ALPHA",
            _valid_alpha,
            warn_fn,
        )
    else:
        mtp = [0.0]

    if dvp_enabled:
        dvp = _normalize_one_candidate_list(
            dvp_candidates,
            fallback.dvp_alpha,
            "DVP_ALPHA",
            _valid_alpha,
            warn_fn,
        )
    else:
        dvp = [0.0]

    if prior_enabled:
        prior = _normalize_one_candidate_list(
            prior_candidates,
            fallback.prior_gamma,
            "PRIOR_GAMMA",
            _valid_gamma,
            warn_fn,
        )
    else:
        prior = [0.0]

    return {"mtp_alpha": mtp, "dvp_alpha": dvp, "prior_gamma": prior}


def normalize_probability(prob, eps):
    """Clamp and normalize a probability tensor over classes."""

    prob = prob.float().clamp_min(float(eps))
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def fuse_dvp_text_features(base_text_features, visual_prototypes, alpha, eps=1e-8):
    """Fuse text features and visual prototypes with a fixed DVP alpha.

    Args:
        base_text_features: Tensor with shape [C, D].
        visual_prototypes: Tensor with shape [C, D].
    """

    alpha = float(alpha)
    fused = (1.0 - alpha) * base_text_features.float() + alpha * visual_prototypes.float()
    return F.normalize(fused, dim=1, eps=float(eps))


def fuse_mtp_text_features(base_text_features, mtp_text_features, alpha, eps=1e-8):
    """Fuse DVP text features and MTP text features with a fixed MTP alpha."""

    alpha = float(alpha)
    if mtp_text_features is None or alpha == 0.0:
        return base_text_features.float()
    fused = (1.0 - alpha) * base_text_features.float() + alpha * mtp_text_features.float()
    return F.normalize(fused, dim=1, eps=float(eps))


def estimate_teacher_prior(logits, temperature=1.0, eps=1e-8):
    """Estimate the class prior from uncorrected teacher logits."""

    temperature = max(float(temperature), float(eps))
    probs = torch.softmax(logits.float() / temperature, dim=1)
    prior = probs.mean(dim=0)
    return normalize_probability(prior, eps)


def apply_prior_correction(logits, prior, gamma, eps=1e-8):
    """Apply CPC-KD prior correction to logits."""

    gamma = float(gamma)
    if gamma == 0.0:
        return logits.float()
    prior_log = torch.log(prior.float().clamp_min(float(eps))).unsqueeze(0)
    return logits.float() - gamma * prior_log


def get_scoring_class_slice(modal, use_base_classes_only, num_classes):
    """Return the class slice used for unlabeled calibration scoring."""

    if modal == "base2novel" and use_base_classes_only:
        end = int(math.ceil(int(num_classes) / 2))
        return slice(0, end), end
    return slice(0, int(num_classes)), int(num_classes)


def score_unlabeled_predictions(
    candidate_logits,
    baseline_logits,
    anchor_weight=0.20,
    shift_weight=0.05,
    eps=1e-8,
):
    """Score unlabeled predictions using normalized MI minus anchor penalties."""

    eps = float(eps)
    candidate_logits = candidate_logits.float()
    baseline_logits = baseline_logits.float()

    if (
        candidate_logits.ndim != 2
        or baseline_logits.ndim != 2
        or candidate_logits.shape != baseline_logits.shape
        or candidate_logits.shape[1] <= 0
        or not torch.isfinite(candidate_logits).all()
        or not torch.isfinite(baseline_logits).all()
    ):
        return _invalid_metrics()

    c = candidate_logits.shape[1]
    log_c = math.log(c) if c > 1 else 1.0

    candidate_probs = normalize_probability(torch.softmax(candidate_logits, dim=1), eps)
    baseline_probs = normalize_probability(torch.softmax(baseline_logits, dim=1), eps)

    h_cond = -(candidate_probs * torch.log(candidate_probs.clamp_min(eps))).sum(dim=1).mean()
    p_bar = normalize_probability(candidate_probs.mean(dim=0), eps)
    h_marg = -(p_bar * torch.log(p_bar.clamp_min(eps))).sum()
    mi_norm = (h_marg - h_cond) / log_c
    h_cond_norm = h_cond / log_c
    h_marg_norm = h_marg / log_c

    mixed = normalize_probability(0.5 * (candidate_probs + baseline_probs), eps)
    kl_candidate = (
        candidate_probs
        * (torch.log(candidate_probs.clamp_min(eps)) - torch.log(mixed.clamp_min(eps)))
    ).sum(dim=1)
    kl_baseline = (
        baseline_probs
        * (torch.log(baseline_probs.clamp_min(eps)) - torch.log(mixed.clamp_min(eps)))
    ).sum(dim=1)
    js_norm = (0.5 * kl_candidate + 0.5 * kl_baseline).mean() / math.log(2.0)

    shift = (candidate_logits - baseline_logits).abs().mean()
    baseline_scale = baseline_logits.abs().mean() + eps
    shift_norm = shift / baseline_scale

    score = mi_norm - float(anchor_weight) * js_norm - float(shift_weight) * shift_norm

    values = {
        "score": score,
        "mi_norm": mi_norm,
        "h_cond_norm": h_cond_norm,
        "h_marg_norm": h_marg_norm,
        "js_norm": js_norm,
        "shift_norm": shift_norm,
    }
    if not all(torch.isfinite(v).item() for v in values.values()):
        return _invalid_metrics()

    return {key: float(value.item()) for key, value in values.items()}


def _invalid_metrics():
    return {
        "score": float("-inf"),
        "mi_norm": float("-inf"),
        "h_cond_norm": float("inf"),
        "h_marg_norm": float("-inf"),
        "js_norm": float("inf"),
        "shift_norm": float("inf"),
    }


def _tie_key(result, fallback):
    return (
        abs(result["mtp_alpha"] - fallback.mtp_alpha)
        + abs(result["dvp_alpha"] - fallback.dvp_alpha)
        + abs(result["prior_gamma"] - fallback.prior_gamma),
        result["mtp_alpha"] + result["dvp_alpha"] + result["prior_gamma"],
        result["mtp_alpha"],
        result["dvp_alpha"],
        result["prior_gamma"],
    )


def _result_is_better(candidate, best, fallback, tie_eps=CALIBRATION_TIE_EPS):
    if best is None:
        return True
    if candidate["score"] > best["score"] + tie_eps:
        return True
    if abs(candidate["score"] - best["score"]) <= tie_eps:
        return _tie_key(candidate, fallback) < _tie_key(best, fallback)
    return False


@torch.no_grad()
def search_calibration_parameters(
    image_features,
    raw_text_features,
    visual_prototypes,
    mtp_text_features,
    logit_scale,
    candidate_space,
    fallback,
    modal="base2novel",
    use_base_classes_only=True,
    prior_temperature=1.0,
    anchor_weight=0.20,
    shift_weight=0.05,
    eps=1e-8,
):
    """Grid-search MTP, DVP, and prior parameters from unlabeled features."""

    # Search-time feature fusion is only used for parameter selection.
    # Final training features and priors are rebuilt by the legacy-exact PromptKD path.
    if not isinstance(fallback, CalibrationFallback):
        fallback = CalibrationFallback(*fallback)

    eps = float(eps)
    image_features = image_features.float()
    raw_text_features = raw_text_features.float()
    if visual_prototypes is None:
        visual_prototypes = raw_text_features
    visual_prototypes = visual_prototypes.float()
    if mtp_text_features is not None:
        mtp_text_features = mtp_text_features.float()

    num_classes = raw_text_features.shape[0]
    scoring_slice, scoring_num_classes = get_scoring_class_slice(
        modal,
        use_base_classes_only,
        num_classes,
    )

    scale = float(logit_scale.item()) if isinstance(logit_scale, torch.Tensor) else float(logit_scale)
    baseline_logits_full = scale * image_features @ raw_text_features.t()
    baseline_logits = baseline_logits_full[:, scoring_slice]
    baseline = score_unlabeled_predictions(
        baseline_logits,
        baseline_logits,
        anchor_weight=anchor_weight,
        shift_weight=shift_weight,
        eps=eps,
    )

    best = None
    best_prior = None
    all_results = []

    for dvp_alpha in candidate_space["dvp_alpha"]:
        t_dvp = fuse_dvp_text_features(raw_text_features, visual_prototypes, dvp_alpha, eps=eps)
        for mtp_alpha in candidate_space["mtp_alpha"]:
            t_candidate = fuse_mtp_text_features(t_dvp, mtp_text_features, mtp_alpha, eps=eps)
            logits_full = scale * image_features @ t_candidate.t()
            prior = estimate_teacher_prior(logits_full, temperature=prior_temperature, eps=eps)

            for prior_gamma in candidate_space["prior_gamma"]:
                corrected_full = apply_prior_correction(logits_full, prior, prior_gamma, eps=eps)
                metrics = score_unlabeled_predictions(
                    corrected_full[:, scoring_slice],
                    baseline_logits,
                    anchor_weight=anchor_weight,
                    shift_weight=shift_weight,
                    eps=eps,
                )
                result = {
                    "mtp_alpha": float(mtp_alpha),
                    "dvp_alpha": float(dvp_alpha),
                    "prior_gamma": float(prior_gamma),
                    **metrics,
                }
                all_results.append(result)
                if math.isfinite(result["score"]) and _result_is_better(result, best, fallback):
                    best = result
                    best_prior = prior.detach().cpu()

    if best is None:
        best = {
            "mtp_alpha": float(fallback.mtp_alpha),
            "dvp_alpha": float(fallback.dvp_alpha),
            "prior_gamma": float(fallback.prior_gamma),
            **_invalid_metrics(),
        }

    return {
        "selected": dict(best),
        "selected_prior": best_prior,
        "baseline": baseline,
        "all_results": all_results,
        "scoring_num_classes": scoring_num_classes,
        "scoring_class_range": [scoring_slice.start, scoring_slice.stop],
    }


def make_calibration_payload(
    metadata,
    fallback,
    candidate_space,
    objective,
    selected,
    baseline,
    all_results,
    num_calibration_samples,
    num_calibration_batches,
    scoring_num_classes,
):
    """Create the JSON-serializable automatic calibration result."""

    if not isinstance(fallback, CalibrationFallback):
        fallback = CalibrationFallback(*fallback)

    return {
        "version": CALIBRATION_CACHE_VERSION,
        "dataset": metadata["dataset"],
        "modal": metadata["modal"],
        "seed": metadata["seed"],
        "teacher_name": metadata["teacher_name"],
        "num_classes": metadata["num_classes"],
        "scoring_num_classes": int(scoring_num_classes),
        "num_calibration_samples": int(num_calibration_samples),
        "num_calibration_batches": int(num_calibration_batches),
        "fallback": {
            "mtp_alpha": float(fallback.mtp_alpha),
            "dvp_alpha": float(fallback.dvp_alpha),
            "prior_gamma": float(fallback.prior_gamma),
        },
        "candidate_space": {
            "mtp_alpha": [float(v) for v in candidate_space["mtp_alpha"]],
            "dvp_alpha": [float(v) for v in candidate_space["dvp_alpha"]],
            "prior_gamma": [float(v) for v in candidate_space["prior_gamma"]],
        },
        "objective": {
            "anchor_weight": float(objective["anchor_weight"]),
            "shift_weight": float(objective["shift_weight"]),
        },
        "selected": selected,
        "baseline": baseline,
        "all_results": all_results,
        "metadata": metadata,
    }


def save_calibration_json(path, payload):
    """Persist an automatic calibration result as JSON."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _metadata_matches(cached, expected):
    return cached == expected


def _selected_is_valid(selected):
    if not isinstance(selected, dict):
        return False
    required = ["mtp_alpha", "dvp_alpha", "prior_gamma", "score"]
    if any(key not in selected for key in required):
        return False
    return (
        _valid_alpha(selected["mtp_alpha"])
        and _valid_alpha(selected["dvp_alpha"])
        and _valid_gamma(selected["prior_gamma"])
        and _is_finite_number(selected["score"])
    )


def load_cached_calibration_json(path, expected_metadata, warn_fn=None):
    """Load a cached calibration JSON only when metadata and selected values match."""

    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(warn_fn, f"Failed to load calibration cache {path}: {exc}")
        return None

    if payload.get("version") != CALIBRATION_CACHE_VERSION:
        _warn(warn_fn, f"Ignoring calibration cache with version {payload.get('version')}.")
        return None

    if not _metadata_matches(payload.get("metadata"), expected_metadata):
        _warn(warn_fn, "Ignoring calibration cache because metadata does not match.")
        return None

    if not _selected_is_valid(payload.get("selected")):
        _warn(warn_fn, "Ignoring calibration cache because selected values are invalid.")
        return None

    return payload


@contextmanager
def preserved_rng(seed):
    """Run calibration with a fixed seed and restore caller RNG states afterward."""

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
