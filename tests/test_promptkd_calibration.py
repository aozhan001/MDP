import math

import torch

from trainers.promptkd_calibration import (
    CALIBRATION_CACHE_VERSION,
    CalibrationFallback,
    apply_prior_correction,
    fuse_dvp_text_features,
    fuse_mtp_text_features,
    load_cached_calibration_json,
    make_calibration_payload,
    parse_batch_calibration,
    resolve_candidate_space,
    save_calibration_json,
    score_unlabeled_predictions,
    search_calibration_parameters,
)


def _unit(x):
    return torch.nn.functional.normalize(torch.tensor(x, dtype=torch.float32), dim=1)


def _fallback():
    return CalibrationFallback(mtp_alpha=0.2, dvp_alpha=0.2, prior_gamma=0.25)


def _candidate_space():
    return {
        "mtp_alpha": [0.0, 0.2],
        "dvp_alpha": [0.0, 0.2],
        "prior_gamma": [0.0, 0.25],
    }


def test_candidate_dedup_and_sort():
    space = resolve_candidate_space(
        [0.3, 0.1, 0.3, 0.0],
        [0.5, 0.2, 0.5],
        [0.75, 0.0, 0.25, 0.25],
        _fallback(),
    )
    assert space["mtp_alpha"] == [0.0, 0.1, 0.3]
    assert space["dvp_alpha"] == [0.2, 0.5]
    assert space["prior_gamma"] == [0.0, 0.25, 0.75]


def test_invalid_alpha_uses_fallback():
    space = resolve_candidate_space(
        [-0.1, 1.2],
        [0.2],
        [0.0],
        _fallback(),
    )
    assert space["mtp_alpha"] == [0.2]


def test_invalid_gamma_uses_fallback():
    space = resolve_candidate_space(
        [0.0],
        [0.0],
        [-1.0],
        _fallback(),
    )
    assert space["prior_gamma"] == [0.25]


def test_disabled_modules_force_zero():
    space = resolve_candidate_space(
        [0.2],
        [0.2],
        [0.25],
        _fallback(),
        mtp_enabled=False,
        dvp_enabled=False,
        prior_enabled=False,
    )
    assert space == {"mtp_alpha": [0.0], "dvp_alpha": [0.0], "prior_gamma": [0.0]}


def test_entropy_js_and_score_are_finite():
    baseline = torch.tensor([[2.0, 0.0, -1.0], [0.2, 1.0, -0.3]])
    candidate = torch.tensor([[1.8, 0.1, -0.7], [0.1, 1.2, -0.2]])
    metrics = score_unlabeled_predictions(candidate, baseline)
    for value in metrics.values():
        assert math.isfinite(value)


def test_search_is_deterministic_for_same_inputs():
    image_features = _unit([[1.0, 0.0], [0.0, 1.0]])
    raw = _unit([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    visual = _unit([[0.9, 0.1], [0.1, 0.9], [1.0, 0.8]])
    mtp = _unit([[0.8, 0.2], [0.2, 0.8], [0.9, 1.0]])
    kwargs = dict(
        image_features=image_features,
        raw_text_features=raw,
        visual_prototypes=visual,
        mtp_text_features=mtp,
        logit_scale=torch.tensor(1.0),
        candidate_space=_candidate_space(),
        fallback=_fallback(),
        modal="cross",
    )
    assert search_calibration_parameters(**kwargs)["selected"] == search_calibration_parameters(**kwargs)["selected"]


def test_tie_prefers_closer_to_fallback():
    image_features = torch.zeros(3, 2)
    raw = _unit([[1.0, 0.0], [0.0, 1.0]])
    fallback = CalibrationFallback(mtp_alpha=0.2, dvp_alpha=0.0, prior_gamma=0.0)
    result = search_calibration_parameters(
        image_features=image_features,
        raw_text_features=raw,
        visual_prototypes=raw,
        mtp_text_features=raw,
        logit_scale=1.0,
        candidate_space={"mtp_alpha": [0.0, 0.2], "dvp_alpha": [0.0], "prior_gamma": [0.0]},
        fallback=fallback,
        modal="cross",
    )
    assert result["selected"]["mtp_alpha"] == 0.2


def test_tie_then_prefers_smaller_parameter_sum():
    image_features = torch.zeros(3, 2)
    raw = _unit([[1.0, 0.0], [0.0, 1.0]])
    fallback = CalibrationFallback(mtp_alpha=0.1, dvp_alpha=0.0, prior_gamma=0.0)
    result = search_calibration_parameters(
        image_features=image_features,
        raw_text_features=raw,
        visual_prototypes=raw,
        mtp_text_features=raw,
        logit_scale=1.0,
        candidate_space={"mtp_alpha": [0.0, 0.2], "dvp_alpha": [0.0], "prior_gamma": [0.0]},
        fallback=fallback,
        modal="cross",
    )
    assert result["selected"]["mtp_alpha"] == 0.0


def test_parse_batch_calibration_does_not_read_label():
    class LabelRaises(dict):
        def __getitem__(self, key):
            if key == "label":
                raise AssertionError("label must not be read")
            return super().__getitem__(key)

    image = torch.randn(2, 3, 4, 4)
    assert parse_batch_calibration(LabelRaises({"img": image, "label": torch.ones(2)})) is image


def test_json_save_load_and_metadata_mismatch(tmp_path):
    metadata = {
        "dataset": "Synthetic",
        "modal": "cross",
        "seed": 1,
        "teacher_name": "teacher",
        "num_classes": 2,
        "candidate_space": _candidate_space(),
        "scoring_class_range": [0, 2],
        "anchor_weight": 0.2,
        "shift_weight": 0.05,
        "mtp_template_set": "auto",
        "mtp_custom_templates_hash": "abc",
        "mtp_templates_hash": "def",
        "mtp_normalize_each_template": True,
        "dvp_cache_version": "v3",
        "dvp_hard": False,
        "dvp_topk": 0,
        "prior_temperature": 1.0,
    }
    selected = {
        "mtp_alpha": 0.2,
        "dvp_alpha": 0.2,
        "prior_gamma": 0.25,
        "score": 0.1,
        "mi_norm": 0.2,
        "h_cond_norm": 0.3,
        "h_marg_norm": 0.5,
        "js_norm": 0.0,
        "shift_norm": 0.0,
    }
    payload = make_calibration_payload(
        metadata,
        _fallback(),
        _candidate_space(),
        {"anchor_weight": 0.2, "shift_weight": 0.05},
        selected,
        {"score": 0.0, "mi_norm": 0.0, "h_cond_norm": 1.0, "h_marg_norm": 1.0},
        [selected],
        num_calibration_samples=4,
        num_calibration_batches=2,
        scoring_num_classes=2,
    )
    path = tmp_path / "auto_calibration.json"
    save_calibration_json(path, payload)
    loaded = load_cached_calibration_json(path, metadata)
    assert loaded["version"] == CALIBRATION_CACHE_VERSION
    assert loaded["selected"] == selected
    changed = dict(metadata)
    changed["seed"] = 2
    assert load_cached_calibration_json(path, changed) is None


def test_dvp_fusion_is_normalized_and_alpha_endpoints():
    base = _unit([[1.0, 0.0], [0.0, 1.0]])
    visual = _unit([[0.0, 1.0], [1.0, 0.0]])
    fused = fuse_dvp_text_features(base, visual, 0.5)
    assert torch.allclose(fused.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(fuse_dvp_text_features(base, visual, 0.0), base, atol=1e-6)
    assert torch.allclose(fuse_dvp_text_features(base, visual, 1.0), visual, atol=1e-6)


def test_mtp_alpha_zero_keeps_dvp_features():
    base = _unit([[1.0, 0.0], [0.0, 1.0]])
    mtp = _unit([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(fuse_mtp_text_features(base, mtp, 0.0), base, atol=1e-6)


def test_gamma_zero_keeps_logits():
    logits = torch.randn(3, 4)
    prior = torch.full((4,), 0.25)
    assert torch.allclose(apply_prior_correction(logits, prior, 0.0), logits.float())


def test_search_never_returns_nan_parameters():
    image_features = _unit([[1.0, 0.0], [0.0, 1.0]])
    raw = _unit([[1.0, 0.0], [0.0, 1.0]])
    result = search_calibration_parameters(
        image_features=image_features,
        raw_text_features=raw,
        visual_prototypes=raw,
        mtp_text_features=raw,
        logit_scale=1.0,
        candidate_space={"mtp_alpha": [0.0], "dvp_alpha": [0.0], "prior_gamma": [0.0]},
        fallback=CalibrationFallback(0.0, 0.0, 0.0),
        modal="cross",
    )
    selected = result["selected"]
    assert all(math.isfinite(selected[key]) for key in ["mtp_alpha", "dvp_alpha", "prior_gamma"])


def test_all_invalid_scores_return_fallback():
    image_features = torch.tensor([[float("nan"), 0.0]])
    raw = _unit([[1.0, 0.0], [0.0, 1.0]])
    fallback = CalibrationFallback(0.2, 0.3, 0.4)
    result = search_calibration_parameters(
        image_features=image_features,
        raw_text_features=raw,
        visual_prototypes=raw,
        mtp_text_features=raw,
        logit_scale=1.0,
        candidate_space={"mtp_alpha": [0.0], "dvp_alpha": [0.0], "prior_gamma": [0.0]},
        fallback=fallback,
        modal="cross",
    )
    assert result["selected"]["mtp_alpha"] == fallback.mtp_alpha
    assert result["selected"]["dvp_alpha"] == fallback.dvp_alpha
    assert result["selected"]["prior_gamma"] == fallback.prior_gamma
    assert not math.isfinite(result["selected"]["score"])
