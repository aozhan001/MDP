import importlib
import sys
import types

import torch
import torch.nn.functional as F


def _install_promptkd_import_stubs(monkeypatch):
    dassl_pkg = types.ModuleType("dassl")
    monkeypatch.setitem(sys.modules, "dassl", dassl_pkg)

    dassl_engine = types.ModuleType("dassl.engine")

    class _Registry:
        def register(self):
            def deco(cls):
                return cls

            return deco

    class _TrainerX:
        pass

    dassl_engine.TRAINER_REGISTRY = _Registry()
    dassl_engine.TrainerX = _TrainerX
    monkeypatch.setitem(sys.modules, "dassl.engine", dassl_engine)
    dassl_pkg.engine = dassl_engine

    dassl_utils = types.ModuleType("dassl.utils")
    dassl_utils.load_pretrained_weights = lambda *args, **kwargs: None
    dassl_utils.load_checkpoint = lambda *args, **kwargs: None
    dassl_utils.mkdir_if_missing = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dassl.utils", dassl_utils)
    dassl_pkg.utils = dassl_utils

    dassl_optim = types.ModuleType("dassl.optim")
    dassl_optim.build_optimizer = lambda *args, **kwargs: None
    dassl_optim.build_lr_scheduler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dassl.optim", dassl_optim)
    dassl_pkg.optim = dassl_optim

    clip_pkg = types.ModuleType("clip")
    clip_mod = types.ModuleType("clip.clip")
    clip_mod.tokenize = lambda text: torch.ones(1, 4, dtype=torch.long)
    clip_pkg.clip = clip_mod
    monkeypatch.setitem(sys.modules, "clip", clip_pkg)
    monkeypatch.setitem(sys.modules, "clip.clip", clip_mod)

    simple_tokenizer = types.ModuleType("clip.simple_tokenizer")

    class _Tokenizer:
        pass

    simple_tokenizer.SimpleTokenizer = _Tokenizer
    monkeypatch.setitem(sys.modules, "clip.simple_tokenizer", simple_tokenizer)

    clip_model = types.ModuleType("clip.model")
    clip_model.convert_weights = lambda module: module
    monkeypatch.setitem(sys.modules, "clip.model", clip_model)


def _promptkd(monkeypatch):
    _install_promptkd_import_stubs(monkeypatch)
    sys.modules.pop("trainers.promptkd", None)
    return importlib.import_module("trainers.promptkd")


def test_legacy_mtp_fusion_matches_original_formula_fp32_and_alpha_zero(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    base = F.normalize(torch.randn(3, 4), dim=1)
    mtp = F.normalize(torch.randn(3, 4), dim=1)
    alpha = 0.2
    actual = promptkd.apply_legacy_mtp_text_features(base, mtp, alpha)
    expected = (1.0 - alpha) * base + alpha * mtp.to(device=base.device, dtype=base.dtype)
    expected = expected / expected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    assert actual.dtype == base.dtype
    assert actual.shape == base.shape
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(promptkd.apply_legacy_mtp_text_features(base, mtp, 0.0), base)


def test_legacy_mtp_fusion_preserves_fp16_dtype(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    base = F.normalize(torch.randn(3, 4), dim=1).half()
    mtp = F.normalize(torch.randn(3, 4), dim=1).half()
    actual = promptkd.apply_legacy_mtp_text_features(base, mtp, 0.3)
    assert actual.dtype == torch.float16


def test_legacy_dvp_fusion_matches_original_formula(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    base = F.normalize(torch.randn(3, 4), dim=1)
    visual = F.normalize(torch.randn(3, 4), dim=1)
    alpha = 0.4
    eps = 1e-6
    actual = promptkd.apply_legacy_dvp_text_features(base, visual, alpha, eps)
    expected = F.normalize((1.0 - alpha) * base + alpha * visual, dim=1, eps=eps)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_prior_correction_uses_selected_gamma(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    trainer = promptkd.PromptKD.__new__(promptkd.PromptKD)
    trainer.prior_correct = True
    trainer.prior_eps = 1e-6
    trainer.teacher_class_prior = torch.tensor([0.2, 0.8])
    trainer.get_prior_gamma = lambda: 0.5
    logits = torch.tensor([[1.0, 2.0]])
    actual = trainer.get_teacher_logits_for_kd(logits)
    expected = logits - 0.5 * torch.log(trainer.teacher_class_prior.clamp_min(1e-6)).unsqueeze(0)
    assert torch.allclose(actual, expected)
    trainer.get_prior_gamma = lambda: 0.0
    assert torch.allclose(trainer.get_teacher_logits_for_kd(logits), logits)


class _CfgNode:
    pass


def _cfg(auto_calibrate=True, cache=False, recompute=True):
    cfg = _CfgNode()
    cfg.DATASET = _CfgNode()
    cfg.DATASET.NAME = "MockData"
    cfg.TRAINER = _CfgNode()
    cfg.TRAINER.MODAL = "cross"
    cfg.TRAINER.PROMPTKD = _CfgNode()
    kd = cfg.TRAINER.PROMPTKD
    kd.DVP_CACHE = cache
    kd.DVP_RECOMPUTE = recompute
    kd.DVP_CACHE_DIR = "unused"
    kd.DVP_EPS = 1e-6
    kd.DVP_MIN_MASS = 1e-4
    kd.DVP_HARD = False
    kd.DVP_TOPK = 0
    kd.TEACHER_NAME = "mock"
    kd.AUTO_CALIBRATE = auto_calibrate
    return cfg


class _MockTeacher:
    dtype = torch.float32

    def __init__(self):
        self.forward_calls = 0
        self.image_encoder_calls = 0

    def eval(self):
        return self

    def image_encoder(self, image):
        self.image_encoder_calls += 1
        raise AssertionError("DVP must use full teacher forward")

    def __call__(self, image):
        self.forward_calls += 1
        image_features = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        text_features = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        return image_features, text_features, logits


def test_dvp_build_uses_teacher_forward_logits_not_image_encoder(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    trainer = promptkd.PromptKD.__new__(promptkd.PromptKD)
    trainer.cfg = _cfg(cache=False)
    trainer.n_cls = 2
    trainer.device = torch.device("cpu")
    trainer.model_teacher = _MockTeacher()
    trainer.train_loader_x = [{"img": torch.zeros(2, 3)}]
    trainer.parse_batch_calibration = lambda batch: batch["img"]
    trainer._resolve_dvp_cache_path = lambda: "unused.pt"
    trainer._filter_soft_assignments = lambda probs, eps: probs
    result = trainer.build_domain_visual_prototypes()
    assert trainer.model_teacher.forward_calls == 1
    assert trainer.model_teacher.image_encoder_calls == 0
    assert torch.allclose(result["visual_prototypes"], torch.eye(2), atol=1e-4)


def test_auto_calibration_disabled_returns_config_values(monkeypatch):
    promptkd = _promptkd(monkeypatch)
    trainer = promptkd.PromptKD.__new__(promptkd.PromptKD)
    trainer.auto_calibration_success = False
    trainer.dvp_force_disabled = False
    trainer.cfg = _CfgNode()
    trainer.cfg.TRAINER = _CfgNode()
    trainer.cfg.TRAINER.PROMPTKD = _CfgNode()
    trainer.cfg.TRAINER.PROMPTKD.MTP_ALPHA = 0.2
    trainer.cfg.TRAINER.PROMPTKD.DVP_ALPHA = 0.3
    trainer.cfg.TRAINER.PROMPTKD.PRIOR_GAMMA = 0.4
    assert trainer.get_mtp_alpha() == 0.2
    assert trainer.get_dvp_alpha() == 0.3
    assert trainer.get_prior_gamma() == 0.4


def test_final_prior_ignores_injected_calibration_prior(monkeypatch, tmp_path):
    promptkd = _promptkd(monkeypatch)
    trainer = promptkd.PromptKD.__new__(promptkd.PromptKD)
    trainer.cfg = _CfgNode()
    trainer.cfg.DATASET = _CfgNode()
    trainer.cfg.DATASET.NAME = "MockData"
    trainer.cfg.SEED = 1
    trainer.cfg.TRAINER = _CfgNode()
    trainer.cfg.TRAINER.MODAL = "cross"
    trainer.cfg.TRAINER.PROMPTKD = _CfgNode()
    kd = trainer.cfg.TRAINER.PROMPTKD
    kd.TEACHER_NAME = "mock"
    kd.MTP_ALPHA = 0.0
    kd.DVP_ALPHA = 0.0
    kd.PRIOR_CORRECT = True
    kd.PRIOR_GAMMA = 0.5
    kd.PRIOR_EPS = 1e-6
    kd.PRIOR_TEMPERATURE = 1.0
    kd.PRIOR_PRINT_TOPK = 2
    kd.PRIOR_CACHE = False
    kd.PRIOR_RECOMPUTE = True
    kd.PRIOR_CACHE_DIR = str(tmp_path)
    kd.AUTO_CALIBRATE = True
    kd.USE_MULTI_TEMPLATE_TEXT = False
    kd.MTP_TEMPLATE_SET = "auto"
    kd.MTP_CUSTOM_TEMPLATES = ""
    kd.MTP_NORMALIZE_EACH_TEMPLATE = True
    kd.DVP_HARD = False
    kd.DVP_TOPK = 0
    kd.DVP_MIN_MASS = 1e-4
    trainer.n_cls = 2
    trainer.classnames = ["a", "b"]
    trainer.device = torch.device("cpu")
    trainer.dvp_force_disabled = False
    trainer.auto_calibration_success = True
    trainer.selected_mtp_alpha = 0.0
    trainer.selected_dvp_alpha = 0.0
    trainer.selected_prior_gamma = 0.5
    trainer.auto_calibration_teacher_prior = torch.tensor([0.99, 0.01])
    trainer.train_loader_x = [{"img": torch.zeros(2, 3)}]
    trainer.parse_batch_calibration = lambda batch: batch["img"]
    trainer.get_prompt_templates = lambda: []
    trainer.get_teacher_guidance = lambda image, apply_prior=False: (
        None,
        None,
        torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        None,
    )
    trainer.model_teacher = types.SimpleNamespace(eval=lambda: None)
    trainer.build_teacher_class_prior()
    assert not torch.allclose(trainer.teacher_class_prior.cpu(), torch.tensor([0.99, 0.01]), atol=1e-3)
    assert torch.allclose(trainer.teacher_class_prior.cpu(), torch.tensor([0.5, 0.5]), atol=1e-3)


def test_v3_dvp_cache_is_not_loaded_by_v4_metadata(monkeypatch, tmp_path):
    promptkd = _promptkd(monkeypatch)
    cache_path = tmp_path / "dvp_v4.pt"
    torch.save(
        {
            "base_text_features": torch.eye(2),
            "visual_prototypes": torch.eye(2),
            "mass": torch.ones(2),
            "metadata": {"cache_version": "v3"},
        },
        cache_path,
    )
    trainer = promptkd.PromptKD.__new__(promptkd.PromptKD)
    trainer.cfg = _cfg(cache=True, recompute=False)
    trainer.n_cls = 2
    trainer.device = torch.device("cpu")
    trainer.model_teacher = _MockTeacher()
    trainer.train_loader_x = [{"img": torch.zeros(2, 3)}]
    trainer.parse_batch_calibration = lambda batch: batch["img"]
    trainer._resolve_dvp_cache_path = lambda: str(cache_path)
    trainer._filter_soft_assignments = lambda probs, eps: probs
    result = trainer.build_domain_visual_prototypes()
    assert result["loaded_from_cache"] is False
    assert trainer.model_teacher.forward_calls == 1
