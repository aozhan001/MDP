import json
import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

_tokenizer = _Tokenizer()

DVP_CACHE_VERSION = "v2"

DATASET_CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        return final_feat.squeeze(-1).squeeze(-1)


def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME

    if backbone_name == "ViT-B/16":
        model_path = "./clip/ViT-B-16.pt"
    elif backbone_name == "ViT-L/14":
        model_path = "./clip/ViT-L-14.pt"
    elif backbone_name == "ViT-B/32":
        model_path = "./clip/ViT-B-32.pt"
    else:
        print("enter the wrong teacher name.")

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if zero_shot_model:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
    else:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 9,
            "language_depth": 9,
            "vision_ctx": 4,
            "language_ctx": 4,
        }

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = "./clip/ViT-B-16.pt"

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "IVLP",
        "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
        "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
        "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
        "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, (
            "In Independent VL prompting, Language prompt depth should be >=1"
            "\nPlease use VPT trainer if you want to learn only vision branch"
        )
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL
        token_device = clip_model.token_embedding.weight.device

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init).to(token_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(
            f"Number of context words (tokens) for Vision prompting: "
            f"{cfg.TRAINER.PROMPTKD.N_CTX_VISION}"
        )
        self.ctx = nn.Parameter(ctx_vectors)

        self.classnames = list(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts_device = tokenized_prompts.to(token_device)

        print(f"classnames size is {len(classnames)}")

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_device).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        if self.train_modal == "base2novel":
            split = math.ceil(self.n_cls / 2)
            self.register_buffer("token_prefix", embedding[:split, :1, :])
            self.register_buffer("token_suffix", embedding[:split, 1 + n_ctx:, :])
            self.register_buffer("token_prefix2", embedding[split:, :1, :])
            self.register_buffer("token_suffix2", embedding[split:, 1 + n_ctx:, :])
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
            self.register_buffer("token_prefix2", embedding[:, :1, :])
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.trainer_name == "PromptKD" and self.train_modal == "base2novel":
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        self.cfg = cfg

        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image=None, label=None):
        prompts = self.prompt_learner()
        text_device = prompts.device
        tokenized_prompts = self.tokenized_prompts.to(text_device)
        text_features = self.text_encoder(prompts.to(text_device), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class PromptKD(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]

    def _warn_once(self, key, message):
        if not hasattr(self, "_warning_once_cache"):
            self._warning_once_cache = set()
        if key not in self._warning_once_cache:
            print(f"[PromptKD][Warning] {message}")
            self._warning_once_cache.add(key)

    def _sanitize_cache_token(self, value):
        return str(value).replace("/", "-").replace("\\", "-").replace(" ", "")

    def get_current_classnames(self):
        if hasattr(self, "dm") and hasattr(self.dm, "dataset") and hasattr(self.dm.dataset, "classnames"):
            classnames = self.dm.dataset.classnames
        elif hasattr(self.model_teacher, "prompt_learner") and hasattr(self.model_teacher.prompt_learner, "classnames"):
            classnames = self.model_teacher.prompt_learner.classnames
        elif hasattr(self.model, "prompt_learner") and hasattr(self.model.prompt_learner, "classnames"):
            classnames = self.model.prompt_learner.classnames
        else:
            self._warn_once(
                "classnames_missing",
                "Class names are unavailable, so text calibration will fall back to original PromptKD."
            )
            return None

        if classnames is None:
            self._warn_once(
                "classnames_none",
                "Class names resolved to None, so text calibration will fall back to original PromptKD."
            )
            return None

        return list(classnames)

    def get_prompt_templates(self):
        cfg_kd = self.cfg.TRAINER.PROMPTKD
        custom_templates = []
        if cfg_kd.MTP_CUSTOM_TEMPLATES:
            custom_templates = [t.strip() for t in cfg_kd.MTP_CUSTOM_TEMPLATES.split("|") if t.strip()]

        template_set = str(cfg_kd.MTP_TEMPLATE_SET).lower()
        templates = []

        if template_set in ["custom", "manual"]:
            templates = custom_templates
        elif template_set in ["select", "imagenet_select"]:
            templates = list(IMAGENET_TEMPLATES_SELECT)
        elif template_set in ["all", "imagenet_all"]:
            templates = list(IMAGENET_TEMPLATES)
        elif template_set == "auto":
            templates = list(IMAGENET_TEMPLATES_SELECT)
            dataset_template = DATASET_CUSTOM_TEMPLATES.get(self.cfg.DATASET.NAME)
            if dataset_template is not None:
                templates.append(dataset_template)
            if custom_templates:
                templates.extend(custom_templates)
        else:
            self._warn_once(
                "mtp_template_set_unknown",
                f"Unknown MTP_TEMPLATE_SET={cfg_kd.MTP_TEMPLATE_SET}, using auto templates instead."
            )
            templates = list(IMAGENET_TEMPLATES_SELECT)
            dataset_template = DATASET_CUSTOM_TEMPLATES.get(self.cfg.DATASET.NAME)
            if dataset_template is not None:
                templates.append(dataset_template)
            if custom_templates:
                templates.extend(custom_templates)

        deduped = []
        seen = set()
        for template in templates:
            if "{}" not in template:
                self._warn_once(
                    f"mtp_invalid_template_{template}",
                    f"Template '{template}' does not contain '{{}}' and will be ignored."
                )
                continue
            if template not in seen:
                deduped.append(template)
                seen.add(template)

        return deduped

    @torch.no_grad()
    def build_multi_template_text_features(self, classnames, device):
        if self.teacher_text_model is None:
            self._warn_once(
                "mtp_teacher_missing",
                "Zero-shot teacher text model is unavailable, so text calibration will fall back to original PromptKD."
            )
            return None

        if not classnames:
            self._warn_once(
                "mtp_no_classnames",
                "No class names available for multi-template text features, falling back to original PromptKD."
            )
            return None

        templates = self.get_prompt_templates()
        if not templates:
            self._warn_once(
                "mtp_no_templates",
                "No valid templates were found, falling back to original PromptKD."
            )
            return None

        cache_key = None
        if self.cfg.TRAINER.PROMPTKD.MTP_CACHE_TEXT_FEATURES:
            cache_key = (
                tuple(classnames),
                tuple(templates),
                bool(self.cfg.TRAINER.PROMPTKD.MTP_NORMALIZE_EACH_TEMPLATE),
                str(device),
            )
            cached = self._mtp_text_feature_cache.get(cache_key)
            if cached is not None:
                return cached.to(device=device, dtype=self.model_teacher.dtype)

        raw_clip_teacher = self.teacher_text_model
        all_template_features = []
        for template in templates:
            prompts = [template.format(name.replace("_", " ")) for name in classnames]
            tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            text_features = raw_clip_teacher.encode_text(tokenized)
            if self.cfg.TRAINER.PROMPTKD.MTP_NORMALIZE_EACH_TEMPLATE:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_template_features.append(text_features.unsqueeze(0))

        if not all_template_features:
            self._warn_once(
                "mtp_empty_features",
                "No template features were produced, falling back to original PromptKD."
            )
            return None

        mean_text_features = torch.cat(all_template_features, dim=0).mean(dim=0)
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean_text_features = mean_text_features.to(dtype=self.model_teacher.dtype)

        if cache_key is not None:
            self._mtp_text_feature_cache[cache_key] = mean_text_features.detach().cpu()

        if self.cfg.TRAINER.PROMPTKD.MTP_DEBUG:
            print(
                f"[PromptKD][MTP] Built multi-template text features with "
                f"{len(templates)} templates for {len(classnames)} classes."
            )

        return mean_text_features

    def _align_text_feature_shape(self, candidate, reference):
        if candidate is None:
            return None
        if candidate.shape != reference.shape:
            self._warn_once(
                "text_shape_mismatch",
                f"Text feature shape mismatch: got {tuple(candidate.shape)}, expected {tuple(reference.shape)}. "
                "Falling back to original PromptKD text features."
            )
            return None
        return candidate.to(device=reference.device, dtype=reference.dtype)

    def _get_shared_text_features(self, device, dtype):
        if self.dvp_text_features is None:
            raise RuntimeError("DVP text features have not been initialized")
        return self.dvp_text_features.to(device=device, dtype=dtype)

    def _normalize_logit_scale(self, logit_scale, device, dtype):
        if isinstance(logit_scale, torch.Tensor):
            if logit_scale.numel() > 1:
                logit_scale = logit_scale.mean()
            return logit_scale.to(device=device, dtype=dtype)
        return torch.tensor(logit_scale, device=device, dtype=dtype)

    def _filter_soft_assignments(self, probs, eps):
        topk = int(self.cfg.TRAINER.PROMPTKD.DVP_TOPK)

        if topk > 0:
            topk = min(topk, probs.shape[1])
            topk_values, topk_indices = probs.topk(topk, dim=1)
            probs = torch.zeros_like(probs).scatter_(1, topk_indices, topk_values)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
            return probs

        prior = 1.0 / probs.shape[1]
        filtered = torch.clamp(probs - prior, min=0.0)
        row_mass = filtered.sum(dim=1, keepdim=True)

        zero_rows = row_mass.squeeze(1) <= eps
        if zero_rows.any():
            fallback_idx = probs.argmax(dim=1, keepdim=True)
            filtered[zero_rows] = 0.0
            filtered[zero_rows].scatter_(1, fallback_idx[zero_rows], 1.0)
            row_mass = filtered.sum(dim=1, keepdim=True)

        filtered = filtered / row_mass.clamp_min(eps)
        return filtered

    def _record_text_calibration_diag(self, original_text_features, calibrated_text_features):
        if calibrated_text_features is None or original_text_features is None:
            return

        cosine = F.cosine_similarity(
            original_text_features.float(),
            calibrated_text_features.float(),
            dim=1,
        )
        delta = (calibrated_text_features.float() - original_text_features.float()).norm(dim=1)

        self._text_calibration_diag = {
            "epoch": int(getattr(self, "epoch", 0)),
            "cosine_mean": float(cosine.mean().item()),
            "cosine_min": float(cosine.min().item()),
            "delta_mean": float(delta.mean().item()),
            "delta_max": float(delta.max().item()),
            "use_multi_template_text": bool(self.cfg.TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT),
        }

    def _get_base_teacher_text_features(self, tea_text_features):
        if self.cfg.TRAINER.PROMPTKD.DVP_ENABLE:
            return self._get_shared_text_features(
                device=tea_text_features.device,
                dtype=tea_text_features.dtype,
            )
        return tea_text_features.to(device=tea_text_features.device, dtype=tea_text_features.dtype)

    @torch.no_grad()
    def get_calibrated_text_features(self, tea_text_features):
        calibrated = tea_text_features

        if self.cfg.TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT:
            classnames = self.get_current_classnames()
            mtp_features = self.build_multi_template_text_features(classnames, tea_text_features.device)
            mtp_features = self._align_text_feature_shape(mtp_features, tea_text_features)
            if mtp_features is not None:
                alpha = float(self.cfg.TRAINER.PROMPTKD.MTP_ALPHA)
                calibrated = (1.0 - alpha) * calibrated + alpha * mtp_features
                calibrated = calibrated / calibrated.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if self.cfg.TRAINER.PROMPTKD.TEXT_CALIBRATION_DIAGNOSE:
            self._record_text_calibration_diag(tea_text_features, calibrated)

        return calibrated

    @torch.no_grad()
    def get_teacher_text_features(self, tea_text_features):
        base_text_features = self._get_base_teacher_text_features(tea_text_features)
        return self.get_calibrated_text_features(base_text_features)

    def _set_teacher_class_prior(self, class_prior):
        class_prior = class_prior.to(self.device)
        if hasattr(self.model_teacher, "teacher_class_prior"):
            self.model_teacher.teacher_class_prior = class_prior
        else:
            self.model_teacher.register_buffer("teacher_class_prior", class_prior)
        self.teacher_class_prior = self.model_teacher.teacher_class_prior

    def _normalize_class_prior(self, class_prior):
        class_prior = class_prior.flatten().float()
        class_prior = class_prior.clamp_min(self.prior_eps)
        class_prior = class_prior / class_prior.sum()
        return class_prior

    def _get_prior_cache_path(self):
        cache_name = (
            f"{self.cfg.TRAINER.MODAL}_{self.cfg.DATASET.NAME}_seed{self.cfg.SEED}"
            f"_ncls{self.n_cls}_temp{self.prior_temperature}.pth"
        )
        return osp.join(self.cfg.TRAINER.PROMPTKD.PRIOR_CACHE_DIR, cache_name)

    def _log_teacher_class_prior(self, class_prior):
        class_prior = class_prior.detach().float().cpu()
        entropy = -(class_prior * class_prior.log()).sum().item()
        topk = min(self.prior_print_topk, class_prior.numel())
        values, indices = torch.topk(class_prior, k=topk)
        topk_pairs = []
        for idx, value in zip(indices.tolist(), values.tolist()):
            cname = self.classnames[idx] if idx < len(self.classnames) else str(idx)
            topk_pairs.append(f"{idx}:{cname}={value:.6f}")

        print(f"CPC-KD enabled: {self.prior_correct}")
        print(f"CPC-KD PRIOR_GAMMA: {self.prior_gamma}")
        print(f"CPC-KD PRIOR_TEMPERATURE: {self.prior_temperature}")
        print(f"CPC-KD prior cache path: {self.prior_cache_path}")
        print(
            "CPC-KD prior stats: "
            f"min={class_prior.min().item():.6e}, "
            f"max={class_prior.max().item():.6e}, "
            f"entropy={entropy:.6f}"
        )
        print("CPC-KD top-k prior classes: {}".format(", ".join(topk_pairs)))

    def get_teacher_logits_for_kd(self, tea_logits):
        if not self.prior_correct:
            return tea_logits

        prior = self.teacher_class_prior.to(tea_logits.device).type_as(tea_logits)
        prior_log = torch.log(prior.clamp_min(self.prior_eps)).unsqueeze(0)
        return tea_logits - self.prior_gamma * prior_log

    def compute_kd_loss(self, teacher_logits, student_logits, temperature):
        teacher_logits = torch.nan_to_num(
            teacher_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_(min=-100.0, max=100.0)
        student_logits = torch.nan_to_num(
            student_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_(min=-100.0, max=100.0)
        temperature = max(float(temperature), 1e-6)

        base_kl = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1)
        base_kl = base_kl * (temperature * temperature) / student_logits.shape[1]
        base_kl = torch.nan_to_num(base_kl, nan=0.0, posinf=0.0, neginf=0.0)
        return base_kl.mean()

    @torch.no_grad()
    def build_domain_visual_prototypes(self):
        dvp_cfg = self.cfg.TRAINER.PROMPTKD
        cache_path = self._resolve_dvp_cache_path()

        if dvp_cfg.DVP_CACHE and osp.exists(cache_path) and not dvp_cfg.DVP_RECOMPUTE:
            cache = torch.load(cache_path, map_location="cpu")
            print(f"Loaded DVP cache from {cache_path}")
            mass = cache["mass"].float()
            return {
                "fused_text_features": cache["fused_text_features"].float(),
                "base_text_features": cache["base_text_features"].float(),
                "visual_prototypes": cache["visual_prototypes"].float(),
                "mass": mass,
                "cache_path": cache_path,
                "loaded_from_cache": True,
                "fallback_mask": mass < float(dvp_cfg.DVP_MIN_MASS),
            }

        loader = getattr(self, "train_loader_x", None)
        if loader is None:
            loader = getattr(self, "train_loader", None)
        if loader is None:
            raise RuntimeError("No training loader available for building DVP prototypes")

        self.model_teacher.eval()
        base_text_features = None
        proto_sum = None
        mass = None
        eps = float(dvp_cfg.DVP_EPS)

        for batch in tqdm(loader, desc="Building DVP", leave=False):
            image, _ = self.parse_batch_train(batch)
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

            tea_image_features = tea_image_features.detach().float()
            tea_text_features = tea_text_features.detach().float()
            tea_logits = tea_logits.detach().float()

            if base_text_features is None:
                base_text_features = tea_text_features
                feat_dim = tea_image_features.shape[1]
                proto_sum = torch.zeros(
                    self.n_cls,
                    feat_dim,
                    device=tea_image_features.device,
                    dtype=torch.float32,
                )
                mass = torch.zeros(
                    self.n_cls,
                    device=tea_image_features.device,
                    dtype=torch.float32,
                )

            if dvp_cfg.DVP_HARD:
                pseudo = tea_logits.argmax(dim=1)
                proto_sum.index_add_(0, pseudo, tea_image_features)
                mass.index_add_(
                    0,
                    pseudo,
                    torch.ones(pseudo.size(0), device=tea_image_features.device, dtype=torch.float32),
                )
            else:
                probs = F.softmax(tea_logits, dim=1)
                probs = self._filter_soft_assignments(probs, eps)
                proto_sum += probs.t() @ tea_image_features
                mass += probs.sum(dim=0)

        if base_text_features is None:
            raise RuntimeError("Failed to build DVP prototypes because no training batches were found")

        visual_proto = proto_sum / mass.clamp_min(eps).unsqueeze(1)
        visual_proto = F.normalize(visual_proto, dim=1, eps=eps)

        fallback_mask = mass < float(dvp_cfg.DVP_MIN_MASS)
        if fallback_mask.any():
            visual_proto[fallback_mask] = base_text_features[fallback_mask]

        alpha = float(dvp_cfg.DVP_ALPHA)
        fused = F.normalize(
            (1.0 - alpha) * base_text_features + alpha * visual_proto,
            dim=1,
            eps=eps,
        )

        cache_payload = {
            "fused_text_features": fused.detach().cpu(),
            "base_text_features": base_text_features.detach().cpu(),
            "visual_prototypes": visual_proto.detach().cpu(),
            "mass": mass.detach().cpu(),
            "dataset": self.cfg.DATASET.NAME,
            "modal": self.cfg.TRAINER.MODAL,
            "alpha": dvp_cfg.DVP_ALPHA,
            "hard": dvp_cfg.DVP_HARD,
            "topk": dvp_cfg.DVP_TOPK,
            "teacher": dvp_cfg.TEACHER_NAME,
            "n_cls": self.n_cls,
            "cache_version": DVP_CACHE_VERSION,
        }

        if dvp_cfg.DVP_CACHE:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            torch.save(cache_payload, cache_path)
            print(f"Saved DVP cache to {cache_path}")

        return {
            **cache_payload,
            "cache_path": cache_path,
            "loaded_from_cache": False,
            "fallback_mask": fallback_mask.detach().cpu(),
        }

    def _resolve_dvp_cache_path(self):
        dvp_cfg = self.cfg.TRAINER.PROMPTKD
        cache_root = dvp_cfg.DVP_CACHE_DIR
        if osp.isabs(cache_root):
            cache_dir = cache_root
        else:
            output_dir = self.cfg.OUTPUT_DIR if self.cfg.OUTPUT_DIR else "."
            cache_parent = osp.dirname(osp.abspath(output_dir))
            cache_dir = osp.join(cache_parent, cache_root)

        dataset = self._sanitize_cache_token(self.cfg.DATASET.NAME)
        modal = self._sanitize_cache_token(self.cfg.TRAINER.MODAL)
        teacher = self._sanitize_cache_token(dvp_cfg.TEACHER_NAME)
        alpha = self._sanitize_cache_token(dvp_cfg.DVP_ALPHA)
        hard = self._sanitize_cache_token(dvp_cfg.DVP_HARD)
        topk = self._sanitize_cache_token(dvp_cfg.DVP_TOPK)
        min_mass = self._sanitize_cache_token(dvp_cfg.DVP_MIN_MASS)
        seed = self._sanitize_cache_token(self.cfg.SEED)
        filename = (
            f"dvp_{DVP_CACHE_VERSION}_{dataset}_{modal}_seed{seed}_{teacher}_"
            f"a{alpha}_hard{hard}_topk{topk}_minm{min_mass}_c{self.n_cls}.pt"
        )
        return osp.join(cache_dir, filename)

    @torch.no_grad()
    def get_teacher_guidance(self, image, label=None, apply_prior=True):
        tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)
        teacher_text_features = self.get_teacher_text_features(tea_text_features)

        if self.cfg.TRAINER.PROMPTKD.DVP_ENABLE or self.cfg.TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT:
            teacher_logit_scale = self._normalize_logit_scale(
                self.model_teacher.logit_scale.exp(),
                tea_image_features.device,
                tea_image_features.dtype,
            )
            teacher_logits = teacher_logit_scale * tea_image_features @ teacher_text_features.t()
        else:
            teacher_logits = tea_logits

        if apply_prior:
            teacher_logits_for_kd = self.get_teacher_logits_for_kd(teacher_logits)
        else:
            teacher_logits_for_kd = teacher_logits

        return tea_image_features, teacher_text_features, teacher_logits, teacher_logits_for_kd

    def build_teacher_class_prior(self):
        prior_cfg = self.cfg.TRAINER.PROMPTKD
        self.prior_correct = prior_cfg.PRIOR_CORRECT
        self.prior_gamma = prior_cfg.PRIOR_GAMMA
        self.prior_eps = prior_cfg.PRIOR_EPS
        self.prior_temperature = prior_cfg.PRIOR_TEMPERATURE
        self.prior_print_topk = prior_cfg.PRIOR_PRINT_TOPK
        self.prior_cache_path = self._get_prior_cache_path()

        if not self.prior_correct:
            class_prior = torch.full((self.n_cls,), 1.0 / self.n_cls, device=self.device)
            self._set_teacher_class_prior(class_prior)
            self.model_teacher.eval()
            self._log_teacher_class_prior(self.teacher_class_prior)
            return

        class_prior = None
        use_cache = prior_cfg.PRIOR_CACHE
        should_load_cache = use_cache and osp.exists(self.prior_cache_path) and not prior_cfg.PRIOR_RECOMPUTE

        if should_load_cache:
            cache = torch.load(self.prior_cache_path, map_location="cpu")
            cached_prior = cache.get("class_prior", None)
            if cached_prior is not None:
                cached_prior = cached_prior.flatten()
                if cached_prior.shape[0] == self.n_cls:
                    class_prior = self._normalize_class_prior(cached_prior).to(self.device)
                    print(f"Loaded teacher class prior from cache: {self.prior_cache_path}")
                else:
                    print(
                        "Ignoring teacher prior cache due to mismatched shape: "
                        f"got {cached_prior.shape[0]}, expected {self.n_cls}"
                    )

        if class_prior is None:
            if use_cache:
                cache_dir = osp.dirname(self.prior_cache_path)
                if cache_dir:
                    mkdir_if_missing(cache_dir)

            loader = getattr(self, "train_loader_x", None)
            if loader is None:
                loader = getattr(self, "train_loader", None)
            if loader is None:
                raise RuntimeError("No training loader available for building teacher class prior")

            prior_sum = torch.zeros(self.n_cls, device=self.device, dtype=torch.float32)
            total_samples = 0
            self.model_teacher.eval()

            for batch in tqdm(loader, desc="Building teacher class prior"):
                image, _ = self.parse_batch_train(batch)
                _, _, teacher_logits, _ = self.get_teacher_guidance(image, apply_prior=False)
                probs = torch.softmax(teacher_logits / self.prior_temperature, dim=1)
                prior_sum += probs.float().sum(dim=0)
                total_samples += probs.shape[0]

            if total_samples == 0:
                raise RuntimeError("No samples found when building teacher class prior")

            class_prior = self._normalize_class_prior(prior_sum / total_samples).to(self.device)

            if use_cache:
                torch.save(
                    {
                        "class_prior": class_prior.cpu(),
                        "dataset": self.cfg.DATASET.NAME,
                        "modal": self.cfg.TRAINER.MODAL,
                        "n_cls": self.n_cls,
                        "prior_temperature": self.prior_temperature,
                    },
                    self.prior_cache_path,
                )
                print(f"Saved teacher class prior to cache: {self.prior_cache_path}")

        self._set_teacher_class_prior(class_prior)
        self.model_teacher.eval()
        self._log_teacher_class_prior(self.teacher_class_prior)

    def _get_eval_loader(self, split):
        if split == "val" and self.val_loader is not None:
            return self.val_loader
        if split == "train":
            return self.train_loader_x
        return self.test_loader

    @torch.no_grad()
    def maybe_run_text_calibration_diagnose(self):
        if not self.cfg.TRAINER.PROMPTKD.TEXT_CALIBRATION_DIAGNOSE:
            return

        split = self.cfg.TRAINER.PROMPTKD.TEXT_CALIBRATION_DIAG_SPLIT
        data_loader = self._get_eval_loader(split)
        if data_loader is None:
            self._warn_once(
                "diag_no_loader",
                f"Diagnostic split '{split}' is unavailable, skipping text calibration diagnostics."
            )
            return

        self.set_model_mode("eval")
        teacher_text = None
        calibrated_text = None
        image_batches = 0
        logit_shift = []
        for batch in data_loader:
            image, _ = self.parse_batch_test(batch)
            tea_image_features, tea_text_features, _ = self.model_teacher(image)
            teacher_text = self._get_base_teacher_text_features(tea_text_features)
            calibrated_text = self.get_calibrated_text_features(teacher_text)
            calibrated_logits = self._normalize_logit_scale(
                self.model_teacher.logit_scale.exp(),
                tea_image_features.device,
                tea_image_features.dtype,
            ) * tea_image_features @ calibrated_text.t()
            base_logits = self._normalize_logit_scale(
                self.model_teacher.logit_scale.exp(),
                tea_image_features.device,
                tea_image_features.dtype,
            ) * tea_image_features @ teacher_text.t()
            shift = (calibrated_logits - base_logits).abs().mean().item()
            logit_shift.append(shift)
            image_batches += 1
            if image_batches >= 1:
                break

        if teacher_text is None or calibrated_text is None:
            return

        self._record_text_calibration_diag(teacher_text, calibrated_text)
        if self._text_calibration_diag is None:
            return

        self._text_calibration_diag["split"] = split
        self._text_calibration_diag["mean_abs_logit_shift"] = float(np.mean(logit_shift)) if logit_shift else 0.0

        diag_path = osp.join(
            self.output_dir,
            self.cfg.TRAINER.PROMPTKD.TEXT_CALIBRATION_DIAG_FILENAME,
        )
        with open(diag_path, "w") as f:
            json.dump(self._text_calibration_diag, f, indent=2)

        print(f"[PromptKD][Diag] Saved text calibration diagnostics to {diag_path}")

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames
        self.classnames = list(classnames)
        self.n_cls = len(classnames)
        self.train_modal = cfg.TRAINER.MODAL
        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
        self._mtp_text_feature_cache = {}
        self._warning_once_cache = set()
        self._text_calibration_diag = None
        self.teacher_text_model = None
        self.dvp_text_features = None
        self.dvp_cache_path = None
        self.dvp_loaded_from_cache = False
        self.dvp_mass = None
        self.dvp_fallback_mask = None
        self.teacher_class_prior = None
        self.prior_cache_path = None

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT or cfg.TRAINER.PROMPTKD.TEXT_CALIBRATION_DIAGNOSE:
            clip_model_teacher_zeroshot = load_clip_to_cpu_teacher(cfg, zero_shot_model=True)
            self.teacher_text_model = clip_model_teacher_zeroshot.to(self.device)
            self.teacher_text_model.eval()
            for param in self.teacher_text_model.parameters():
                param.requires_grad_(False)

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)

        if cfg.TRAINER.MODAL == "base2novel":
            model_path = "./teacher_model/" + str(cfg.DATASET.NAME) + "/VLPromptLearner/model-best.pth.tar"
        elif cfg.TRAINER.MODAL == "cross":
            model_path = "./teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20"
        else:
            raise ValueError(f"Unsupported modal: {cfg.TRAINER.MODAL}")

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]

        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]
        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]

        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()

        if cfg.TRAINER.PROMPTKD.DVP_ENABLE:
            dvp_result = self.build_domain_visual_prototypes()
            self.dvp_text_features = dvp_result["fused_text_features"].to(self.device).detach()
            self.dvp_text_features.requires_grad_(False)
            self.dvp_cache_path = dvp_result["cache_path"]
            self.dvp_loaded_from_cache = dvp_result["loaded_from_cache"]
            self.dvp_mass = dvp_result["mass"].float()
            self.dvp_fallback_mask = dvp_result["fallback_mask"].bool()

            mass = self.dvp_mass
            fallback_count = int(self.dvp_fallback_mask.sum().item())
            print(
                f"DVP enabled: alpha={cfg.TRAINER.PROMPTKD.DVP_ALPHA}, "
                f"hard={cfg.TRAINER.PROMPTKD.DVP_HARD}, "
                f"topk={cfg.TRAINER.PROMPTKD.DVP_TOPK}"
            )
            if not cfg.TRAINER.PROMPTKD.DVP_HARD and int(cfg.TRAINER.PROMPTKD.DVP_TOPK) == 0:
                print("DVP soft pooling uses uniform-debiased assignments when topk=0")
            print(
                f"DVP prototype mass stats: min={mass.min().item():.6f}, "
                f"mean={mass.mean().item():.6f}, max={mass.max().item():.6f}"
            )
            print(f"DVP fallback classes: {fallback_count}")
            print(f"DVP cache path: {self.dvp_cache_path}")
        else:
            print("DVP disabled: using original shared class vectors")

        self.build_teacher_class_prior()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        with torch.no_grad():
            _, teacher_text_features, _, teacher_logits_for_kd = self.get_teacher_guidance(image)

        model = self.model
        optim = self.optim
        prec = self.cfg.TRAINER.PROMPTKD.PREC

        with autocast(enabled=prec == "amp"):
            image_ft, logit_scale = model(image, label)
            logit_scale = self._normalize_logit_scale(logit_scale, image_ft.device, image_ft.dtype)
            student_text_features = teacher_text_features.to(device=image_ft.device, dtype=image_ft.dtype)
            stu_logits = logit_scale * image_ft @ student_text_features.t().detach()
            loss = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * self.compute_kd_loss(
                teacher_logits_for_kd.to(stu_logits.device),
                stu_logits,
                self.temperature,
            )

        optim.zero_grad()
        if prec == "amp":
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader_x
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            _, teacher_text_features, _, _ = self.get_teacher_guidance(image, label, apply_prior=False)
            image_ft, logit_scale = self.model(image, label)
            logit_scale = self._normalize_logit_scale(logit_scale, image_ft.device, image_ft.dtype)
            teacher_text_features = teacher_text_features.to(device=image_ft.device, dtype=image_ft.dtype)

            if self.train_modal == "base2novel":
                split_point = math.ceil(self.n_cls / 2)
                if split == "val":
                    classifier = teacher_text_features[:split_point, :]
                elif split == "test":
                    classifier = teacher_text_features[split_point:, :]
                else:
                    classifier = teacher_text_features
            elif self.train_modal == "cross":
                classifier = teacher_text_features
            else:
                raise ValueError(f"Unsupported modal: {self.train_modal}")

            output = logit_scale * image_ft @ classifier.t()
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        self.maybe_run_text_calibration_diagnose()

        return list(results.values())[0]
