from __future__ import annotations

from importlib import import_module


_VARIANT_TO_MODULE = {
    "c_concat": "models.mask_transformer.transformer_c_concat",
    "c_concat_v4": "models.mask_transformer.transformer_c_concat_v4",
    "t_concat": "models.mask_transformer.transformer_t_concat",
    "t_concat_v2": "models.mask_transformer.transformer_t_concat_v2",
    "t_concat_v3": "models.mask_transformer.transformer_t_concat_v3",
    "t_concat_v4": "models.mask_transformer.transformer_t_concat_v4",
}


def normalize_transformer_variant(variant: str | None) -> str:
    normalized = (variant or "c_concat").strip().lower()
    if normalized not in _VARIANT_TO_MODULE:
        supported = ", ".join(sorted(_VARIANT_TO_MODULE))
        raise ValueError(
            f"Unsupported transformer_variant={variant!r}. Supported values: {supported}"
        )
    return normalized


def resolve_mask_transformer_class(variant: str | None):
    normalized = normalize_transformer_variant(variant)
    module = import_module(_VARIANT_TO_MODULE[normalized])
    return getattr(module, "MaskTransformer")


def build_mask_transformer(variant: str | None, **kwargs):
    cls = resolve_mask_transformer_class(variant)
    return cls(**kwargs)
