"""Runtime injection for the official MiniMax-H3 model.

Strategy (user-confirmed):
  model.clone() + ModelPatcher.add_object_patch("...attn.forward", fn)
  fn is a *plain* function (no self binding): the patcher set_attr's it onto
  the Attention instance, ComfyUI snapshot/restores it around sampling, and
  the attention module itself is captured by closure.

No comfy/ source file is touched.
"""
from __future__ import annotations

from typing import Optional

import torch
import comfy  # noqa: F401  (native rope math runs through comfy.quant_ops)
import comfy.quant_ops  # noqa: F401  (ensure comfy.quant_ops.ck is imported)
from comfy import model_management
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

from .payload import (
    SolarWMProPE,
    apply_prope,
    resolve_camera_relative_frames,
    sequence_prope_matrices,
)

# Key under which the PRoPE payload is exposed to attention wrappers.
OPT_KEY = "solarwm_prope"


def _locate_diffusion_model(patcher):
    """Return (net, prefix) for the actual H3 network inside the patcher.

    ComfyUI wraps the H3 net as BaseModel.diffusion_model; fall back to the
    patcher's own model in case a raw H3 net is handed in.
    """
    root = patcher.model
    if hasattr(root, "diffusion_model"):
        dm = root.diffusion_model
        # sanity: it should expose the H3 top-level module lists
        if hasattr(dm, "blocks"):
            return dm, "diffusion_model"
    if hasattr(root, "blocks"):
        return root, ""
    raise RuntimeError(
        "SolarWMAttach: could not locate the MiniMax-H3 network "
        "(expected BaseModel.diffusion_model with 'blocks'). "
        f"patched model type: {type(root).__name__}"
    )


def _find_attention_attr(block) -> Optional[str]:
    """Name of the single Attention submodule inside a DiT/Refiner block."""
    for name, module in block.named_modules():
        if name and type(module).__name__ == "Attention":
            return name.split(".")[-1]
    return None


def _collect_attn_targets(net, prefix):
    """Return list of (attention_module, full_attr_path_before_forward)."""
    out = []
    for container_name in ("blocks", "token_refiner"):
        container = getattr(net, container_name, None)
        if container is None:
            continue
        # net.blocks is itself the ModuleList; net.token_refiner is a wrapper
        # module whose real list lives at .blocks -- the attr path must include
        # that extra segment or set_attr resolves "token_refiner.0" and fails.
        nested = getattr(container, "blocks", None)
        if nested is not None:
            blocks, extra = nested, ".blocks"
        else:
            blocks, extra = container, ""
        try:
            entries = list(blocks)
        except TypeError:
            continue
        base = f"{prefix}.{container_name}{extra}" if prefix else f"{container_name}{extra}"
        for i, block in enumerate(entries):
            attn_name = _find_attention_attr(block)
            if attn_name is None:
                continue
            attn = getattr(block, attn_name)
            out.append((attn, f"{base}.{i}.{attn_name}"))
    return out


def _seq_prope_matrices(payload: Optional[SolarWMProPE], seq_len: int):
    """Fused [1,S,4,4] (query, kv, output) matrices, or None to stay stock.

    Built once per distinct sequence length and cached on the payload object
    so all 50 DiT + 2 refiner patches share a single copy.  None covers every
    degenerate case (static camera, missing row plan, unusable layout).
    """
    if payload is None or payload.row_plan is None:
        return None
    cache = getattr(payload, "_matrix_cache", None)
    if cache is None:
        cache = payload._matrix_cache = {}
    if seq_len in cache:
        return cache[seq_len]
    mats = None
    try:
        rel = resolve_camera_relative_frames(payload.camera, payload.row_plan)
        if rel is not None:
            eye = torch.eye(4, dtype=rel.dtype)
            if not torch.allclose(rel, eye.expand_as(rel), atol=1e-6):
                mats = sequence_prope_matrices(payload.row_plan, rel, seq_len)
    except Exception as e:
        mats = None
        print(f"[SolarWM-H3] PRoPE matrix build skipped ({type(e).__name__}: {e})")
    cache[seq_len] = mats
    return mats


def _make_forward_patch(attn, orig_forward, prope: Optional[SolarWMProPE]):
    """Closure-captured stand-in forward for one Attention module.

    Stock pass-through only when no payload is present (unconnected node);
    otherwise the payload is available both via closure (`prope`) and via
    transformer_options under OPT_KEY (per-call layout data can ride there
    once per-token frame mapping is implemented).
    """
    def forward(x, rope_freqs=None, transformer_options=None):
        transformer_options = transformer_options if isinstance(transformer_options, dict) else {}
        payload = prope
        per_call = transformer_options.get(OPT_KEY)
        if per_call is not None and per_call is not payload:
            payload = per_call
        if payload is None:
            return orig_forward(x, rope_freqs, transformer_options)

        # --- Branch A: faithful copy of the stock forward + camera PRoPE ---
        # comfy/ldm/minimax/model.py:169-197 exactly, with the fused projective
        # matrices applied right after rope/norm (on q, k, v) and once more on
        # the attention output (SolarWM fuses `proj` into the out path).
        # PRoPE engages only on the rope path (packed DiT sequence): the text
        # token_refiner blocks call this without rope_freqs and stay stock.
        if rope_freqs is None:
            return orig_forward(x, rope_freqs, transformer_options)

        s = x.shape[0]
        heads = attn.heads
        head_dim = attn.head_dim

        q, k, v = attn.qkv_proj(x).split(heads * head_dim, dim=-1)
        v = v.view(s, heads, head_dim)
        q = q.view(1, s, heads, head_dim)
        k = k.view(1, s, heads, head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]

        mats = _seq_prope_matrices(payload, s)
        if mats is None:
            # Static camera / unusable row layout -> pristine stock behaviour
            # (the tail rows got their rope already, which is exactly what the
            # stock kernel sees too).
            return orig_forward(x, rope_freqs, transformer_options)

        if getattr(payload, "log_first", True):
            payload.log_first = False
            print(f"[SolarWM-H3] PRoPE forward engaged at {type(attn).__name__} "
                  f"(heads={heads}, head_dim={head_dim}, seq={s})")
        q_matrix, kv_matrix, out_matrix = mats

        # [S,H,D] -> [1,H,S,D] attention layout, then project each head's
        # [96:128) suffix: q uses proj^T, k/v use inverse-pose/inverse-focal.
        q = apply_prope(q.transpose(0, 1).unsqueeze(0), q_matrix)
        k = apply_prope(k.transpose(0, 1).unsqueeze(0), kv_matrix)
        v = apply_prope(v.clone().transpose(0, 1).unsqueeze(0), kv_matrix)

        out = optimized_attention(
            AttentionTensorContainer(q), AttentionTensorContainer(k),
            AttentionTensorContainer(v), heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options)
        # Stock output is [1,S,heads*head_dim] (head-last).  SolarWM composes
        # the fused `proj` into the attention output per head; re-split to
        # [1,H,S,D], project the suffix, then restore head-last for out_proj.
        ob = out.reshape(1, s, heads, head_dim).permute(0, 2, 1, 3)
        ob = apply_prope(ob, out_matrix)
        out = ob.permute(0, 2, 1, 3).reshape(1, s, heads * head_dim)
        return attn.out_proj(out.squeeze(0))

    return forward


def attach_solarwm(model, prope: Optional[SolarWMProPE] = None,
                   channel: str = "patch_forward",
                   row_plan: Optional["SolarWMRowPlan"] = None):
    """Clone `model` and wire the PRoPE payload + per-attention forward patch.

    `row_plan` (built by the node from positive/latent, see payload.build_row_plan)
    is stashed on the payload so the PRoPE branch can map DiT rows to camera
    frames without re-deriving geometry.

    Diagnostics go to the ComfyUI console (print), not to a graph output --
    this is inspection data, so it must not force the node to carry a second
    output slot.  Returns the cloned patcher.
    """
    patcher = model.clone()
    if prope is not None and row_plan is not None:
        prope.row_plan = row_plan

    if channel not in ("patch_forward",):
        raise NotImplementedError(
            f"SolarWMAttach channel {channel!r}: 'patch_forward' is wired; "
            "the 'optimized_attention_override' kernel route is pending a real "
            "workflow test (see wrap_attn in comfy/ldm/modules/attention.py)."
        )

    net, prefix = _locate_diffusion_model(patcher)

    # Slot the payload into model_options so it also reaches forwards that
    # receive a fresh transformer_options each call.
    if prope is not None:
        mo = patcher.model_options
        to = mo.get("transformer_options")
        if to is None:
            to = mo["transformer_options"] = {}
        elif not isinstance(to, dict):
            to = mo["transformer_options"] = dict(to)
        to[OPT_KEY] = prope

    targets = _collect_attn_targets(net, prefix)
    n_main = n_ref = 0
    for attn, path in targets:
        if ".token_refiner." in path:
            n_ref += 1
        else:
            n_main += 1
        patcher.add_object_patch(f"{path}.forward",
                                 _make_forward_patch(attn, attn.forward, prope))

    status = (f"SolarWM-H3: patched {len(targets)} Attention.forward "
              f"({n_main} DiT + {n_ref} refiner)"
              + (", PRoPE payload attached" if prope is not None else ""))
    print(status)
    return patcher
