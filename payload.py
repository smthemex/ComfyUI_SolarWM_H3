"""Camera / PRoPE payload types shared by the SolarWM-H3 nodes.

Everything is plain Python objects riding inside model_options /
transformer_options; no tensors are baked at attach time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

# Custom ComfyUI types used by this pack (old-style string types).
TYPE_CAMERA = "SOLARWM_CAMERA"
TYPE_PROPE = "SOLARWM_PROPE"


@dataclass
class SolarWMCamera:
    """A trajectory of camera-to-world matrices, one per output latent frame.

    c2w: [F, 4, 4] float32 CPU tensor, row-vector convention used by SolarWM.
    """
    c2w: torch.Tensor
    fps: float = 24.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.c2w, torch.Tensor):
            self.c2w = torch.as_tensor(self.c2w, dtype=torch.float32)
        if self.c2w.dim() != 3 or self.c2w.shape[1:] != (4, 4):
            raise ValueError(f"SolarWMCamera expects [F,4,4], got {tuple(self.c2w.shape)}")
        if self.c2w.dtype != torch.float32:
            self.c2w = self.c2w.float()

    @property
    def frames(self) -> int:
        return self.c2w.shape[0]


@dataclass
class SolarWMRowPlan:
    """Attach-time cache describing how DiT rows map onto the camera track.

    ComfyUI's DiT sequence always ends with the *target video* segment
    (segments order [text|cond|audio|video]): those rows are contiguous and
    frame-major, latent frame f occupying rows
    [S - video_rows + f*frame_rows, ... + (f+1)*frame_rows).

    `pixel_frame_ids[f]` is the trajectory index whose camera latent frame f
    follows, so the full camera timeline can be reconstructed without ever
    touching the encoder.
    """
    latent_t: int
    latent_h: int
    latent_w: int
    frame_rows: int            # rows per latent frame = (h//2)*(w//2)
    video_rows: int            # latent_t * frame_rows (trailing DiT rows)
    pixel_frame_ids: tuple[int, ...]
    trajectory_frames: int
    text_len: Optional[int] = None
    warning: str = ""

    def signature(self) -> tuple:
        return (self.latent_t, self.latent_h, self.latent_w,
                self.trajectory_frames, self.text_len)


@dataclass
class SolarWMProPE:
    """Configuration handed to the attention-patch at sampling time.

    Presence of the payload *is* the switch: any node feeding a SolarWMProPE
    into SolarWMAttach opts into the camera-aware path, and leaving the input
    disconnected keeps the model stock.  There is deliberately no boolean on
    the payload itself -- camera-aware sampling is the product behaviour.
    """
    camera: SolarWMCamera
    log_first: bool = True
    meta: dict = field(default_factory=dict)
    # Row/camera timeline plan derived at attach time (see SolarWMRowPlan).
    row_plan: Optional[SolarWMRowPlan] = None


# ---------------------------------------------------------------------------
# Camera helpers. The orbit builder below is only a trajectory *source* so the
# graph can be wired today; the exact SolarWM fused-PRoPE math (liftK /
# fixed-K / logd4 / projection matrices) is in the section further down.
# ---------------------------------------------------------------------------

def build_orbit_camera(frames: int, turn: float = 1.0, radius: float = 1.0,
                       height: float = 0.0,
                       radius_end: Optional[float] = None) -> torch.Tensor:
    """Return [F,4,4] c2w: camera orbiting the Y axis while looking at origin.

    `radius` is the starting camera distance; `radius_end` (optional) sweeps
    it linearly across the clip.  None / equal to `radius` keeps a pure orbit;
    radius_end < radius pushes toward the origin, radius_end > radius pulls
    back.  With orbit_turns = 0 the shrink degenerates to a straight dolly
    in/out along one axis -- the standard push-in test.

    This is intentionally simple; replace by SolarWM trajectory code later.
    """
    start_r = float(radius)
    end_r = start_r if radius_end is None else float(radius_end)
    step_den = max(1, frames - 1)
    eyes = []
    for i in range(frames):
        frac = (i / step_den) if frames > 1 else 0.0
        # Never let the camera collapse onto the look-at origin: a zero eye
        # vector makes the c2w frame degenerate (non-invertible).
        r = max(start_r + (end_r - start_r) * frac, 1e-3)
        yaw = turn * 2.0 * math.pi * i / max(1, frames)
        cx, cz = r * math.cos(yaw), r * math.sin(yaw)
        # look-at (0,0,0): forward = normalize(origin - eye)
        fx, fy, fz = -cx, -height, -cz
        n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / n, fy / n, fz / n
        up = (0.0, 1.0, 0.0)
        sx = fy * up[2] - fz * up[1]
        sy = fz * up[0] - fx * up[2]
        sz = fx * up[1] - fy * up[0]
        sn = math.sqrt(sx * sx + sy * sy + sz * sz) or 1.0
        sx, sy, sz = sx / sn, sy / sn, sz / sn
        ux = sy * fz - sz * fy
        uy = sz * fx - sx * fz
        uz = sx * fy - sy * fx
        # Column-vector c2w (p_world = c2w @ p_cam): rows below are the 4x4 in
        # row-major storage, i.e. columns = [side, up, back=-forward, eye].
        # Homogeneous side/up/back zeros keep a right-handed frame.
        eyes.append([
            [sx, ux, -fx, cx],
            [sy, uy, -fy, height],
            [sz, uz, -fz, cz],
            [0.0, 0.0, 0.0, 1.0],
        ])
    return torch.tensor(eyes, dtype=torch.float32)


# ---------------------------------------------------------------------------
# H3 fused camera-PRoPE math (faithful ports of SolarWM backend)
# ---------------------------------------------------------------------------
# Reference:
#   SolarWM-main/src/solarwm/backends/minimax_h3/camera.py      (constants,
#     np-side validation / first_frame_relative_w2c)
#   SolarWM-main/src/solarwm/backends/minimax_h3/torch_prope.py (torch runtime
#     used inside the H3 forward)
# These functions are *pure* (tensors in / tensors out) so they can be used by
# node code and unit-tested standalone; nothing here touches the model.

# Native MM-RoPE only rotates head dims [0,96); the suffix [96,128) is free and
# is where the camera PRoPE lives.
H3_NATIVE_ROPE_DIM = 96
H3_CAMERA_PROPE_DIM_START = 96
H3_CAMERA_PROPE_DIM_END = 128
H3_ATTENTION_HEAD_DIM = 128
# The projective matrix is applied to 4-wide feature chunks.
H3_PROPE_CHUNK = 4

# Shared normalized Wan intrinsics used by H3's fused PRoPE (camera.py:22-25).
# NOTE: cx/cy=0.5 in camera.py, but the *runtime* torch path clears cx/cy to 0
# (torch_prope._fixed_focal_K); only fx/fy survive.
WAN_FIXED_FX = 969.6969696969696 / (960.0 * 2.0)
WAN_FIXED_FY = 969.6969696969696 / (540.0 * 2.0)

# Temporal chunk geometry (ComfyUI nodes_minimax_h3.py uses the same 17n+5
# alignment; geometry.py offsets select each latent frame's start pixel).
PIXEL_FRAMES_PER_CHUNK = 17
LATENT_FRAMES_PER_CHUNK = 5
LATENT_PREFIX_FRAMES = 2
PIXEL_OFFSETS = (0, 1, 5, 9, 13)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def video_latent_t(pixel_frames: int) -> int:
    """Pixel frames -> latent frames (ComfyUI / SolarWM shared rule)."""
    n = int(pixel_frames)
    return 2 if n <= 5 else ((n - 5) // PIXEL_FRAMES_PER_CHUNK) * LATENT_FRAMES_PER_CHUNK + LATENT_PREFIX_FRAMES


def latent_pixel_count(latent_t: int) -> int:
    """Total pixel frames implied by `latent_t` (ComfyUI frame_count rule)."""
    return int(sum(FRAME_PER_TOKEN[k % LATENT_FRAMES_PER_CHUNK] for k in range(int(latent_t))))


def latent_pixel_start_indices(latent_t: int) -> tuple[int, ...]:
    """For each latent frame f, the pixel-frame index its tokens represent.

    Mirrors SolarWM geometry.latent_aligned_pixel_indices: within every
    17-pixel / 5-latent chunk the representative pixel frames are
    chunk_base + (0,1,5,9,13) -- the *start* of each latent's pixel slot
    (FRAME_PER_TOKEN=(1,4,4,4,4)).
    """
    out = []
    for f in range(int(latent_t)):
        chunk = f // LATENT_FRAMES_PER_CHUNK
        within = f % LATENT_FRAMES_PER_CHUNK
        out.append(chunk * PIXEL_FRAMES_PER_CHUNK + PIXEL_OFFSETS[within])
    return tuple(out)


def build_row_plan(*, latent_t: int, latent_h: int, latent_w: int,
                   trajectory_frames: int, text_len: Optional[int] = None) -> SolarWMRowPlan:
    """Derive the DiT row -> camera-frame plan from sizes only (no tensors)."""
    latent_t = int(latent_t)
    latent_h = int(latent_h)
    latent_w = int(latent_w)
    trajectory_frames = int(trajectory_frames)
    frame_rows = (latent_h // 2) * (latent_w // 2)
    video_rows = latent_t * frame_rows
    ids = latent_pixel_start_indices(latent_t)
    warnings = []

    if latent_t <= 0:
        warnings.append("latent_t<=0: row plan is empty (is the LATENT an H3 AV latent?)")
    if frame_rows <= 0:
        warnings.append(f"latent canvas {latent_h}x{latent_w} yields no rows per frame")

    implied_pixels = latent_pixel_count(latent_t)
    if implied_pixels != trajectory_frames:
        warnings.append(
            f"camera trajectory has {trajectory_frames} frames but this latent "
            f"timeline needs {implied_pixels} pixel frames "
            f"(latent_t={latent_t}); the mapping will be misaligned"
        )
    if ids and max(ids) >= trajectory_frames:
        warnings.append(
            f"trajectory only has {trajectory_frames} frames; last needed "
            f"pixel index is {max(ids)}"
        )

    return SolarWMRowPlan(
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        frame_rows=frame_rows,
        video_rows=video_rows,
        pixel_frame_ids=ids,
        trajectory_frames=trajectory_frames,
        text_len=text_len,
        warning="; ".join(warnings),
    )


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms [...,4,4] via transpose (no generic inverse)."""
    output = transforms.new_zeros(transforms.shape)
    rotation = transforms[..., :3, :3].transpose(-1, -2)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation, transforms[..., :3, 3])
    output[..., 3, 3] = 1.0
    return output


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """[...,3,3] pinhole K -> [...,4,4] homogeneous."""
    output = Ks.new_zeros((*Ks.shape[:-2], 4, 4))
    output[..., :3, :3] = Ks
    output[..., 3, 3] = 1.0
    return output


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    output = Ks.new_zeros(Ks.shape)
    output[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    output[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    output[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    output[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    output[..., 2, 2] = 1.0
    return output


def _fixed_focal_K(reference: torch.Tensor) -> torch.Tensor:
    """Runtime fixed intrinsics: diag(fx, fy, 1) with cx/cy cleared to 0."""
    output = reference.new_zeros(reference.shape)
    output[..., 0, 0] = WAN_FIXED_FX
    output[..., 1, 1] = WAN_FIXED_FY
    output[..., 2, 2] = 1.0
    return output


def logd4_relative_viewmats(viewmats: torch.Tensor) -> torch.Tensor:
    """Zero-safe ``t*log1p(||t||)/(4*||t||)`` (torch_prope.py:22-40)."""
    if tuple(viewmats.shape[-2:]) != (4, 4) or not viewmats.is_floating_point():
        raise ValueError("viewmats must be floating tensors ending in [4,4]")
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    translation = viewmats[..., :3, 3].to(compute_dtype)
    norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    safe = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(norm > 0, torch.log1p(norm) / (4.0 * safe), torch.zeros_like(norm))
    out = viewmats.clone()
    out[..., :3, 3] = (translation * scale).to(viewmats.dtype)
    return out


def first_frame_relative_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """``inverse(c2w[t]) @ c2w[0]``; first frame identity (camera.py:118-132)."""
    if tuple(c2w.shape[-2:]) != (4, 4) or c2w.ndim < 3:
        raise ValueError("c2w must be [..,N,4,4]")
    out = _invert_se3(c2w) @ c2w[..., :1, :, :]
    out[..., 0, :, :] = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    return out


def fused_prope_matrices(relative_viewmats: torch.Tensor,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projective matrices for Q, K/V, attention-output (torch_prope.py:118-133).

    Args:
        relative_viewmats: first-frame-relative W2C [..,N,4,4].
    Returns:
        (query_matrix, kv_matrix, output_matrix), each [..,N,4,4].
        Query uses proj^T; K/V use inverse pose + inverse focal; output uses
        proj. Intrinsics are the fixed Wan focal (fx/fy only).
    """
    views = logd4_relative_viewmats(relative_viewmats)
    fixed = _fixed_focal_K(views.new_zeros((*views.shape[:-2], 3, 3)))
    projection = torch.einsum("...ij,...jk->...ik", _lift_K(fixed), views)
    query_matrix = projection.transpose(-1, -2).to(views.dtype)
    kv_matrix = torch.einsum(
        "...ij,...jk->...ik", _invert_se3(views), _lift_K(_invert_K(fixed))
    ).to(views.dtype)
    return query_matrix, kv_matrix, projection.to(views.dtype)


def apply_prope(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a token-aligned [B,S,4,4] matrix to the suffix [96:128) of [B,H,S,D].

    Native dims [0,96) pass through untouched; the suffix is tiled into 4-wide
    chunks and transformed with ``einsum('bsij,bhspj->bhspi')`` -- identical to
    torch_prope._project.
    """
    batch, heads, sequence, width = features.shape
    if width != H3_ATTENTION_HEAD_DIM:
        raise ValueError(f"H3 camera PRoPE requires head_dim=128, got {width}")
    projective = H3_PROPE_CHUNK
    if tuple(matrix.shape) != (batch, sequence, projective, projective):
        raise ValueError(
            f"matrix must be token-aligned [B,S,{projective},{projective}], "
            f"got {tuple(matrix.shape)} for features {tuple(features.shape)}"
        )
    matrix = matrix.to(device=features.device, dtype=features.dtype)
    native = features[..., :H3_CAMERA_PROPE_DIM_START]
    camera = features[..., H3_CAMERA_PROPE_DIM_START:H3_CAMERA_PROPE_DIM_END]
    suffix_tiles = (H3_CAMERA_PROPE_DIM_END - H3_CAMERA_PROPE_DIM_START) // projective
    tiled = camera.reshape(batch, heads, sequence, suffix_tiles, projective)
    return torch.cat(
        (native, torch.einsum("bsij,bhspj->bhspi", matrix, tiled).reshape(camera.shape)),
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Sequence-level PRoPE matrices (branch-A runtime support)
# ---------------------------------------------------------------------------

def _quaternion_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """[...,3,3] rotation -> [...,4] quaternion (w,x,y,z)."""
    q = m.new_zeros((*m.shape[:-2], 4))
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q[..., 0] = 0.25 * torch.sqrt((trace + 1.0).clamp_min(0.0)) * 4.0
    # Numerically stable branch per element is overkill here; use the trace form
    # only for near-identity and fall back to the Shepperd method when negative.
    pos = trace > 0
    t = (trace + 1.0).clamp_min(1e-12)
    s = 0.5 / torch.sqrt(t)
    w = 0.25 / s
    q[..., 0] = torch.where(pos, w, q[..., 0])
    q[..., 1] = torch.where(pos, (m[..., 2, 1] - m[..., 1, 2]) * s, q[..., 1])
    q[..., 2] = torch.where(pos, (m[..., 0, 2] - m[..., 2, 0]) * s, q[..., 2])
    q[..., 3] = torch.where(pos, (m[..., 1, 0] - m[..., 0, 1]) * s, q[..., 3])
    neg = ~pos
    if bool(neg.any()):
        # Shepperd's method for the negative-trace branch.
        n = neg.sum()
        qm = m[neg]
        qn = q[neg]
        i = torch.tensor([0, 1, 2], device=m.device)
        diag = torch.stack([qm[..., 0, 0], qm[..., 1, 1], qm[..., 2, 2]], -1)  # [n,3]
        argmax = diag.argmax(-1)
        for ax in range(3):
            mask = argmax == ax
            if not bool(mask.any()):
                continue
            mm = qm[mask]
            nxt = (ax + 1) % 3
            nxt2 = (ax + 2) % 3
            t2 = mm[..., ax, ax] - mm[..., nxt, nxt] - mm[..., nxt2, nxt2] + 1.0
            s2 = 0.5 / torch.sqrt(t2.clamp_min(1e-12))
            qn[mask, 0] = (mm[..., nxt2, nxt] - mm[..., nxt, nxt2]) * s2
            qn[mask, ax + 1] = 0.25 / s2
            qn[mask, nxt + 1] = (mm[..., ax, nxt] + mm[..., nxt, ax]) * s2
            qn[mask, nxt2 + 1] = (mm[..., ax, nxt2] + mm[..., nxt2, ax]) * s2
        q[neg] = qn
    return q


def _matrix_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """[...,4] (w,x,y,z) -> [...,3,3] rotation."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = q.new_zeros((*q.shape[:-1], 3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _slerp_c2w(c2w: torch.Tensor, n: int) -> torch.Tensor:
    """Resample [F,4,4] rigid transforms onto n evenly spaced keyframes."""
    f = c2w.shape[0]
    if f == n:
        return c2w
    src = torch.linspace(0, f - 1, n, device=c2w.device, dtype=c2w.dtype)
    lo = src.floor().long().clamp(max=f - 1)
    hi = (lo + 1).clamp(max=f - 1)
    frac = (src - lo).unsqueeze(-1).unsqueeze(-1)
    r0 = _quaternion_from_matrix(c2w[lo, :3, :3])
    r1 = _quaternion_from_matrix(c2w[hi, :3, :3])
    dot = (r0 * r1).sum(-1, keepdim=True)
    r1 = torch.where(dot < 0, -r1, r1)
    omega = torch.acos((r0 * r1).sum(-1, keepdim=True).clamp(-1.0, 1.0))
    # Small-angle samples (duplicate source keyframe / n>f endpoint) fall back
    # to plain lerp weights so the last resampled pose never divides by ~0.
    so = omega.sin()
    near = so.abs() < 1e-7
    w0 = torch.where(near, (1.0 - frac), ((1.0 - frac) * omega).sin() / so.clamp_min(torch.finfo(c2w.dtype).tiny))
    w1 = torch.where(near, frac, (frac * omega).sin() / so.clamp_min(torch.finfo(c2w.dtype).tiny))
    r = _matrix_from_quaternion(w0 * r0 + w1 * r1)
    t = torch.lerp(c2w[lo, :3, 3], c2w[hi, :3, 3], frac[..., 0])
    out = c2w.new_zeros((n, 4, 4))
    out[:, :3, :3] = r
    out[:, :3, 3] = t
    out[:, 3, 3] = 1.0
    return out


def resolve_camera_relative_frames(camera: SolarWMCamera, row_plan: SolarWMRowPlan,
                                   ) -> Optional[torch.Tensor]:
    """[latent_t,4,4] first-frame-relative W2C for every target latent frame.

    Mapping follows SolarWM stage0p5 semantics:
      F == 1             -> static camera (identity relative views)
      F == latent_t      -> per-latent-frame trajectory
      F >= pixel_count   -> trajectory sampled at the latent pixel slots
      otherwise          -> trajectory re-sampled onto the pixel timeline first
    """
    t = int(row_plan.latent_t)
    if t <= 0:
        return None
    c2w = camera.c2w.float()
    f = int(c2w.shape[0])
    ids = torch.as_tensor(row_plan.pixel_frame_ids, dtype=torch.long)
    if f == 1:
        return torch.eye(4, dtype=torch.float32).expand(t, 4, 4).clone()
    if f == t:
        return first_frame_relative_w2c(c2w)
    implied = latent_pixel_count(t)
    traj = c2w if f >= implied else _slerp_c2w(c2w, implied)
    traj = traj[:implied]
    rel = first_frame_relative_w2c(traj)
    return rel[ids]


def sequence_prope_matrices(row_plan: SolarWMRowPlan, relative_frames: torch.Tensor,
                            seq_len: int):
    """[1,seq_len,4,4] (query, kv, output) matrices over a full packed sequence.

    Non-video rows get the identity-view fused matrices (fixed-focal F etc.),
    exactly matching SolarWM's full-path default; target video rows carry the
    per-latent-frame projective matrices.  Returns None when the row layout is
    unusable for `seq_len` (guards refiner sub-sequences etc.).
    """
    video_rows = int(row_plan.video_rows)
    frame_rows = int(row_plan.frame_rows)
    if video_rows <= 0 or frame_rows <= 0:
        return None
    if seq_len < video_rows:
        return None
    va = seq_len - video_rows
    eye = torch.eye(4, dtype=torch.float32)
    base_q, base_kv, base_o = fused_prope_matrices(eye.unsqueeze(0))
    if relative_frames is not None and relative_frames.shape[0] == row_plan.latent_t:
        fq, fkv, fo = fused_prope_matrices(relative_frames)
        if fq.shape[0] == row_plan.latent_t:
            tile = lambda m: m.repeat_interleave(frame_rows, dim=0)  # frame-major
            fq, fkv, fo = tile(fq), tile(fkv), tile(fo)
        else:
            fq = fkv = fo = None
    else:
        fq = fkv = fo = None

    def build(base, frame):
        full = base.squeeze(0).expand(seq_len, 4, 4).clone()
        if frame is not None:
            full[va:] = frame[:video_rows]
        return full.unsqueeze(0)

    if fq is None:
        # Static / degenerate camera: identity-view transforms everywhere are
        # mathematically neutral, so expose no matrices (stock behaviour).
        return None
    return build(base_q, fq), build(base_kv, fkv), build(base_o, fo)


def is_static_camera(camera: SolarWMCamera, row_plan: SolarWMRowPlan) -> bool:
    """True when every target latent frame carries the same pose (no motion)."""
    rel = resolve_camera_relative_frames(camera, row_plan)
    if rel is None:
        return True
    eye = torch.eye(4, dtype=torch.float32)
    return bool(torch.allclose(rel, eye.expand_as(rel), atol=1e-6))
