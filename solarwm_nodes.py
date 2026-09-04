"""SolarWM-H3 custom nodes.

Single-user-node skeleton (v0): SolarWMCameraAttach clones the H3 model and
wires a camera-aware PRoPE path (per-attention forward patch) with an inline
placeholder orbit trajectory.

Real SolarWM camera / logd4 / suffix-projection math and the LoRA load land
in the next iteration; until then trajectory generation lives inline here.
The camera/prope payload has no consumer other than the attach step, so both
used to be separate nodes and are now one.  Bypassing this node in the UI
keeps the model stock -- that *is* the A/B reference, there is no enable
switch on purpose.
"""
from __future__ import annotations

from .inject import attach_solarwm
from .payload import (
    SolarWMCamera,
    SolarWMProPE,
    build_orbit_camera,
    build_row_plan,
    latent_pixel_count,
)


def _conditioning_text_len(positive):
    """Text token count from a ComfyUI CONDITIONING input.

    ComfyUI CONDITIONING is always [[tensor, {}], ...]: positive[0][0] is the
    text embedding tensor (its row count is text_len for cross-attention), the
    dict holds sidecar data (pooled_output etc.) and must NOT be relied on for
    the context. The H3 fork builds its layout with text_len = context.shape[1]
    (comfy/ldm/minimax/model.py), so this tensor's row count is authoritative.
    """
    try:
        context = positive[0][0]
    except Exception:
        return None
    if context is None:
        return None
    shape = tuple(getattr(context, "shape", ()))
    if len(shape) == 3:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[0])
    return None


def _latent_video_dims(latent):
    """Best-effort (latent_t, latent_h, latent_w) of the H3 *video* latent.

    The AV latent's ``samples`` is a NestedTensor((video, audio)) with
    video = [B,24,T,H,W]; returns None when the input isn't usable.
    """
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            return None
        if hasattr(samples, "is_nested") and getattr(samples, "is_nested", False):
            video = samples.tensors[0]
        elif isinstance(samples, (tuple, list)):
            video = samples[0]
        else:
            video = samples
        shape = tuple(getattr(video, "shape", ()))
        if len(shape) != 5:  # [B,24,T,H,W]
            return None
        return int(shape[2]), int(shape[3]), int(shape[4])
    except Exception:
        return None


class SolarWMCameraAttach:
    """Clone the H3 model and attach a camera-aware PRoPE path (one node).

    MODEL in -> MODEL out.  A placeholder [F,4,4] orbit/dolly trajectory is
    built inline and always attached; intermediate camera/PRoPE values were
    previously separate graph outputs with no consumer besides the attach
    step, so they now live entirely inside this node.  Attaching *is* the
    opt-in -- bypass the node to keep stock (the A/B reference).  A future
    real SolarWM trajectory source simply replaces build_orbit_camera below.

    Orbit/dolly controls: orbit_turns sweeps the camera around the Y axis
    (negative = reverse); radius is the starting distance to the origin and
    radius_end the final one, swept linearly -- radius_end < radius pushes
    toward the subject (dolly-in, with orbit_turns=0 this is a pure push
    along the view axis) while radius_end > radius pulls back.  Equal
    radius/radius_end = pure orbit.  The trajectory length is *derived* from
    the required latent (latent_t -> pixel-frame count, e.g. 5 -> 17), so
    there is no frames input.  The positive CONDITIONING is required too: it
    supplies the text context length for the row plan.  Row-plan construction
    is not optional -- a plan that cannot be derived fails loudly instead of
    silently attaching a no-op patch that would quietly run stock.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "orbit_turns": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                # Dolly-in/out: radius sweeps linearly toward radius_end across
                # the clip.  Equal values (both 1.0 by default) = pure orbit.
                # 0 is internally floored to 1e-3 so the c2w never collapses
                # onto the look-at origin (that would be non-invertible).
                "radius_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                # The H3 AV LATENT is required: it drives both the trajectory
                # length (latent_t -> pixel-frame count, e.g. 5 -> 17) and the
                # DiT row -> camera-frame table derived at attach time.
                "latent": ("LATENT",),
                # The positive CONDITIONING supplies the text context length
                # for the row plan; every H3 video graph has it anyway.
                "positive": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    CATEGORY = "SolarWM-H3"

    def attach(self, model, orbit_turns, radius, radius_end, latent, positive):
        # The latent drives the trajectory length (latent_t -> pixel-frame
        # count, e.g. 5 -> 17) and the positive conditioning supplies the text
        # context length.  Both are required inputs; a row plan that cannot be
        # derived must fail loudly instead of attaching a silent no-op patch
        # that would quietly run stock.
        video = _latent_video_dims(latent)
        if video is None:
            raise ValueError(
                "SolarWMCameraAttach needs the H3 AV LATENT (samples as "
                "[B,24,T,H,W] video); check what feeds the latent input"
            )
        latent_t, latent_h, latent_w = video
        traj_frames = latent_pixel_count(latent_t)

        text_len = _conditioning_text_len(positive)
        if text_len is None:
            raise ValueError(
                "SolarWMCameraAttach needs the H3 positive CONDITIONING for "
                "its text context length (context tensor not found); every "
                "H3 video graph already has this conditioning"
            )

        prope = SolarWMProPE(
            camera=SolarWMCamera(
                c2w=build_orbit_camera(
                    traj_frames, turn=float(orbit_turns), radius=float(radius),
                    radius_end=float(radius_end),
                )
            )
        )

        row_plan = build_row_plan(
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            trajectory_frames=prope.camera.frames,
            text_len=text_len,
        )
        if row_plan.warning:
            print(f"[SolarWM-H3] [row plan] {row_plan.warning}")
        print(
            "[SolarWM-H3] row plan: "
            f"latent_t={row_plan.latent_t} canvas="
            f"{row_plan.latent_h}x{row_plan.latent_w} "
            f"frame_rows={row_plan.frame_rows} video_rows={row_plan.video_rows} "
            f"text_len={row_plan.text_len} traj={row_plan.trajectory_frames} "
            f"pixel_ids={row_plan.pixel_frame_ids[:8]}{'...' if len(row_plan.pixel_frame_ids) > 8 else ''}"
        )

        patcher = attach_solarwm(model, prope=prope, row_plan=row_plan)
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "SolarWMCameraAttach": SolarWMCameraAttach,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SolarWMCameraAttach": "SolarWM Camera Attach",
}
