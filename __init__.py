"""ComfyUI_SolarWM_H3 - SolarWM camera-PRoPE + LoRA injector for the official MiniMax-H3.

Everything is injected at runtime through ModelPatcher / model_options /
transformer_options. No comfy/ source file is touched.
"""
from .solarwm_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
