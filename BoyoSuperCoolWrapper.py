import os
import torch
import folder_paths
from safetensors.torch import load_file
try:
    from .model import SuperCool
except ImportError as e:
    raise ImportError(f"Failed to import SuperCool from model.py: {str(e)}. Ensure model.py is in ComfyUI/custom_nodes/BoyoSupercoolWrapper/")

class BoyoSuperCoolWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["Small (4x)", "Medium (4x)", "Large (4x)"], {"default": "Small (4x)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}  # Cache loaded models to avoid reloading

    def load_model(self, model_type):
        # Map model type to filename, upscale ratio, num_channels, num_layers
        model_map = {
            "Small (4x)": ("supercool_2x.safetensors", 4, 64, 16),   # Misnamed, actually 4x
            "Medium (4x)": ("supercool_4x.safetensors", 4, 128, 32), # From config.json
            "Large (4x)": ("supercool_8x.safetensors", 4, 256, 32),  # From config.json
        }
        model_filename, upscale_ratio, num_channels, num_layers = model_map[model_type]

        # Check if model is already cached
        if model_filename in self.model_cache:
            return self.model_cache[model_filename]

        # Get model path
        model_dir = os.path.join(folder_paths.get_folder_paths("upscale_models")[0], "SuperCool")
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Define model arguments
        model_args = {
            "base_upscaler": "bicubic",
            "upscale_ratio": upscale_ratio,
            "num_channels": num_channels,
            "hidden_ratio": 2,
            "num_layers": num_layers,
        }

        # Initialize the SuperCool model
        try:
            model = SuperCool(**model_args)
            model.remove_weight_norms()  # Remove weight norm before loading
            model.to(self.device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SuperCool model: {str(e)}")

        # Load .safetensors weights
        try:
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict, strict=False)  # Ignore extra/missing keys
        except Exception as e:
            raise RuntimeError(f"Failed to load .safetensors model {model_filename}: {str(e)}")

        self.model_cache[model_filename] = model
        return model

    def upscale(self, image, model_type):
        try:
            # ComfyUI image is [B, H, W, C], convert to [B, C, H, W]
            image = image.permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]
        except Exception as e:
            raise RuntimeError(f"Failed to process image tensor: {str(e)}")

        model = self.load_model(model_type)

        try:
            with torch.no_grad():
                upscaled_tensor = model(image)  # Expects [B, 3, H, W]
            # Convert back to ComfyUI format [B, H, W, C]
            upscaled_tensor = upscaled_tensor.permute(0, 2, 3, 1)
        except Exception as e:
            raise RuntimeError(f"Upscaling failed: {str(e)}")

        return (upscaled_tensor,)

NODE_CLASS_MAPPINGS = {
    "BoyoSuperCoolWrapper": BoyoSuperCoolWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoSuperCoolWrapper": "Boyo SuperCool Upscaler"
}