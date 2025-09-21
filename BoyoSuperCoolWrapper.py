import os
import gc
import torch
import folder_paths
from safetensors.torch import load_file
try:
    from .model import SuperCool
except ImportError as e:
    raise ImportError(f"Failed to import SuperCool from model.py: {str(e)}. Ensure model.py is in ComfyUI/custom_nodes/BoyoSupercoolWrapper/")

class BoyoSuperCoolWrapper:
    MODEL_MAP = {
        "Small (4x)": ("supercool_2x.safetensors", 4, 64, 16),
        "Medium (4x)": ("supercool_4x.safetensors", 4, 128, 32),
        "Large (4x)": ("supercool_8x.safetensors", 4, 256, 32),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["Small (4x)", "Medium (4x)", "Large (4x)"], {"default": "Small (4x)"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "unload_after_run": ("BOOLEAN", {"default": True}),
                "force_gc": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Upscales images using SuperCool models with memory optimization and internal batching"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}
        print(f" SuperCool Wrapper initialized on {self.device.upper()} device")

    def load_model(self, model_type):
        """Load model from cache or disk with error handling"""
        model_info = self.MODEL_MAP.get(model_type)
        if not model_info:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_filename, upscale_ratio, num_channels, num_layers = model_info

        # Return cached model if available
        if model_filename in self.model_cache:
            return self.model_cache[model_filename]

        # Resolve model path
        model_dir = os.path.join(folder_paths.get_folder_paths("upscale_models")[0], "SuperCool")
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Model configuration
        model_args = {
            "base_upscaler": "bicubic",
            "upscale_ratio": upscale_ratio,
            "num_channels": num_channels,
            "hidden_ratio": 2,
            "num_layers": num_layers,
        }

        # Initialize model
        try:
            model = SuperCool(**model_args)
            model.remove_weight_norms()
            model.to(self.device)
            model.eval()
            print(f" Loaded {model_type} model on {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

        # Load weights
        try:
            state_dict = load_file(model_path)
            # Handle potential key mismatches
            model_state_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            missing_keys = [k for k in model_state_dict.keys() if k not in state_dict]
            
            if missing_keys:
                print(f" Missing keys in checkpoint: {len(missing_keys)} keys not loaded")
            
            model.load_state_dict(filtered_dict, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {str(e)}")

        self.model_cache[model_filename] = model
        return model

    def unload_model(self, model_type, force_gc=True):
        """Unload model and clean up memory"""
        model_info = self.MODEL_MAP.get(model_type)
        if not model_info:
            return
            
        model_filename = model_info[0]
        
        if model_filename in self.model_cache:
            model = self.model_cache.pop(model_filename)
            
            # Move to CPU before deletion
            model.to("cpu")
            del model
            
            # Clean up memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
            
            # Force garbage collection
            if force_gc:
                gc.collect()
            
            print(f" Unloaded {model_type} model and cleaned memory")

    def upscale(self, image, model_type, batch_size=10, unload_after_run=True, force_gc=True):
        """Process image with memory management and internal batching"""
        total_images = image.shape[0]
        
        # If batch is small enough, process normally
        if total_images <= batch_size:
            return self._process_single_batch(image, model_type, unload_after_run, force_gc)
        
        # Process in batches for large inputs
        print(f" Processing {total_images} images in batches of {batch_size}")
        
        results = []
        model = self.load_model(model_type)
        
        try:
            for i in range(0, total_images, batch_size):
                end_idx = min(i + batch_size, total_images)
                batch = image[i:end_idx]
                current_batch_num = (i // batch_size) + 1
                total_batches = (total_images + batch_size - 1) // batch_size
                
                print(f" Processing batch {current_batch_num}/{total_batches} ({end_idx - i} images)")
                
                # Convert ComfyUI image [B, H, W, C] -> PyTorch [B, C, H, W]
                batch = batch.permute(0, 3, 1, 2).to(self.device)
                
                # Process with memory optimization
                with torch.no_grad():
                    with torch.autocast(device_type=self.device, enabled=('cuda' in self.device)):
                        upscaled_batch = model(batch)
                
                # Convert back to ComfyUI format [B, H, W, C]
                result_batch = upscaled_batch.permute(0, 2, 3, 1).cpu().float()
                results.append(result_batch)
                
                # Clean intermediate tensors
                del batch, upscaled_batch
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
        
        finally:
            # Always unload if requested
            if unload_after_run:
                self.unload_model(model_type, force_gc)
        
        # Concatenate all results
        final_result = torch.cat(results, dim=0)
        print(f" Completed processing {total_images} images")
        return (final_result,)

    def _process_single_batch(self, image, model_type, unload_after_run, force_gc):
        """Process a single batch (original logic)"""
        # Convert ComfyUI image [B, H, W, C] -> PyTorch [B, C, H, W]
        try:
            image = image.permute(0, 3, 1, 2).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")

        # Load model
        model = self.load_model(model_type)
        
        try:
            # Process with memory optimization
            with torch.no_grad():
                with torch.autocast(device_type=self.device, enabled=('cuda' in self.device)):
                    upscaled_tensor = model(image)
            
            # Convert back to ComfyUI format [B, H, W, C]
            result = upscaled_tensor.permute(0, 2, 3, 1).cpu().float()
            
            # Clean intermediate tensors
            del upscaled_tensor
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        finally:
            # Always unload if requested
            if unload_after_run:
                self.unload_model(model_type, force_gc)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "BoyoSuperCoolWrapper": BoyoSuperCoolWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoSuperCoolWrapper": "Boyo SuperCool Upscaler (Memory Optimized + Batching)"
}