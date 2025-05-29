# BoyoSupercoolWrapper

This is a ComfyUI wrapper for Andrew DalPino's SuperCool upscaler, enabling its use directly within ComfyUI's node workflow. No extra dependencies required—just drop in the models and go.

## 💡 What This Is

The original SuperCool repo ([https://github.com/andrewdalpino/SuperCool](https://github.com/andrewdalpino/SuperCool)) provides a fast and high-quality image upscaler. This wrapper integrates that functionality into ComfyUI using its native Python environment—no external pip installs needed.

All upscaling models are fixed at **4x** internally—despite the file naming for compatibility with ComfyUI conventions (see below).

---

## 🧱 Setup

### 1. Clone or install this repo into your ComfyUI custom nodes directory:

```
git clone https://github.com/DragonDiffusionbyBoyo/BoyoSupercoolWrapper
```

So it ends up like:

```
ComfyUI/custom_nodes/BoyoSupercoolWrapper/
```

### 2. Download the models

| Model Size              | Download Link                                                                                                        | Rename To     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------- |
| Small (fast)            | [https://huggingface.co/andrewdalpino/SuperCool-4x-Small](https://huggingface.co/andrewdalpino/SuperCool-4x-Small)   | supercool\_2x |
| Medium                  | [https://huggingface.co/andrewdalpino/SuperCool-4x-Medium](https://huggingface.co/andrewdalpino/SuperCool-4x-Medium) | supercool\_4x |
| Large (slow, high qual) | [https://huggingface.co/andrewdalpino/SuperCool-4x-Large](https://huggingface.co/andrewdalpino/SuperCool-4x-Large)   | supercool\_8x |

**Note:** All models upscale by **4x**. The naming (`2x`, `4x`, `8x`) is just used for node dropdown compatibility—they do not reflect actual scale factors.

### 3. Put the renamed models here:

```
ComfyUI/models/upscale_models/SuperCool/
```

If the `SuperCool` folder does not exist, create it manually.

---

## ✅ Usage

Once everything is in place, launch ComfyUI and look for the **SuperCool Upscaler** node under the `upscale` category. You can select from three quality/speed presets via dropdown.

---

## 🫼 No Extra Dependencies

This wrapper uses only ComfyUI's standard packages. No additional installations are required.

---

## 🐉 Made by Dragon Diffusion UK

Maintained by Dragon Diffusion UK: [https://www.dragondiffusionuk.co.uk](https://www.dragondiffusionuk.co.uk)
We build practical, bleeding-edge AI integrations for actual humans.

Pull requests, feedback, and banter welcome.
