import os
import sys

GPU=sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import torch
from diffsynth import ModelManager, save_video, VideoData
from diffsynth import WanVideoPipeline
from modelscope import snapshot_download
import pickle
import numpy as np
import pandas as pd

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "../Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "../Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "../Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)


df = pd.read_csv(sys.argv[2])


suffix = sys.argv[3]

lora_file = sys.argv[4]
idx_start = int(sys.argv[5])
idx_final = int(sys.argv[6])


if not os.path.exists(lora_file):
    print("Not exist:")
    print(lora_file)

    if "step=00000" in lora_file:
        exp_dir = os.path.dirname(os.path.dirname(lora_file))
        gen_dir = os.path.join(exp_dir, f"{os.path.basename(lora_file)}_eval_{suffix}")
    else:
        exit()
else:
    model_manager.load_lora(lora_file, lora_alpha=1.0)

    exp_dir = os.path.dirname(os.path.dirname(lora_file))
    gen_dir = os.path.join(exp_dir, f"{os.path.basename(lora_file)}_eval_{suffix}")

pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)


os.makedirs(gen_dir, exist_ok=True)


num_frames = 81

for kdx, row in df.iterrows():
    prompt = row["prompt"]
    prompt_id = row["prompt_id"]

    if not (kdx % idx_start == idx_final):
        continue

    for seed in range(5):
    
        gen_file = os.path.join(gen_dir, f"{prompt_id}_F{num_frames:02d}_seed{seed:02d}.mp4")
        if not os.path.exists(gen_file):
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                seed=seed, 
                tiled=False,
                num_frames=num_frames,
            )
            save_video(video, gen_file, fps=16, quality=5)

            