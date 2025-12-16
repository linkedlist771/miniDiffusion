# # Get Checkpoints to load for VAE, CLIP, T5, and SD3
# # ////////////////////////// NOT MEANT TO BE RAN WHILE INFERENCE OR TRAINING ////////////////////////////////////////////////////////////////////////

# from transformers import CLIPTextModelWithProjection, T5EncoderModel
# from diffusers import StableDiffusion3Pipeline
# from diffusers import SD3Transformer2DModel
# import torch
# import os
# from pathlib import Path

# ROOT_DIR = Path(__file__).parent.parent
# CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
# CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# def get_path(name: str):
#     return os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", name)

# vae_cache = get_path("vae")

# clip_repo = "openai/clip-vit-large-patch14"
# clip_cache = get_path("clip")

# clip_2_repo = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
# clip_2_cache = get_path("clip_2")

# t5_repo = "google/t5-v1_1-xxl"
# t5_cache = get_path("t5")

# vae_filename = "diffusion_pytorch_model.safetensors"
# clip_filename = "pytorch_model.bin"

# diff_repo = "stabilityai/stable-diffusion-3.5-medium"
# diff_filename = get_path("sd3")

# if not os.path.exists(vae_cache):
#     pipe = StableDiffusion3Pipeline.from_pretrained(diff_repo, token = "", torch_dtype = torch.bfloat16)
#     vae_weights_path = pipe.vae.state_dict()

#     torch.save(vae_weights_path, get_path("vae.pth"))

# if not os.path.exists(diff_filename):

#     model = SD3Transformer2DModel.from_pretrained(
#         diff_repo,  subfolder="transformer",
#         token = "",
#         torch_dtype = torch.bfloat16
#     )
#     torch.save(model.to(torch.bfloat16).state_dict(), get_path("sd3_diff.pth"))

# # --- CLIP Part using Transformers ---

# if not os.path.exists(clip_cache):
#     clip_model = CLIPTextModelWithProjection.from_pretrained(clip_repo, torch_dtype = torch.bfloat16)
#     clip_model_state_dict = clip_model.state_dict()

#     torch.save(clip_model_state_dict,  get_path("clip_model.pth"))
#     print("Saved CLIP state_dict to .pth")

# # must add token = "[TOKEN]" here
# if not os.path.exists(clip_2_cache):
#     clip_model = CLIPTextModelWithProjection.from_pretrained(clip_2_repo, cache_dir = clip_2_cache, torch_dtype = torch.bfloat16,
#                                                          token = "")
#     clip_model_state_dict = clip_model.state_dict()

#     torch.save(clip_model_state_dict,  get_path("clip2.pth"))
#     print(f"Saved CLIP state_dict to {clip_2_cache}")

# if not os.path.exists(t5_cache):

#     t5_encoder = T5EncoderModel.from_pretrained(t5_repo, cache_dir = t5_cache, torch_dtype = torch.bfloat16)
#     t5_encoder_state_dict = t5_encoder.state_dict()

#     torch.save(t5_encoder_state_dict, get_path("t5_encoder.pth"))
#     print("Saved T5 encoder state_dict to .pth")

# Get Checkpoints to load for VAE, CLIP, T5, and SD3
# ////////////////////////// NOT MEANT TO BE RAN WHILE INFERENCE OR TRAINING ////////////////////////////////////////////////////////////////////////

from transformers import CLIPTextModelWithProjection, T5EncoderModel
from diffusers import StableDiffusion3Pipeline
from diffusers import SD3Transformer2DModel
import torch
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def get_path(name: str):
    """返回 checkpoints 目录下的路径"""
    return CHECKPOINTS_DIR / name

# 定义各个模型的保存路径
vae_path = get_path("vae.pth")
clip_path = get_path("clip_model.pth")
clip_2_path = get_path("clip2.pth")
t5_path = get_path("t5_encoder.pth")
sd3_diff_path = get_path("sd3_diff.pth")

# 模型仓库
clip_repo = "openai/clip-vit-large-patch14"
clip_2_repo = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
t5_repo = "google/t5-v1_1-xxl"
diff_repo = "stabilityai/stable-diffusion-3.5-medium"

# --- VAE Part ---
if not vae_path.exists():
    print(f"Downloading VAE and saving to {vae_path}")
    pipe = StableDiffusion3Pipeline.from_pretrained(diff_repo, torch_dtype=torch.bfloat16)
    vae_weights_path = pipe.vae.state_dict()
    torch.save(vae_weights_path, vae_path)
    print(f"Saved VAE state_dict to {vae_path}")

# --- SD3 Transformer Part ---
if not sd3_diff_path.exists():
    print(f"Downloading SD3 Transformer and saving to {sd3_diff_path}")
    model = SD3Transformer2DModel.from_pretrained(
        diff_repo, subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    torch.save(model.to(torch.bfloat16).state_dict(), sd3_diff_path)
    print(f"Saved SD3 Transformer state_dict to {sd3_diff_path}")

# --- CLIP Part using Transformers ---
if not clip_path.exists():
    print(f"Downloading CLIP and saving to {clip_path}")
    clip_model = CLIPTextModelWithProjection.from_pretrained(clip_repo, torch_dtype=torch.bfloat16)
    clip_model_state_dict = clip_model.state_dict()
    torch.save(clip_model_state_dict, clip_path)
    print(f"Saved CLIP state_dict to {clip_path}")

# --- CLIP 2 Part (requires token) ---
if not clip_2_path.exists():
    print(f"Downloading CLIP 2 and saving to {clip_2_path}")
    clip_model = CLIPTextModelWithProjection.from_pretrained(
        clip_2_repo, 
        torch_dtype=torch.bfloat16,
    )
    clip_model_state_dict = clip_model.state_dict()
    torch.save(clip_model_state_dict, clip_2_path)
    print(f"Saved CLIP 2 state_dict to {clip_2_path}")

# --- T5 Part ---
if not t5_path.exists():
    print(f"Downloading T5 and saving to {t5_path}")
    t5_encoder = T5EncoderModel.from_pretrained(t5_repo, torch_dtype=torch.bfloat16)
    t5_encoder_state_dict = t5_encoder.state_dict()
    torch.save(t5_encoder_state_dict, t5_path)
    print(f"Saved T5 encoder state_dict to {t5_path}")

print(f"\nAll checkpoints saved to: {CHECKPOINTS_DIR}")