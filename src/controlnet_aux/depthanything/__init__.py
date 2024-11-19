import os

import cv2
import numpy as np
import torch
import safetensors.torch
from torchvision import transforms
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from contextlib import nullcontext

from ..util import HWC3, resize_image
from .depth_anything_v2.dpt import DepthAnythingV2

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

class DepthAnythingDetector:
    def __init__(self, model, dtype, is_metric):
        self.model = model
        self.dtype = dtype
        self.is_metric = is_metric
        
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None, local_files_only=False):
        device = torch.device("cpu")
        
        filename = filename or "depth_anything_v2_vitl_fp32.safetensors"

        dtype = torch.float16 if "fp16" in filename else torch.float32
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            #'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir, local_files_only=local_files_only)

        if "vitl" in filename:
            encoder = "vitl"
        elif "vitb" in filename:
            encoder = "vitb"
        elif "vits" in filename:
            encoder = "vits"

        if "hypersim" in filename:
            max_depth = 20.0
        else:
            max_depth = 80.0

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            if 'metric' in filename:
                model = DepthAnythingV2(**{**model_configs[encoder], 'is_metric': True, 'max_depth': max_depth})
            else:
                model = DepthAnythingV2(**model_configs[encoder])
        
        state_dict = safetensors.torch.load_file(model_path, device=device.type)
        
        if is_accelerate_available:
            for key in state_dict:
                set_module_tensor_to_device(model, key, device=device, dtype=dtype, value=state_dict[key])
        else:
            model.load_state_dict(state_dict)

        model.eval()
        
        is_metric = model.is_metric

        return cls(model, dtype, is_metric)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type=None, gamma_corrected=False):
        device = next(iter(self.model.parameters())).device
        offload_device = torch.device("cpu")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        assert input_image.ndim == 3

        orig_H, orig_W, C = input_image.shape

        images = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float().numpy()  # Shape: (1, C, H, W)

        H, W = orig_H, orig_W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if H != orig_H or W != orig_W:
            resized_images = []
            for img in images:
                img_np = np.transpose(img, (1, 2, 0))
                img_resized = cv2.resize(img_np, (W, H), interpolation=cv2.INTER_LINEAR)
                resized_images.append(np.transpose(img_resized, (2, 0, 1)))
            images = np.stack(resized_images, axis=0)

        images = torch.tensor(images).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)

        out = []
        self.model.to(device)
        autocast_condition = (self.dtype != torch.float32)
        with torch.autocast("cuda", dtype=self.dtype) if autocast_condition else nullcontext():
            for img in images:
                depth = self.model(img.unsqueeze(0).to(device))
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                out.append(depth.cpu())
        self.model.to(offload_device)

        depth_out = torch.cat(out, dim=0)
        depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()

        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2
        if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
            depth_out_np = depth_out.numpy()
            resized_depths = []
            for img in depth_out_np:
                img_resized = cv2.resize(img, (final_W, final_H), interpolation=cv2.INTER_LINEAR)
                resized_depths.append(img_resized)
            depth_out = torch.tensor(np.stack(resized_depths, axis=0))

        depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min())
        depth_out = torch.clamp(depth_out, 0, 1)
        if self.is_metric:
            depth_out = 1 - depth_out

        resized_image = resize_image(input_image, image_resolution)
        H, W, C = resized_image.shape

        detected_map = depth_out.numpy()
        detected_map = cv2.resize(detected_map[0], (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray((detected_map * 255).astype(np.uint8))

        return detected_map
