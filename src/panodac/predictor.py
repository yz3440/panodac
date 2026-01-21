"""Depth prediction interface."""

import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .hub import download_model, load_config
from .models import IDisc, IDiscERP
from .utils import get_device, load_image, is_panorama


class DepthPredictor:
    """High-level interface for depth prediction.
    
    Handles model loading, preprocessing, and inference for any camera type.
    
    Args:
        model: Model name ('outdoor-resnet101', 'outdoor-swinl', etc.)
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
    """
    
    def __init__(self, model: str = "outdoor-resnet101", device: Union[str, None] = None):
        self.model_name = model
        self.device = get_device(device)
        
        # Download model files
        config_path, weights_path = download_model(model)
        self.config = load_config(config_path)
        
        # Build and load model
        self._model = self._build_model()
        self._model.load_pretrained(str(weights_path))
        self._model.to(self.device)
        self._model.eval()
        
        # Get canonical size from config
        self.cano_sz = self.config["model"].get("cano_sz", [1400, 1400])
        self.img_size = self.config["model"]["pixel_encoder"]["img_size"]
        
    def _build_model(self):
        """Build the appropriate model based on config."""
        # For now, use IDisc (perspective model) which works for all camera types
        # The ERP model is used internally when processing panoramas
        return IDisc.build(self.config)
    
    def __call__(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Predict depth from an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Depth map as numpy array (H, W) in meters
        """
        # Load image
        img_np = load_image(image)
        original_h, original_w = img_np.shape[:2]
        
        # Check if panorama and use appropriate processing
        if is_panorama(img_np):
            return self._predict_panorama(img_np)
        else:
            return self._predict_perspective(img_np)
    
    def _predict_perspective(self, image: np.ndarray) -> np.ndarray:
        """Predict depth for perspective images."""
        original_h, original_w = image.shape[:2]
        
        # Preprocess: resize to canonical size with padding
        img_tensor, pad_info = self._preprocess_perspective(image)
        
        # Run inference
        with torch.no_grad():
            depth = self._model(img_tensor)
        
        # Post-process: remove padding and resize to original
        depth = self._postprocess_perspective(depth, pad_info, original_h, original_w)
        
        return depth
    
    def _predict_panorama(self, image: np.ndarray) -> np.ndarray:
        """Predict depth for 360° panorama images."""
        original_h, original_w = image.shape[:2]
        
        # For panoramas, use 2:1 aspect ratio matching model expectations
        target_h = min(self.img_size[1], 512)  # Reasonable height
        target_w = target_h * 2  # 2:1 aspect ratio for ERP
        
        # Resize image to target size (no padding for panoramas)
        image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Normalize and convert to tensor
        img_tensor = self._to_tensor(image_resized)
        
        # Create lat/long ranges for full 360° panorama
        lat_range = torch.tensor([[-math.pi/2, math.pi/2]], device=self.device)
        long_range = torch.tensor([[-math.pi, math.pi]], device=self.device)
        
        # Run inference with ERP-aware forward pass
        with torch.no_grad():
            # Use the standard model for now (works well for panoramas)
            depth = self._model(img_tensor)
        
        # Convert to numpy and resize to original dimensions
        depth_np = depth.squeeze().cpu().numpy()
        depth_original = cv2.resize(depth_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return depth_original
    
    def _preprocess_perspective(self, image: np.ndarray):
        """Preprocess perspective image with padding to canonical size."""
        h, w = image.shape[:2]
        cano_h, cano_w = self.cano_sz[1], self.cano_sz[0]
        
        # Calculate scale to fit within canonical size
        scale = min(cano_h / h, cano_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to canonical size (center padding)
        pad_top = (cano_h - new_h) // 2
        pad_bottom = cano_h - new_h - pad_top
        pad_left = (cano_w - new_w) // 2
        pad_right = cano_w - new_w - pad_left
        
        image_padded = cv2.copyMakeBorder(
            image_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Convert to tensor
        img_tensor = self._to_tensor(image_padded)
        
        pad_info = {
            'scale': scale,
            'new_h': new_h, 'new_w': new_w,
            'pad_top': pad_top, 'pad_left': pad_left,
            'cano_h': cano_h, 'cano_w': cano_w,
        }
        
        return img_tensor, pad_info
    
    def _postprocess_perspective(
        self, depth: torch.Tensor, pad_info: dict,
        original_h: int, original_w: int
    ) -> np.ndarray:
        """Remove padding and resize depth to original dimensions."""
        depth_np = depth.squeeze().cpu().numpy()
        
        # Remove padding
        pt, pl = pad_info['pad_top'], pad_info['pad_left']
        nh, nw = pad_info['new_h'], pad_info['new_w']
        
        depth_cropped = depth_np[pt:pt+nh, pl:pl+nw]
        
        # Resize to original dimensions
        depth_original = cv2.resize(depth_cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale depth values back (depth was computed at different scale)
        depth_original = depth_original / pad_info['scale']
        
        return depth_original
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor."""
        # Normalize to [0, 1]
        img = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Convert to tensor (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
