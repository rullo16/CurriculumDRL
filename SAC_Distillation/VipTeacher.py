"""
VIP Teacher Model for Visual Feature Extraction
================================================

This module provides the VIP (Visual Pretraining) teacher model
for extracting pretrained visual features using ResNet50.

VIP is trained on large-scale video data and provides strong
visual representations for robotic control tasks.

No critical bugs - only improvements for clarity.

Author: Improved Implementation
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchrl.envs.transforms import VIPTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# VIP TEACHER MODEL
# ============================================================================

class VipTeacher(nn.Module):
    """
    VIP (Visual Pretraining) Teacher Model
    
    Uses pretrained ResNet50 from VIP to extract visual features.
    VIP is trained on large-scale video data and provides robust
    visual representations for downstream control tasks.
    
    Features:
    - ResNet50 backbone pretrained on video data
    - 2048-dimensional output features
    - Efficient inference (no gradients)
    """
    
    def __init__(self):
        """
        Initialize VIP teacher model.
        
        Downloads pretrained ResNet50 weights if not already cached.
        """
        super().__init__()
        
        # Resize images to 224x224 (VIP input size)
        self.resize = Resize(224)
        
        # Load pretrained VIP transform
        # This wraps the ResNet50 model with VIP weights
        self.tfm = VIPTransform(
            in_keys=["pixels"],
            download=True,  # Download weights if needed
            model_name="resnet50"
        ).eval().to(DEVICE)
        
        # Freeze all parameters (teacher is not trained)
        for param in self.tfm.parameters():
            param.requires_grad = False
        
        # ResNet50 output dimension
        self.out_dim = 2048

    @torch.no_grad()
    def forward(self, x):
        """
        Extract visual features from images.
        
        Args:
            x: Input images
               - Shape: (B, C, H, W)
               - Can be uint8 [0,255] or float [0,1]
               - Any spatial resolution (will be resized to 224x224)
        
        Returns:
            features: Visual features
               - Shape: (B, 2048)
               - Normalized embeddings from ResNet50
        """
        # Ensure float dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        # Normalize to [0, 1] if needed
        if x.max() > 1.01:
            x = x / 255.0
        
        # VIP expects 3-channel RGB input
        # If grayscale or single channel, repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 channels
            x = x[:, :3, :, :]
        
        # Resize to 224x224 (VIP input size)
        x = self.resize(x)
        
        # Extract VIP features
        # Returns dictionary with "vip_vec" key
        return self.tfm._vip(x)["vip_vec"]
    
    def extract_features_batched(self, images, batch_size=32):
        """
        Extract features for large number of images in batches.
        
        Args:
            images: Tensor of images (N, C, H, W)
            batch_size: Batch size for processing
        
        Returns:
            features: Tensor of features (N, 2048)
        """
        all_features = []
        
        num_images = images.shape[0]
        num_batches = (num_images + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            
            batch = images[start_idx:end_idx]
            features = self.forward(batch)
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_feature_similarity(features1, features2):
    """
    Compute cosine similarity between two sets of features.
    
    Args:
        features1: First set of features (N, dim)
        features2: Second set of features (M, dim)
    
    Returns:
        similarity: Cosine similarity matrix (N, M)
    """
    # Normalize features
    features1_norm = torch.nn.functional.normalize(features1, dim=1)
    features2_norm = torch.nn.functional.normalize(features2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(features1_norm, features2_norm.t())
    
    return similarity


def extract_vip_dataset(images, teacher_model, batch_size=32, save_path=None):
    """
    Extract VIP features for a dataset of images.
    
    Args:
        images: Numpy array or tensor of images (N, C, H, W)
        teacher_model: VipTeacher instance
        batch_size: Batch size for processing
        save_path: Optional path to save features
    
    Returns:
        features: Extracted features (N, 2048)
    """
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images)
    
    features = teacher_model.extract_features_batched(images, batch_size)
    
    if save_path is not None:
        torch.save(features, save_path)
        print(f"Features saved to {save_path}")
    
    return features