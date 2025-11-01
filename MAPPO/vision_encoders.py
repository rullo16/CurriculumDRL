"""
Replace FeatureExtractionNet with better and more efficient CNN architectures for continuous control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EfficientVisionEncoder(nn.Module):
    """
    Efficient CNN encoder using depthwise separable convolutions.

    From MobileNet and EfficientNet architectures, has fewer params, faster training, better sample efficiency
    and is less prone to overfitting.

    Best for General purpose RL with visual observations

    input_shape: Tuple of (channels, height, width), e.g., (1, 84, 84)
    output_dim: Output feature dimension, default 256
    """

    def __init__(self, input_shape, output_dim=256):
        super().__init__()

        channels, height, width = input_shape

        # Feature extraction - outputs 32 dimensions
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        
        if output_dim != 32:
            self.projection = nn.Sequential(
                nn.Linear(32, output_dim),
                nn.LayerNorm(output_dim),
                nn.Tanh()
            )
        else:
            self.projection = nn.Identity()

        self._init_weights()

    def _depthwise_sep_conv(self, in_channels, out_channels, stride=1):
        """
        Depthwise separable convolution block.

        More efficient than standard conv because (in_channels*kernel_size^2) + (in_channels*out_channels) params
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            #Pointwise: 1x1 conv to combine channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, channels, height, width]

        return features: Output tensor of shape [batch_size, output_dim]
        """

        x = self.features(x)
        x = self.projection(x)
        return x

