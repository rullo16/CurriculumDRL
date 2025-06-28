import torch, torch.nn as nn
from torchvision.transforms import Resize
from torchrl.envs.transforms import VIPTransform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VipTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = Resize(224)
        self.tfm = VIPTransform(in_keys=["pixels"], download=True, model_name="resnet50").eval().to(DEVICE)
        self.out_dim = 2048  # ResNet50 output dimension

    @torch.no_grad()
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() >1.01:
            x = x / 255.0
        x = self.resize(x).repeat(1, 3, 1, 1)
        return self.tfm._vip(x)["vip_vec"]
    
    