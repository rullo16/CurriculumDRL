from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import MobileViTImageProcessor, MobileViTModel
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer, student_model, num_samples=10000, encode_batch=512, device=device):
        super().__init__()
        buf_size= replay_buffer.size
        idx = np.random.choice(buf_size, size=min(num_samples, buf_size), replace=False)
        cams = replay_buffer.camera_obs[idx]

        cams = cams.astype(np.float32) / 255.0  # Normalize to [0, 1]
        self.camera_obs = torch.from_numpy(cams)

        student_model.eval().to(device)
        feats = []
        with torch.no_grad():
            for i in range(0, len(self.camera_obs), encode_batch):
                batch = self.camera_obs[i:i+encode_batch].to(device)
                feat = student_model(batch, distill=True).cpu()
                feats.append(feat)
        self.student_feats = torch.cat(feats, dim=0)

        assert len(self.camera_obs) == len(self.student_feats), "Mismatch in dataset lengths"

    def __len__(self):
        return len(self.camera_obs)
    def __getitem__(self, idx):
        return self.camera_obs[idx], self.student_feats[idx]
    
class TeacherStudentPairs(torch.utils.data.Dataset):
    def __init__(self, cams, embeds, device = "cuda"):
        self.cams = cams
        self.embeds = embeds

    def __len__(self):
        return len(self.cams)
    
    def __getitem__(self, idx):
        cam = self.cams[idx]
        embed = self.embeds[idx]
        return cam, embed

class TeacherModel(nn.Module):
    def __init__(self, path = "apple/mobilevit-small", proj_dim=12_800):
        super().__init__()
        self.teacher = MobileViTModel.from_pretrained(path)
        self.head = nn.Linear(640, proj_dim)
        self.processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small", do_rescale=False)

    def _get_inputs(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0,1] already, any dtype
        pv = self.processor(                       # Mobile-ViT processor
            images=x.to("cpu"),                    # keep on host RAM
            return_tensors="pt",
            do_rescale=False,
        ).pixel_values                             # [B, 3, H, W] fp32
        return pv.float().to(device, non_blocking=True)    # send once to GPU

    def forward(self, x, no_grad=True):
        imgs = self._get_inputs(x)  # Already on GPU, no repeated .to(device)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = self.teacher(imgs).last_hidden_state
            feats = out.mean(dim=[2,3])
        return self.head(feats)
    
    def contrastive_loss(self, features, margin=1.0):
        #Ensure different states have distinct representations
        distances = torch.cdist(features, features, p=2)
        loss = torch.mean(torch.relu(margin - distances))
        return loss
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.teacher.state_dict(), path / "teacher_head.pth")
        self.teacher.save_pretrained(path/"backbone")
        
    def load(self, path):
        path = Path(path)
        self.teacher = MobileViTModel.from_pretrained(path/"backbone")
        self.load_state_dict(torch.load(path / "teacher_head.pth"))