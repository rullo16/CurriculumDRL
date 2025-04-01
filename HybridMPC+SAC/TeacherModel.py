import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import MobileViTImageProcessor, MobileViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer, student_model, num_samples=10000):
        self.camera_obs = []
        self.student_features = []

        samples = replay_buffer.sample(num_samples)
        for sample in samples["camera_obs"]:
            obs = torch.tensor(sample, dtype=torch.float32).to(device)
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)

            with torch.no_grad():
                feature = student_model(obs).detach().cpu().squeeze(0)

            self.camera_obs.append(obs.squeeze(0).cpu())  # save as 3D again
            self.student_features.append(feature)

    def __len__(self):
        return len(self.camera_obs)
    
    def __getitem__(self, idx):
        return self.camera_obs[idx], self.student_features[idx]


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = MobileViTModel.from_pretrained("SavedModels/Teacher")
        self.head = nn.Linear(640, 12800)
        self.processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small", do_rescale=False)

    def _get_inputs(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.processor(x, return_tensors="pt").pixel_values.to(device)
        return x

    def forward(self, x):
        x = self._get_inputs(x)  # Already on GPU, no repeated .to(device)
        features = self.teacher(x).last_hidden_state
        features = features.mean(dim=[2,3])
        return self.head(features)
    
    def contrastive_loss(self, features, margin=1.0):
        #Ensure different states have distinct representations
        distances = torch.cdist(features, features, p=2)
        loss = torch.mean(torch.relu(margin - distances))
        return loss
    
    def save(self, path):
        self.teacher.save_pretrained(path)

    def load(self, path):
        self.teacher = MobileViTModel.from_pretrained(path)