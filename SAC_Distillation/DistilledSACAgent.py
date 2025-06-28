import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import wandb
from .Nets import SACNet, CentralizedCriticNet, ICM
from .VipTeacher import VipTeacher
from .TeacherModel import DistillationDataset, TeacherStudentPairs
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explanation of key math formulas used in DistilledSAC:
# 1. Actor loss:
#    L_actor = -E[log π(a|s) * d] - λ * H[π(a|s)]
# 2. Critic loss:
#    L_critic = E[(Q(s, a) - (r + γ * V(s')))^2]
# 3. Knowledge distillation loss:
#    L_KD = KL( softmax(teacher) || log_softmax(student) )
#

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_proj, teacher_proj):
        s = F.normalize(student_proj, dim=1)
        t = F.normalize(teacher_proj, dim=1)
        logits = (s @ t.T) / self.temperature
        targets = torch.arange(s.size(0), device=s.device)
        return F.cross_entropy(logits, targets)


def _process_vector(vector):
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    else:
        vector = vector.to(device, dtype=torch.float32)
    if len(vector.shape) == 1:
        vector = vector.unsqueeze(0)
    return vector

def _process_image(image):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32, device=device)
    else:
        image = image.to(device, dtype=torch.float32)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return image

def cosine_sigma(init, final, frac):
    return final + (init - final) * 0.5 *(1.0 + np.cos(np.pi * frac))
    
class DistilledSAC:

    def __init__(self, camera_obs_dim, vector_obs_dim, action_dims,num_agents, params):
        self.device = device
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dims = action_dims[0]
        self.critic_lr = params.get('critic_lr', 3e-4)
        self.actor_lr = params.get('actor_lr', 1e-4)
        self.alpha_lr = params.get('alpha_lr', 1e-5)
        self.max_steps = params.get('max_steps', 5_000_000)
        self.distill_lr = params.get('distill_lr', 3e-5)
        self.policy_delay = params.get('policy_delay', 2)
        self.gamma = params.get('gamma', 0.99)
        self.batch_size = params.get('batch_size', 1024)
        self.train_epochs = params.get('train_epochs', 1)
        self.critic_updates = params.get('critic_updates', 3)
        self.actor_updates = params.get('actor_updates', 1)
        self.distill_coef = params.get('distill_coef', 0.06)
        self.init_noise = 0.4
        self.final_noise = 0.05
        self.noise_decay = 0.3

        # Initialize Networks
        self.model = SACNet(camera_obs_dim, vector_obs_dim, self.action_dims).to(self.device)
        
        self.num_agents = num_agents
        print("Number of agents: ", num_agents)
        print("Action dimensions: ", self.action_dims)

        feat_dim = self.model.feat_dim
        self.ccritic = CentralizedCriticNet(feat_dim, self.action_dims, num_agents).to(self.device)

        self.icm = ICM(feat_dim, self.action_dims).to(self.device)
        self.icm_optimizer = optim.AdamW(self.icm.parameters(), lr=1e-4)
        self.curiosity_coef = params.get('curiosity_coef', 0.01)

        self.ccritic_tgt = copy.deepcopy(self.ccritic).to(self.device)
        self.actor_optimizer = optim.AdamW(self.model.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        self.critic_optim = optim.AdamW(self.ccritic.parameters(), lr=self.critic_lr, weight_decay=1e-4)
        self.critic_head_optim = optim.AdamW(self.ccritic.q_heads.parameters(), lr=5*self.critic_lr)
        self.ccritic_tgt.load_state_dict(self.ccritic.state_dict())
        self.ccritic_tgt.eval()

        self.target_entropy = -float(self.action_dims)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.alpha_lr)

        self._distill_dataset = None
        self._distill_loader = None

        self.min_target_entropy = -float(self.action_dims)

        #Teacher and student networks
        self.teacher = VipTeacher().to(self.device)
        self.teacher.eval()
        self.distill_optimizer = optim.Adam(self.model.convolution_pipeline.parameters(), lr=self.distill_lr)
        
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=self.max_steps, eta_min=0.3*self.actor_lr)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optim, T_max=self.max_steps, eta_min=0.3*self.critic_lr)

        self._offline_done = False
        self._conv_unfrozen = False
        self.warmup_steps = getattr(params, 'warmup_steps', 20000)
        self.scaler = torch.amp.GradScaler('cuda',enabled=device.type == 'cuda')

    def _encode(self, cam, vec):
        feat = self.model.encode(cam, vec)
        return feat
    
    @staticmethod
    def _sample(dist):
        z = dist.rsample()
        action = torch.tanh(z)
        log_pi = (dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_pi.sum(dim=-1, keepdim=True)
    
    def get_action(self, camera_obs, vector_obs, train=False):
        camera_obs = _process_image(camera_obs)
        vector_obs = _process_vector(vector_obs)

        feats = self.model.encode(camera_obs, vector_obs)
        dist = self.model.dist_from_feats(feats)
        action, log_prob = self._sample(dist)
        # if add_noise and not train:
        #     frac = step_fraction if step_fraction is not None else 1.0
        #     sigma = cosine_sigma(self.init_noise, self.final_noise, frac)
        #     action = (action + torch.randn_like(action) * sigma).clamp(-1.0, 1.0)
        if train:
            return action, log_prob
        return action
    
    def act(self, cam, vec):
        a = self.get_action(cam, vec, train=False)
        return a
    
    def _exploration_noise(self, step_fraction):
        frac = min(step_fraction / self.noise_decay, 1.0)
        return cosine_sigma(self.init_noise, self.final_noise, frac)

    def _get_teacher_output(self, camera_obs):
        with torch.no_grad():
            teacher_features = self.teacher(camera_obs)
        return teacher_features
    
    def fine_tune_teacher(self,replay_buffer, epochs=10, lr=3e-4, lambda_mse=1.0, lambda_contrast=0.1, margin=1.0):
        student = self.model.convolution_pipeline.eval().to(device)
        for p in student.parameters():
            p.requires_grad = False
        
        replay_buffer = DistillationDataset(replay_buffer, student, num_samples=50000, device=device)
        dataloader = DataLoader(replay_buffer, batch_size=128, shuffle=True, pin_memory=True)

        self.teacher.train().to(device)
        optimizer = optim.Adam(self.teacher.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1,epochs+1):
            for camera_obs, student_feats in dataloader:
                camera_obs = _process_image(camera_obs)
                student_feats = student_feats.to(device).squeeze(1)
                with torch.amp.autocast('cuda',enabled=device.type == 'cuda'):
                    teacher_features = self.teacher(camera_obs)

                    mse_loss = F.mse_loss(teacher_features, student_feats)
                    contrastive_loss = self.teacher.contrastive_loss(teacher_features, margin=margin)

                    total_loss = lambda_mse * mse_loss + lambda_contrast * contrastive_loss

                optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
        
            print("Epoch {} completed".format(epoch))
        print("Teacher model fine-tuning completed")

    def offline_distill(self, frame_buffer, epochs=5, batch_size=128, lr = None, clip_grad=1.0, num_frames=80000, temperature=0.07):
        device = next(self.model.parameters()).device
        lr = lr or self.distill_lr

        if hasattr(frame_buffer, 'camera_obs'):
            cams = frame_buffer.camera_obs[:num_frames]
        else:
            cams = frame_buffer[:num_frames]

        cams = torch.from_numpy(cams)
        cams = cams.float().div(255.0) # Normalize to [0, 1]

        self.teacher.eval().to(device)
        with torch.no_grad():
            batch_teach, emb_list = 128, []
            for i in range(0, len(cams), batch_teach):
                imgs = cams[i:i+batch_teach]
                if imgs.ndim == 3:
                    imgs = imgs.unsqueeze(0)  # Ensure batch dimension
                else:
                    imgs = imgs.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
                emb = self.teacher(imgs.to(device)).float().cpu()
                emb_list.append(emb)
                del imgs, emb
                torch.cuda.empty_cache()
            teacher_embeds = torch.cat(emb_list)

        loader = DataLoader(TeacherStudentPairs(cams, teacher_embeds),
                            batch_size = batch_size,
                            shuffle=True)

        def toggle_grad(module, flag:bool):
            for p in module.parameters():
                p.requires_grad_(flag)

        toggle_grad(self.model, False)
        toggle_grad(self.model.convolution_pipeline, True)

        self.model.eval().to(device)
        scaler = torch.amp.GradScaler('cuda',enabled=device.type == 'cuda')
        opt = optim.AdamW(self.model.convolution_pipeline.parameters(), lr=lr, weight_decay=1e-4)
        criterion = InfoNCELoss(temperature=temperature)

        for ep in range(1,epochs+1):
            running, count  = 0.0, 0
            for imgs, tgt in loader:
                imgs, tgt = imgs.to(device), tgt.to(device)
                with torch.amp.autocast('cuda',enabled=device.type == 'cuda'):
                    std = self.model.convolution_pipeline(imgs, distill=True)
                    loss = criterion(std, tgt)
                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.convolution_pipeline.parameters(), max_norm=clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

                running += loss.item()*imgs.size(0)
                count += imgs.size(0)

            print(f"Epoch {ep} - Distillation Loss: {running/count:.4f}")

        toggle_grad(self.model, True)
        toggle_grad(self.model.convolution_pipeline, False)

        self._offline_done = True
        self._conv_unfrozen = False
        print("Offline distillation completed")


    def _policy_dist(self, cam,vec):
        B,N = cam.shape[:2]
        cam = cam.view(B*N, *cam.shape[2:])
        vec = vec.view(B*N, -1)
        feats = self.model.encode(cam, vec)
        dist = self.model.dist_from_feats(feats)
        return dist, B,N
    
    def train(self, trajectories, step):
        if self._offline_done and not self._conv_unfrozen:
            if step >= self.warmup_steps:
                for p in self.model.convolution_pipeline.parameters():
                    p.requires_grad = True
                if hasattr(self.model.convolution_pipeline, 'distilled_converter'):
                    for p in self.model.convolution_pipeline.distilled_converter.parameters():
                        p.requires_grad = True
                self._conv_unfrozen = True

        if len(trajectories) < self.batch_size * self.num_agents:
            return

        self.model.train()
        step_fraction = 1.0 - np.exp(-step / 5_000_000)

        beta = 0.4 + step_fraction * 0.6

        
        curr_policy_delay = self.policy_delay
            
        tau_min, tau_max = 0.002, 0.01
        curr_tau = tau_min + step_fraction * (tau_max - tau_min)

        actor_loss_l, critic_loss_l = [], []
        for i in range(self.train_epochs):

            sample = trajectories.sample_joint(self.batch_size, alpha=0.6, beta=beta, n_step=3)
            if sample is None:
                return 0.0,0.0
            camera_obs = _process_image(sample["camera_obs"])
            vector_obs = _process_vector(sample["vector_obs"])
            actions = _process_vector(sample["actions"])
            rewards = _process_vector(sample["rewards"])
            done_flags = _process_vector(sample["dones"])
            next_camera_obs = _process_image(sample["next_camera_obs"])
            next_vector_obs = _process_vector(sample["next_vector_obs"])
            weights = sample["weights"]
            weights = torch.as_tensor(sample["weights"],
                          device=self.device,
                          dtype=torch.float32)       # (B, N, 1)  or (B,1)

            if weights.ndim > 2:                                # came from sample_joint
                B, N, _ = weights.shape
                # This is still needed for the actor loss later, which expects a flattened tensor
                weights_flat   = weights.view(B * N, 1)         # Shape: (4096, 1)

                # Squeeze the last dimension of the original weights tensor to get the correct shape
                weights_critic = weights.squeeze(-1)            # Shape: (1024, 4)
            else:                                               # single-agent sample
                weights_flat   = weights                        
                weights_critic = weights.squeeze(-1) # Also squeeze here for consistency

            if camera_obs.dim() > 4:
                B,N = camera_obs.shape[:2]
                camera_obs = camera_obs.view(B*N, *camera_obs.shape[2:])
                next_camera_obs = next_camera_obs.view(B*N, *next_camera_obs.shape[2:])
            if vector_obs.dim() > 2:
                B,N = vector_obs.shape[:2]
                vector_obs = vector_obs.view(B*N, -1)
                next_vector_obs = next_vector_obs.view(B*N, -1)
            if actions.dim() > 2:
                B,N = actions.shape[:2]
                actions = actions.view(B*N, -1)

            with torch.no_grad():
                feats_next = self._encode(next_camera_obs, next_vector_obs)
            feats_now = self._encode(camera_obs, vector_obs)

            self.icm_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=device.type == 'cuda'):
                intrinsic_rewards, fwd_loss, inv_loss = self.icm(feats_now.detach(), feats_next.detach(), actions)
                icm_loss = (0.8 * fwd_loss) + (0.2 * inv_loss)

            self.scaler.scale(icm_loss).backward(retain_graph=True)
            self.scaler.unscale_(self.icm_optimizer)
            


            # --- Critic Training ---

            with torch.no_grad():
                next_action = self.get_action(next_camera_obs, next_vector_obs, train=False)
                
                sigma = torch.tensor(self._exploration_noise(step_fraction), device=device, dtype=next_action.dtype)
                eps = torch.randn_like(next_action) * sigma
                next_action = (next_action + eps).clamp(-1.0, 1.0)

                q_next = self.ccritic_tgt(feats_next, next_action)

                r_env = rewards.view(-1,1)
                done_env = done_flags.view(-1,1).float()

                decay_factor = max(0.0, 1.0 - step_fraction)
                self.current_curiosity_coef = self.curiosity_coef * decay_factor

                total_reward = r_env + self.current_curiosity_coef * intrinsic_rewards.detach()
                total_reward_reshaped = total_reward.view(B,N)
                done_reshaped = done_env.view(B,N)

                td_target = total_reward_reshaped + (1.0 - done_reshaped) * self.gamma * q_next
            
            for _ in range(self.critic_updates):
                self.critic_optim.zero_grad(set_to_none=True)
                self.critic_head_optim.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=device.type == 'cuda'):
                    q_pred = self.ccritic(feats_now.detach(), actions)
                    critic_loss = (weights_critic * F.huber_loss(q_pred, td_target, delta=10.0, reduction='none')).mean()

                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_optim)
                self.scaler.unscale_(self.critic_head_optim)
                torch.nn.utils.clip_grad_norm_(self.ccritic.parameters(), max_norm=1.0)
                scale = self.scaler.get_scale()
                self.scaler.step(self.critic_optim)
                self.scaler.step(self.critic_head_optim)
                self.scaler.update()

                if not scale != self.scaler.get_scale():
                    self.critic_scheduler.step()

                with torch.no_grad():
                    for p_t, p in zip(self.ccritic_tgt.parameters(), self.ccritic.parameters()):
                        p_t.data.lerp_(p.data, curr_tau)

            critic_loss_l.append(critic_loss.item())

            # --- Actor Training ---
            if i % curr_policy_delay == 0:


                for _ in range(self.actor_updates):
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda',enabled=device.type == 'cuda'):
                        new_a, logp = self.get_action(camera_obs, vector_obs, train=True)
                        q_new = self.ccritic(feats_now, new_a)
                        q_new_min = q_new.min(dim=1, keepdim=True).values
                        q_new_min_exp = q_new_min.repeat(1,N).view(B*N,1)
                        actor_loss = (weights_flat *(self.log_alpha.exp().detach()*logp-q_new_min_exp)).mean()
                        actor_loss_l.append(actor_loss.item())

                    self.scaler.scale(actor_loss).backward()
                    scale = self.scaler.get_scale()
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.update()
                    if not scale != self.scaler.get_scale():
                        self.actor_scheduler.step()
                
                
                # # --- Entropy Regularization ---
                entropy_err = (logp + self.target_entropy).detach()
                entropy_loss = -(weights_flat * self.log_alpha * entropy_err).mean()

                self.alpha_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.alpha_optimizer)
                self.scaler.update()

                with torch.no_grad():
                    self.log_alpha.clamp_(min=np.log(1e-5), max=np.log(1e2))
            
            td_err = (q_pred - td_target).abs()
            td_err_per_env = td_err.max(dim=1).values
            priorities = td_err_per_env.repeat_interleave(self.num_agents)
            priorities = priorities.detach().clamp(min=1e-3).cpu().numpy()
            flat_idx = sample['indices'].reshape(-1)

            trajectories.update_priorities(flat_idx, priorities)
        
        if step_fraction > 0.6:
            progress = step /step_fraction
            cosine = 0.5 * (1+np.cos(np.pi * progress))
            sched = self.min_target_entropy + cosine * (-float(self.action_dims)-self.min_target_entropy)
            self.target_entropy = sched
        # Record all losses for logging
        # Return meaningful average loss values clearly
        return (
            np.mean(actor_loss_l) if actor_loss_l else 0.0,
            np.mean(critic_loss_l) if critic_loss_l else 0.0,
            intrinsic_rewards.mean().item(),
            icm_loss.item()
        )


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

