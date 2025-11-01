import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import wandb
from .Nets import SACNet, CentralizedCriticNet, RND, RunningStat
from .VipTeacher import VipTeacher
from .TeacherModel import DistillationDataset, TeacherStudentPairs
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image = torch.tensor(image, device=device)
    else:
        image = image.to(device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
    return image


def cosine_sigma(init, final, frac):
    return final + (init - final) * 0.5 *(1.0 + np.cos(np.pi * frac))
    
class DistilledSAC:

    def __init__(self, camera_obs_dim, vector_obs_dim, action_dims, num_agents, params):
        self.device = device
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dims = action_dims[0]
        self.critic_lr = params.get('critic_lr', 3e-4)
        self.actor_lr = params.get('actor_lr', 1e-4)
        self.alpha_lr = params.get('alpha_lr', 3e-4)
        self.max_steps = params.get('max_steps', 5_000_000)
        self.distill_lr = params.get('distill_lr', 1e-4)
        self.policy_delay = params.get('policy_delay', 2)
        self.gamma = params.get('gamma', 0.99)
        self.batch_size = params.get('batch_size', 512)
        self.train_epochs = params.get('train_epochs', 1)
        self.critic_updates = params.get('critic_updates', 3)
        self.actor_updates = params.get('actor_updates', 2)
        self.distill_coef = params.get('distill_coef', 0.06)
        
        # FIX 1: Unified reward scaling - use single coefficient
        self.reward_scale = params.get('reward_scale', 0.1)
        self.intrinsic_coef_init = params.get('intrinsic_coef_init', 0.3)
        self.intrinsic_coef_final = params.get('intrinsic_coef_final', 0.05)
        self.intrinsic_decay_steps = params.get('intrinsic_coef_decay_steps', 2_000_000)
        
        self.init_noise = 0.3
        self.final_noise = 0.05
        self.noise_decay = 0.5

        self.reward_stat = None

        # Initialize Networks
        self.model = SACNet(camera_obs_dim, vector_obs_dim, self.action_dims).to(self.device)
        self.num_agents = num_agents
        feat_dim = self.model.feat_dim
        self.ccritic = CentralizedCriticNet(feat_dim, self.action_dims, num_agents).to(self.device)
        self.ccritic_tgt = copy.deepcopy(self.ccritic).to(self.device)

        # FIX 2: Increased RND update proportion for stability
        self.rnd_lr = params.get('rnd_lr', 5e-5)
        self.rnd_update_proportion = params.get('rnd_update_proportion', 0.25)
        self.rnd = RND(input_dim=feat_dim).to(self.device)
        self.rnd_optimizer = optim.AdamW(self.rnd.parameters(), lr=self.rnd_lr, weight_decay=1e-5)
        
        self.intrinsic_reward_normalizer = RunningStat(shape=(1,))

        self.actor_optimizer = optim.AdamW(self.model.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        head_params = list(self.ccritic.q_head_1.parameters()) + list(self.ccritic.q_head_2.parameters())
        base_params = [p for p in self.ccritic.parameters() if id(p) not in {id(param) for param in head_params}]
        self.critic_optim = optim.AdamW(base_params, lr=self.critic_lr, weight_decay=1e-4)
        self.critic_head_optim = optim.AdamW(head_params, lr=self.critic_lr * 1.5, weight_decay=1e-4)
        
        # FIX 3: Gentler entropy target range
        self.max_target_entropy = -float(self.action_dims * 0.5)
        self.min_target_entropy = -float(self.action_dims * 0.2)
        self.ent_decay = params.get("target_entropy_decay_steps", 2_000_000)
        self.target_entropy = self.max_target_entropy

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.alpha_lr)
        self._alpha_min = np.log(1e-4)
        self._alpha_max = np.log(5.0)

        self._distill_dataset = None
        self._distill_loader = None

        self.teacher = VipTeacher().to(self.device)
        self.teacher.eval()
        self.distill_optimizer = optim.Adam(self.model.convolution_pipeline.parameters(), lr=self.distill_lr)
        
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.max_steps, eta_min=0.3*self.actor_lr
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optim, T_max=self.max_steps, eta_min=0.3*self.critic_lr
        )

        self._offline_done = False
        self._conv_unfrozen = False
        # FIX 4: Delay unfreezing to allow more stable learning
        self.warmup_steps = params.get('warmup_steps', 150_000)
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

        # FIX 5: Enhanced DrQ with intensity augmentation
        self.use_drq = params.get('use_drq', True)
        self.drq_pad = params.get('drq_pad', 4)
        self.use_intensity_aug = params.get('use_intensity_aug', True)


    def _random_shift(self, x, pad=4):
        if pad <= 0:
            return x
        b,c,h,w = x.shape
        x = F.pad(x, (pad,pad,pad,pad), mode='replicate')
        eps = 2*pad + 1
        crops = []
        for i in range(b):
            ox = int(torch.randint(0, eps, (1,), device=x.device).item())
            oy = int(torch.randint(0, eps, (1,), device=x.device).item())
            crops.append(x[i:i+1, :, oy:oy+h, ox:ox+w])
        return torch.cat(crops, dim=0)

    def _intensity_aug(self, x):
        """Apply random brightness/contrast for robustness"""
        if not self.training or not self.use_intensity_aug:
            return x
        brightness = torch.rand(1, device=x.device) * 0.4 + 0.8  # [0.8, 1.2]
        contrast = torch.rand(1, device=x.device) * 0.4 + 0.8
        x = x * contrast + (brightness - 1.0) * 0.5
        return x.clamp(0, 1)

    def _encode(self, cam, vec):
        feat = self.model.encode(cam, vec)
        return feat
    
    @staticmethod
    def _sample(dist):
        z = dist.rsample()
        action = torch.tanh(z)
        log_pi = (dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_pi
    
    def get_action(self, camera_obs, vector_obs, train=False):
        camera_obs = _process_image(camera_obs)
        vector_obs = _process_vector(vector_obs)

        feats = self.model.encode(camera_obs, vector_obs)
        dist = self.model.dist_from_feats(feats)
        action, log_prob = self._sample(dist)
        
        if train:
            return action, log_prob
        return action
    
    def act(self, cam, vec):
        with torch.no_grad():
            return self.get_action(cam, vec, train=False)
    

    def _get_teacher_output(self, camera_obs):
        with torch.no_grad():
            teacher_features = self.teacher(camera_obs)
        return teacher_features
    
    def fine_tune_teacher(self, replay_buffer, epochs=10, lr=3e-4, lambda_mse=1.0, lambda_contrast=0.1, margin=1.0):
        student = self.model.convolution_pipeline.eval().to(device)
        for p in student.parameters():
            p.requires_grad = False
        
        replay_buffer = DistillationDataset(replay_buffer, student, num_samples=50000, device=device)
        dataloader = DataLoader(replay_buffer, batch_size=128, shuffle=True, pin_memory=True)

        self.teacher.train().to(device)
        optimizer = optim.Adam(self.teacher.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1, epochs+1):
            for camera_obs, student_feats in dataloader:
                camera_obs = _process_image(camera_obs)
                student_feats = student_feats.to(device).squeeze(1)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    teacher_features = self.teacher(camera_obs)
                    mse_loss = F.mse_loss(teacher_features, student_feats)
                    contrastive_loss = self.teacher.contrastive_loss(teacher_features, margin=margin)
                    total_loss = lambda_mse * mse_loss + lambda_contrast * contrastive_loss

                optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
        
            print(f"Epoch {epoch} completed")
        print("Teacher model fine-tuning completed")

    def offline_distill(self, frame_buffer, epochs=5, batch_size=128, lr=None, clip_grad=1.0, num_frames=80000, temperature=0.07):
        device = next(self.model.parameters()).device
        lr = lr or self.distill_lr

        if hasattr(frame_buffer, 'camera_obs'):
            cams = frame_buffer.camera_obs[:num_frames]
        else:
            cams = frame_buffer[:num_frames]

        cams = torch.from_numpy(cams)
        cams = cams.float().div(255.0)

        self.teacher.eval().to(device)
        with torch.no_grad():
            batch_teach, emb_list = 128, []
            for i in range(0, len(cams), batch_teach):
                imgs = cams[i:i+batch_teach]
                if imgs.ndim == 3:
                    imgs = imgs.unsqueeze(0)
                else:
                    imgs = imgs.permute(0, 3, 1, 2)
                emb = self.teacher(imgs.to(device)).float().cpu()
                emb_list.append(emb)
                del imgs, emb
                torch.cuda.empty_cache()
            teacher_embeds = torch.cat(emb_list)

        loader = DataLoader(
            TeacherStudentPairs(cams, teacher_embeds),
            batch_size=batch_size,
            shuffle=True
        )

        def toggle_grad(module, flag: bool):
            for p in module.parameters():
                p.requires_grad_(flag)

        toggle_grad(self.model, False)
        toggle_grad(self.model.convolution_pipeline, True)

        self.model.eval().to(device)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        opt = optim.AdamW(self.model.convolution_pipeline.parameters(), lr=lr, weight_decay=1e-4)
        criterion = InfoNCELoss(temperature=temperature)

        for ep in range(1, epochs+1):
            running, count = 0.0, 0
            for imgs, tgt in loader:
                imgs, tgt = imgs.to(device), tgt.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    std = self.model.convolution_pipeline(imgs, distill=True)
                    loss = criterion(std, tgt)
                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.convolution_pipeline.parameters(), max_norm=clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                running += loss.item() * imgs.size(0)
                count += imgs.size(0)

            print(f"Epoch {ep} - Distillation Loss: {running/count:.4f}")

        toggle_grad(self.model, True)
        toggle_grad(self.model.convolution_pipeline, False)

        self._offline_done = True
        self._conv_unfrozen = False
        print("Offline distillation completed")

    def _policy_dist(self, cam, vec):
        B, N = cam.shape[:2]
        cam = cam.view(B*N, *cam.shape[2:])
        vec = vec.view(B*N, -1)
        feats = self.model.encode(cam, vec)
        dist = self.model.dist_from_feats(feats)
        return dist, B, N
    
    def train(self, trajectories, step):
        # FIX 4: Gradual unfreezing at higher warmup
        if self._offline_done and not self._conv_unfrozen:
            if step >= self.warmup_steps:
                for p in self.model.convolution_pipeline.parameters():
                    p.requires_grad = True
                
                self.actor_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                   lr = self.actor_lr, weight_decay=1e-4)
                if hasattr(self.model.convolution_pipeline, 'distilled_converter'):
                    for p in self.model.convolution_pipeline.distilled_converter.parameters():
                        p.requires_grad = True
                self._conv_unfrozen = True
                print(f"[INFO] Unfroze conv layers at step {step}")

        if step < self.warmup_steps:
            return 0.0, 0.0, 0.0, 0.0

        self.model.train()

        # FIX 1: Simplified intrinsic coefficient decay
        t_int = min(1.0, float(step) / float(self.intrinsic_decay_steps))
        intrinsic_coef = self.intrinsic_coef_init * (1.0 - t_int) + self.intrinsic_coef_final * t_int

        # FIX 3: Gentler entropy decay with linear interpolation
        t_ent = min(1.0, float(step) / float(self.ent_decay))
        self.target_entropy = self.max_target_entropy * (1.0 - t_ent) + self.min_target_entropy * t_ent

        step_fraction = min(1.0, step / self.max_steps)
        beta = 0.4 + step_fraction * 0.6

        curr_policy_delay = self.policy_delay
        tau_min, tau_max = 0.003, 0.01
        curr_tau = tau_min + step_fraction * (tau_max - tau_min)

        actor_loss_l, critic_loss_l = [], []

        if self.reward_stat is None:
            self.reward_stat = trajectories.reward_stat

        for i in range(self.train_epochs):
            sample = trajectories.sample_joint(self.batch_size, alpha=0.6, beta=beta, n_step=3)
            if sample is None:
                return 0.0, 0.0, 0.0, 0.0
            
            camera_obs = _process_image(sample["camera_obs"])
            vector_obs = _process_vector(sample["vector_obs"])
            actions = _process_vector(sample["actions"])
            rewards = _process_vector(sample["rewards"])
            done_flags = _process_vector(sample["dones"])
            next_camera_obs = _process_image(sample["next_camera_obs"])
            next_vector_obs = _process_vector(sample["next_vector_obs"])
            weights = torch.as_tensor(sample["weights"], device=self.device, dtype=torch.float32)

            if weights.ndim > 2:
                B, N, _ = weights.shape
                weights_flat = weights.reshape(B*N, 1)
                weights_critic = weights.squeeze(-1)
            else:
                weights_flat = weights
                weights_critic = weights.squeeze(-1)

            if camera_obs.dim() > 4:
                B, N = camera_obs.shape[:2]
                camera_obs = camera_obs.view(B*N, *camera_obs.shape[2:])
                next_camera_obs = next_camera_obs.view(B*N, *next_camera_obs.shape[2:])
            if vector_obs.dim() > 2:
                B, N = vector_obs.shape[:2]
                vector_obs = vector_obs.view(B*N, -1)
                next_vector_obs = next_vector_obs.view(B*N, -1)
            if actions.dim() > 2:
                B, N = actions.shape[:2]
                actions = actions.view(B*N, -1)

            # FIX 5: Apply augmentations
            if self.use_drq:
                camera_obs = self._random_shift(camera_obs, self.drq_pad)
                camera_obs = self._intensity_aug(camera_obs)
                next_camera_obs = self._random_shift(next_camera_obs, self.drq_pad)

            with torch.no_grad():
                feats_next = self._encode(next_camera_obs, next_vector_obs)
                if feats_next.dim() > 2:
                    feats_next = feats_next.reshape(-1, feats_next.shape[-1])
            
            feats_now_c = self._encode(camera_obs, vector_obs)
            if feats_now_c.dim() > 2:
                feats_now_c = feats_now_c.reshape(-1, feats_now_c.shape[-1])

            weights_flat = weights_flat.detach()
            weights_critic = weights_critic.detach()
            
            # FIX 2: Train RND on more samples
            self.rnd_optimizer.zero_grad(set_to_none=True)
            flat_bs = feats_now_c.shape[0]
            num_updates = int(flat_bs * self.rnd_update_proportion)
            perm = torch.randperm(flat_bs, device=feats_next.device)[:num_updates]
            
            rnd_loss = self.rnd.compute_loss(feats_next[perm])
            r_intrinsic = self.rnd(feats_next, normalise_reward=False)
            self.rnd.update_obs_stats(feats_next)
            
            self.scaler.scale(rnd_loss).backward()
            self.scaler.unscale_(self.rnd_optimizer)
            torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), max_norm=1.0)
            self.scaler.step(self.rnd_optimizer)
            
            BN = camera_obs.shape[0]
            N = self.num_agents
            assert BN % N == 0, f"Flat batch {BN} not divisible by num_agents {N}"
            B = BN // N

            # --- Critic Training ---
            with torch.no_grad():
                next_action, next_logp = self.get_action(next_camera_obs, next_vector_obs, train=True)
            next_logp = next_logp.view(BN, 1)

            q1_next, q2_next = self.ccritic_tgt(feats_next, next_action)
            if q1_next.dim() == 1:
                q1_next = q1_next.unsqueeze(1)
            if q2_next.dim() == 1:
                q2_next = q2_next.unsqueeze(1)

            if q1_next.shape[0] == BN:
                q1_next = q1_next.view(B, N, -1)
                q2_next = q2_next.view(B, N, -1)

            
            min_q_next = torch.min(q1_next, q2_next)
            next_logp = next_logp.view(B,N,1)
            alpha = self.log_alpha.exp()
            soft_q_target = min_q_next - alpha * next_logp
            soft_q_target_joint = soft_q_target.sum(dim=1)

            r_env = rewards.view(BN, 1)
            done_e = done_flags.view(BN, 1).float()

            # FIX 1: Unified reward scaling
            mean_r = torch.from_numpy(self.reward_stat.mean).to(self.device)
            std_r = torch.from_numpy(self.reward_stat.std).to(self.device)
            # r_env_norm = (r_env.view(B, N, 1) - mean_r) / (std_r + 1e-8)

            # Normalize intrinsic rewards consistently
            self.intrinsic_reward_normalizer.update(r_intrinsic)
            m_i = torch.from_numpy(self.intrinsic_reward_normalizer.mean).to(self.device)
            s_i = torch.from_numpy(self.intrinsic_reward_normalizer.std).to(self.device)
            r_int_norm = (r_intrinsic.view(B, N, 1) - m_i) / (s_i + 1e-8)
            r_int_norm = torch.clamp(r_int_norm, min=0.0)

            r_env_joint = r_env.sum(dim=1)
            r_int_joint = r_int_norm.sum(dim=1)
            
            # Apply single reward scale
            total_reward = self.reward_scale * (r_env_joint + intrinsic_coef * r_int_joint)

            if "nstep_returns" in sample and "nstep_next_idxs" in sample:
                nstep_env = torch.tensor(sample["nstep_returns"], device=self.device)
                if nstep_env.dim() > 2:
                    nstep_env = nstep_env.sum(dim=1)
                total_reward = self.reward_scale * ((nstep_env - mean_r) / (std_r + 1e-8) + intrinsic_coef * r_int_joint)

            done_joint = done_e.view(B, N, 1).max(dim=1).values
            td_target = total_reward + (1.0 - done_joint) * self.gamma * soft_q_target_joint
            td_target = td_target.clamp(min=-100.0, max=100.0)
            
            self.critic_optim.zero_grad(set_to_none=True)
            self.critic_head_optim.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                q1_pred, q2_pred = self.ccritic(feats_now_c, actions)
                if q1_pred.dim() == 1:
                    q1_pred = q1_pred.unsqueeze(1)
                if q2_pred.dim() == 1:
                    q2_pred = q2_pred.unsqueeze(1)

                if q1_pred.shape[0] == BN:
                    q1_pred = q1_pred.view(B, N, -1).mean(dim=1)
                    q2_pred = q2_pred.view(B, N, -1).mean(dim=1)
                elif q1_pred.shape[0] != B:
                    raise RuntimeError(f"Unexpected critic shape: {q1_pred.shape}")

                weights_joint = weights_critic.view(B, N).mean(dim=1, keepdim=True).detach()

                critic_loss_1 = (weights_joint * F.huber_loss(q1_pred, td_target.detach(), delta=10.0, reduction='none')).mean()
                critic_loss_2 = (weights_joint * F.huber_loss(q2_pred, td_target.detach(), delta=10.0, reduction='none')).mean()
                critic_loss = critic_loss_1 + critic_loss_2

            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optim)
            self.scaler.unscale_(self.critic_head_optim)
            torch.nn.utils.clip_grad_norm_(self.ccritic.parameters(), max_norm=1.0)
            self.scaler.step(self.critic_optim)
            self.scaler.step(self.critic_head_optim)

            critic_loss_l.append(critic_loss.item())

            # --- Actor Training ---
            if i % curr_policy_delay == 0:
                for p in self.ccritic.parameters():
                    p.requires_grad_(False)

                feats_now_a = self._encode(camera_obs, vector_obs)
                if feats_now_a.dim() > 2:
                    feats_now_a = feats_now_a.reshape(-1, feats_now_a.shape[-1])
                
                self.actor_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    new_a, logp = self.get_action(camera_obs, vector_obs, train=True)
                    q1_new, q2_new = self.ccritic(feats_now_a, new_a)
                    if q1_new.dim() == 1:
                        q1_new = q1_new.unsqueeze(1)
                    if q2_new.dim() == 1:
                        q2_new = q2_new.unsqueeze(1)
                    q_new = torch.min(q1_new, q2_new)

                    if q_new.shape[0] == B:
                        q_new = q_new.repeat_interleave(N, dim=0)
                    elif q_new.shape[0] != BN:
                        raise RuntimeError(f"Unexpected actor critic eval shape: {q_new.shape}")

                    logp = logp.view(BN, 1)
                    weights_flat = weights_flat.view(BN, 1).detach()

                    actor_loss = (weights_flat * (self.log_alpha.exp().detach() * logp - q_new)).mean()
                    act_reg = 1e-4 * (new_a.pow(2).mean())
                    actor_loss = actor_loss + act_reg
                
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.actor_optimizer)

                for p in self.ccritic.parameters():
                    p.requires_grad_(True)

                actor_loss_l.append(actor_loss.item())

                # Alpha (temperature) update
                self.alpha_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    _, logp_temp = self.get_action(camera_obs, vector_obs, train=True)
                    entropy_err = (logp_temp.detach() + self.target_entropy)
                    alpha_loss = -(self.log_alpha * entropy_err).mean()
                
                self.scaler.scale(alpha_loss).backward()
                self.scaler.unscale_(self.alpha_optimizer)
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                self.scaler.step(self.alpha_optimizer)
                self.actor_scheduler.step()

            # Soft target update
            with torch.no_grad():
                for p_t, p in zip(self.ccritic_tgt.parameters(), self.ccritic.parameters()):
                    p_t.data.lerp_(p.data, curr_tau)
                self.log_alpha.clamp_(min=self._alpha_min, max=self._alpha_max)

            self.scaler.update()
            self.critic_scheduler.step()
            
            # FIX 6: Improved PER priority update
            with torch.no_grad():
                td_err_1 = (q1_pred - td_target).abs()
                td_err_2 = (q2_pred - td_target).abs()
                per_td = torch.max(td_err_1, td_err_2).reshape(-1)
                
                idx = sample["indices"]
                assert idx.ndim == 2, f"indices must be (B,N), got {idx.shape}"
                B_idx, N_idx = idx.shape

                if per_td.numel() == B_idx * N_idx:
                    per_td_joint = per_td.view(B_idx, N_idx).max(dim=1).values
                elif per_td.numel() == B_idx:
                    per_td_joint = per_td
                else:
                    per_td_joint = per_td[:B_idx]
                
                # Add small constant to prevent zero priorities
                per_td_joint = per_td_joint + 1e-6
                idx_flat = idx.reshape(-1)
                priorities_repeated = per_td_joint.repeat_interleave(N)

                trajectories.update_priorities(
                    idx_flat,
                    priorities_repeated
                )

        return (
            np.mean(actor_loss_l) if actor_loss_l else 0.0,
            np.mean(critic_loss_l) if critic_loss_l else 0.0,
            r_intrinsic.mean().item(),
            rnd_loss.item()
        )

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'ccritic': self.ccritic.state_dict(),
            'ccritic_tgt': self.ccritic_tgt.state_dict(),
            'rnd': self.rnd.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if 'ccritic' in checkpoint:
            self.ccritic.load_state_dict(checkpoint['ccritic'])
            self.ccritic_tgt.load_state_dict(checkpoint['ccritic_tgt'])
        if 'rnd' in checkpoint:
            self.rnd.load_state_dict(checkpoint['rnd'])
        if 'log_alpha' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha'].data