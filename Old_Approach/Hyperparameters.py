from types import SimpleNamespace

SAC_DISTILLED = {
    # ── Core RL Parameters ────────────────────────────────────────────────
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "tau": 0.01,

    # ── Optimizers (FIX: Balanced learning rates) ─────────────────────────
    "actor_lr":   2e-4,      # ↓ Reduced for stability
    "critic_lr":  3e-4,      # ↑ Faster critic learning
    "alpha_lr":   3e-4,      # Keep fast entropy tuning
    "distill_lr": 1e-4,      # Maintained for pretraining
    "rnd_lr":     5e-5,      # Lower for stable curiosity

    "critic_updates": 3,     # Multiple critic updates per step
    "actor_updates":  2,     # Fewer actor updates for stability

    # ── Distillation ──────────────────────────────────────────────────────
    "distill_coef":   0.06,
    "distill_epochs": 3,     # ↑ More epochs for better features
    "distill_batch":  256,
    "distill_temp":   0.07,
    "distill_frames": 80_000,  # ↑ More frames for diversity

    # ── Replay Buffer ─────────────────────────────────────────────────────
    "buffer_size": 1_000_000,
    "batch_size":  512,      # Good balance for multi-agent

    # ── Exploration (FIX: Unified reward scaling) ─────────────────────────
    "seed_episodes": 2,      # ↑ More initial exploration
    "n_steps_random_exploration": 15_000,  # ↑ More random steps
    "noise_std": 0.2,
    "smooth_clip": 0.2,
    
    # FIX 1: Simplified intrinsic reward coefficients
    "reward_scale": 0.1,     # NEW: Single scale for all rewards
    "intrinsic_coef_init":  0.3,   # Starting intrinsic weight
    "intrinsic_coef_final": 0.05,  # Final intrinsic weight
    "intrinsic_coef_decay_steps": 2_000_000,  # Linear decay horizon
    
    # FIX 2: Increased RND update proportion
    "rnd_update_proportion": 0.25,  # ↑ From 0.05, reduces variance

    # ── Training Loop ─────────────────────────────────────────────────────
    "max_steps":    5_000_000,
    "policy_delay": 2,
    "ema_alpha":    0.01,

    # FIX 3: Gentler entropy target decay
    "target_entropy_decay_steps": 2_000_000,  # ↑ Longer decay

    # FIX 4: Delayed conv unfreezing
    "warmup_steps": 150_000,  # ↑ From 100k, more stable pretraining

    # FIX 5: Enhanced DrQ augmentation
    "use_drq": True,
    "drq_pad": 4,
    "use_intensity_aug": True,  # NEW: Brightness/contrast augmentation
    
    # ── Additional Improvements ───────────────────────────────────────────
    "gradient_clip_actor": 1.0,
    "gradient_clip_critic": 1.0,
    "gradient_clip_rnd": 1.0,
    "huber_delta": 10.0,     # ↑ Less aggressive clipping
}

MAPPO = {
    'learning_rate': 3e-4,
    'clip_param': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_steps': 2048,
    'num_mini_batches': 8,
    'ppo_epochs': 4,
}

HYPERPARAMS = {
    'sac_distilled': SAC_DISTILLED,
    'mappo': MAPPO
}