from types import SimpleNamespace

SAC_DISTILLED = {
    # ── RL core ───────────────────────────────────────────────────────────
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "tau": 0.005,

    # ── Optimisers ────────────────────────────────────────────────────────
    "actor_lr":   1e-4,
    "critic_lr":  1e-4,
    "alpha_lr": 1e-5,          # single temp optimiser LR
    "distill_lr": 3e-5,
    "rnd_lr": 1e-4,

    "critic_updates": 3,
    "actor_updates":  2,

    # ── Distillation ──────────────────────────────────────────────────────
    "distill_coef":   0.06,
    "distill_epochs": 2,         # new name matches notebook
    "distill_batch":  256,
    "distill_temp":   0.07,
    "distill_frames": 50_000,

    # ── Replay buffer ─────────────────────────────────────────────────────
    "buffer_size": 1_000_000,
    "batch_size":  1_024,

    # ── Exploration ───────────────────────────────────────────────────────
    "seed_episodes": 1,
    "n_steps_random_exploration": 10_000,
    "noise_std": 0.2,
    "smooth_clip": 0.2,
    "rnd_update_proportion":0.5,
    "intrinsic_reward_coef": 1.0,
    "extrinsic_reward_coef": 1.5,

    # ── Training loop ─────────────────────────────────────────────────────
    "max_steps":   5_000_000,
    "policy_delay": 2,
    "ema_alpha":   0.01,

    "curiosity_coeff": 0.1,
    "warmup_steps": 50_000,
}
HYPERPARAMS = {
    'sac_distilled': SAC_DISTILLED,
}
