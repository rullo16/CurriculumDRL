from types import SimpleNamespace

SAC_DISTILLED = {
    "gamma": 0.99,
    "gae_lambda": 0.95,   # (unused by SAC but harmless if present)
    "tau": 0.01,

    # ── Optimisers ────────────────────────────────────────────────────────
    "actor_lr":   3e-4,
    "critic_lr":  2e-4,
    "alpha_lr":   3e-4,   # ↑ was 1e-4; faster entropy tuning helps early stability
    "distill_lr": 1e-4,   # ↑ was 3e-5; speeds up feature pretraining convergence
    "rnd_lr":     5e-5,

    "critic_updates": 3,
    "actor_updates":  2,

    # ── Distillation ──────────────────────────────────────────────────────
    "distill_coef":   0.06,
    "distill_epochs": 2,
    "distill_batch":  256,
    "distill_temp":   0.07,
    "distill_frames": 50_000,

    # ── Replay buffer ─────────────────────────────────────────────────────
    "buffer_size": 1_000_000,
    "batch_size":  512,   # ↓ was 1024; smaller batch improves update frequency & stability

    # ── Exploration / intrinsic ───────────────────────────────────────────
    "seed_episodes": 1,
    "n_steps_random_exploration": 10_000,
    "noise_std": 0.2,
    "smooth_clip": 0.2,
    "rnd_update_proportion": 0.05,
    "intrinsic_reward_coef_init":        0.5,      # starting weight
    "intrinsic_reward_coef_final":  0.15,      # NEW: target weight after decay
    "intrinsic_coef_decay_steps":   1_000_000,# NEW: linear decay horizon
    "extrinsic_reward_coef":        1.0,

    # ── Training loop ─────────────────────────────────────────────────────
    "max_steps":    5_000_000,
    "policy_delay": 2,
    "ema_alpha":    0.01,

    "curiosity_coeff": 0.1,
    "warmup_steps": 100_000,

    "use_drq": True,
    "drq_pad": 4,
}
HYPERPARAMS = {
    'sac_distilled': SAC_DISTILLED,
}
