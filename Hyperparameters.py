from types import SimpleNamespace

SAC_DISTILLED = {
    # ── RL core ───────────────────────────────────────────────────────────
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "tau": 0.001,

    # ── Optimisers ────────────────────────────────────────────────────────
    "actor_lr":   2e-4,
    "critic_lr":  3e-4,
    "entropy_lr": 3e-5,          # single temp optimiser LR
    "distill_lr": 3e-5,

    "critic_updates": 3,
    "actor_updates":  1,

    # ── Distillation ──────────────────────────────────────────────────────
    "distill_coef":   0.06,
    "distill_epochs": 2,         # new name matches notebook
    "distill_batch":  256,
    "distill_temp":   0.07,
    "distill_frames": 20_000,

    # ── Replay buffer ─────────────────────────────────────────────────────
    "buffer_size": 400_000,
    "batch_size":  1_024,

    # ── Exploration ───────────────────────────────────────────────────────
    "seed_episodes":            5,
    "n_steps_random_exploration": 5 * 1_024,
    "noise_std": 0.2,
    "smooth_clip": 0.2,

    # ── Training loop ─────────────────────────────────────────────────────
    "max_steps":   5_000_000,
    "policy_delay": 2,
    "ema_alpha":   0.01,
}
HYPERPARAMS = {
    'sac_distilled': SAC_DISTILLED,
}
