from types import SimpleNamespace

HYPERPARAMS = {
    'ppo_distilled': SimpleNamespace(**{
        'stop_reward': 200.0,
        'gamma': 0.907,
        'lam': 0.95,
        'train_epochs': 10,
        'epsilon': 0.2,
        'value_loss_coef': 0.2,
        'entropy_coef': 0.04,
        'lr': 2e-5,
        'trajectory_length': 2048,
        'batch_size': 64,
        'mini_batch_size': 64,
        'clip_grad_norm': 0.3,
        'distillation_coef': 0.5,
        'n_steps': 2048,
        'n_steps_random_exploration': 10000,
        'max_steps': 1000000,
        'action_std': 0.3,
        'seed_episodes': 5,
    }),
    'sac_distilled': SimpleNamespace(**{
        'stop_reward': 200.0,
        'lambda_factor': 0.95,
        'gamma': 0.907,
        'tau': 0.005,                       # soft update coefficient
        'buffer_size': 1000000,             # replay buffer size
        'batch_size': 256,
        'actor_lr': 3e-4,                   # learning rate for the actor
        'critic_lr': 3e-4,                  # learning rate for the critic
        'alpha_lr': 3e-4,                   # learning rate for entropy coefficient
        'distill_lr': 3e-4,                 # learning rate for the distilled policy
        'automatic_entropy_tuning': True,   # enable automatic entropy tuning
        'target_update_interval': 1,
        'train_iterations': 1,
        'n_steps': 2048,
        'n_steps_random_exploration': 10000,
        # 'max_steps': 1000000,
        'max_steps': 10000,
        # 'seed_episodes': 5,
        'seed_episodes': 2,
        'batch_size': 64,
        'mini_batch_size': 64,
        'train_epochs': 10,
    }),
}
