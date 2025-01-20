from types import SimpleNamespace

HYPERPARAMS = {
    'ppo_distilled': SimpleNamespace(**{
        'stop_reward': 200.0,
        'gamma': 0.99,
        'lam': 0.95,
        'train_epochs': 10,
        'epsilon': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'lr': 2.5e-4,
        'trajectory_length': 2048,
        'batch_size': 64,
        'mini_batch_size': 64,
        'clip_grad_norm': 0.5,
        'distillation_coef': 0.5,
        'n_steps': 2048,
        'n_steps_random_exploration': 10000,
        'max_steps': 1000000,
    }
    ),
}