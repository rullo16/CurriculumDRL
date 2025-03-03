from types import SimpleNamespace

HYPERPARAMS = {
    'ppo_distilled': SimpleNamespace(**{
        'gamma': 0.99,  # Discount factor for future rewards
        'lam': 0.95,  # GAE (Generalized Advantage Estimation) lambda
        'train_epochs': 10,  # Number of training epochs per update
        'epsilon': 0.2,  # PPO clipping parameter
        'value_loss_coef': 0.5,  # Coefficient for value function loss
        'entropy_coef': 0.01,  # Coefficient for entropy regularization
        'lr': 3e-4,  # Learning rate
        'trajectory_length': 2048,  # Number of steps per trajectory rollout
        'batch_size': 2048,  # Batch size for training
        'mini_batch_size': 64,  # Size of mini-batches for each epoch
        'clip_grad_norm': 0.5,  # Maximum norm for gradient clipping
        'distillation_coef': 0.5,  # Coefficient for distillation loss
        'n_steps': 2048,  # Number of steps per update (should match trajectory_length)
        'n_steps_random_exploration': 100000,  # Steps of random exploration at start
        'max_steps': 1000000,  # Total number of training steps
        'action_std': 0.5,  # Initial standard deviation for action distribution
        'seed_episodes': 5,  # Number of episodes for seeding
    }),
    'sac_distilled': SimpleNamespace(**{
        'gamma': 0.99,
        'tau': 0.005,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'alpha_lr': 3e-4,
        'entropy_lr': 3e-4,
        'distill_lr': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,
        'lambda_': 0.95,
        'target_entropy': -2,
        'train_epochs': 10,
        'seed_episodes': 5,
        'max_steps': 1000000,
        'n_steps': 2048,
        'n_steps_random_exploration': 10000,
    }),
}
