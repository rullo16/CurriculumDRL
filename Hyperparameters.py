from types import SimpleNamespace

HYPERPARAMS = {
    'ppo_distilled': SimpleNamespace(**{
        'gamma': 0.97,  # Discount factor for future rewards
        'lam': 0.95,  # GAE (Generalized Advantage Estimation) lambda
        'train_epochs': 10,  # Number of training epochs per update
        'epsilon': 0.2,  # PPO clipping parameter
        'value_loss_coef': 0.5,  # Coefficient for value function loss
        'entropy_coef': 0.03,  # Coefficient for entropy regularization
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
        # Core RL parameters
        'gamma': 0.99,                 # Discount factor
        'tau': 0.001,                  # Target network soft update rate
        'lambda_': 0.95,               # GAE-style decay for value smoothing
        
        # Learning rates (tuned per optimizer)
        'actor_lr': 3e-4,
        'critic_lr': 2e-5,             # Slightly higher critic LR helps with stability
        'alpha_lr': 1e-4,              # Slower entropy tuning prevents oscillations
        'entropy_lr': 3e-4,
        'distill_lr': 5e-4,            # Slightly higher distillation rate for fast convergence

        # Distillation behavior
        'distill_coef': 0.75,          # Stronger early distillation influence (decays over time)

        # Replay buffer
        'buffer_size': 300_000,        # Large enough to avoid old data bias, but still memory-safe
        'batch_size': 1024,

        # Entropy
        'target_entropy': None,        # Will be dynamically learned

        # Training loop controls
        'train_epochs': 2,             # Double update per step is usually enough
        'seed_episodes': 5,            # Initial random experience
        'n_steps_random_exploration': 5 * 2048,  # Equals 5 full random episodes


        # Main training loop
        'n_steps': 2048,
        'max_steps': 1_000_000,
    }),
}
