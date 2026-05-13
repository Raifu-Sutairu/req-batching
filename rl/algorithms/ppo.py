from stable_baselines3 import PPO

def create_ppo_model(env, tensorboard_log="./ppo_batch_tensorboard/"):
    # As per design document: 2 hidden layers of 64, no shared layers
    # SB3's default MLP does not use BatchNorm, so we are safe from batch statistic issues
    # at inference time.
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Critical for preventing collapse to always-wait or always-flush
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    return model
