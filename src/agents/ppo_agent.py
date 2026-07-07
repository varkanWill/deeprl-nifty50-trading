from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from settings.config import (
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    MODELS_DIR,
    LOGS_DIR,
)


class PPOAgent:
    """
    PPO agent wrapper around Stable-Baselines3.

    Monitor wraps the env so SB3 can track episode reward
    and episode length automatically during training.
    """

    def __init__(self, env: gym.Env, run_name: str = "ppo_run_1"):
        self.run_name = run_name

        # Monitor records ep_rew_mean and ep_len_mean for tensorboard
        self.env = Monitor(env)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
        )

        print(f"\nPPO Agent ready")
        print(f"  Run name      : {run_name}")
        print(f"  Learning rate : {PPO_LEARNING_RATE}")
        print(f"  Batch size    : {PPO_BATCH_SIZE}")
        print(f"  N epochs      : {PPO_N_EPOCHS}")
        print(f"  Checkpoints   → {MODELS_DIR / run_name}")
        print(f"  Tensorboard   → {LOGS_DIR}")

    def train(self, total_timesteps: int) -> None:
        """Train with automatic checkpoint saving every 50k steps."""

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODELS_DIR / self.run_name),
            name_prefix="ppo_ckpt",
            verbose=1,
        )

        print(f"\nTraining for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            tb_log_name=self.run_name,
            progress_bar=True,
        )

        print("\nTraining complete.")

    def save(self) -> str:
        """Save the final trained model."""
        save_path = str(MODELS_DIR / f"{self.run_name}_final")
        self.model.save(save_path)
        print(f"Final model saved → {save_path}.zip")
        return save_path

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from: {path}")

    def predict(self, obs, deterministic: bool = True):
        """Get an action from the trained model."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action