from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from settings.config import (
    DDPG_LEARNING_RATE,
    DDPG_BUFFER_SIZE,
    DDPG_LEARNING_STARTS,
    DDPG_BATCH_SIZE,
    DDPG_TAU,
    DDPG_GAMMA,
    MODELS_DIR_DDPG,
    LOGS_DIR,
    DEVICE,
)


class DDPGAgent:

    def __init__(self, env: gym.Env, run_name: str = "ddpg_run_1"):
        self.run_name = run_name
        self.env = Monitor(env)

        MODELS_DIR_DDPG.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.model = DDPG(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=DDPG_LEARNING_RATE,
            buffer_size=DDPG_BUFFER_SIZE,
            learning_starts=DDPG_LEARNING_STARTS,
            batch_size=DDPG_BATCH_SIZE,
            tau=DDPG_TAU,
            gamma=DDPG_GAMMA,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
            device=DEVICE,
        )

        print(f"\nDDPG Agent ready")
        print(f"  Run name        : {run_name}")
        print(f"  Device          : {DEVICE}")
        print(f"  Learning rate   : {DDPG_LEARNING_RATE}")
        print(f"  Buffer size     : {DDPG_BUFFER_SIZE:,}")
        print(f"  Learning starts : {DDPG_LEARNING_STARTS:,}")
        print(f"  Batch size      : {DDPG_BATCH_SIZE}")
        print(f"  Checkpoints     → {MODELS_DIR_DDPG / run_name}")

    def train(self, total_timesteps: int) -> None:

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODELS_DIR_DDPG / self.run_name),
            name_prefix="ddpg_ckpt",
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
        save_path = str(MODELS_DIR_DDPG / f"{self.run_name}_final")
        self.model.save(save_path)
        print(f"Final model saved → {save_path}.zip")
        return save_path

    def load(self, path: str) -> None:
        self.model = DDPG.load(path, env=self.env)
        print(f"Model loaded from: {path}")

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action