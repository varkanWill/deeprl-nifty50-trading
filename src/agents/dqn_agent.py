from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from settings.config import (
    DQN_LEARNING_RATE,
    DQN_BUFFER_SIZE,
    DQN_LEARNING_STARTS,
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_TARGET_UPDATE_INTERVAL,
    DQN_EXPLORATION_FRACTION,
    DQN_EXPLORATION_FINAL_EPS,
    MODELS_DIR_DQN,
    LOGS_DIR,
    DEVICE,
)


class DQNAgent:

    def __init__(self, env: gym.Env, run_name: str = "dqn_run_1"):
        self.run_name = run_name
        self.env = Monitor(env)

        MODELS_DIR_DQN.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=DQN_LEARNING_RATE,
            buffer_size=DQN_BUFFER_SIZE,
            learning_starts=DQN_LEARNING_STARTS,
            batch_size=DQN_BATCH_SIZE,
            gamma=DQN_GAMMA,
            target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
            exploration_fraction=DQN_EXPLORATION_FRACTION,
            exploration_final_eps=DQN_EXPLORATION_FINAL_EPS,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
            device=DEVICE,
        )

        print(f"\nDQN Agent ready")
        print(f"  Run name           : {run_name}")
        print(f"  Device             : {DEVICE}")
        print(f"  Learning rate      : {DQN_LEARNING_RATE}")
        print(f"  Buffer size        : {DQN_BUFFER_SIZE:,}")
        print(f"  Exploration frac   : {DQN_EXPLORATION_FRACTION}")
        print(f"  Final epsilon      : {DQN_EXPLORATION_FINAL_EPS}")
        print(f"  Checkpoints        → {MODELS_DIR_DQN / run_name}")

    def train(self, total_timesteps: int) -> None:

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODELS_DIR_DQN / self.run_name),
            name_prefix="dqn_ckpt",
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
        save_path = str(MODELS_DIR_DQN / f"{self.run_name}_final")
        self.model.save(save_path)
        print(f"Final model saved → {save_path}.zip")
        return save_path

    def load(self, path: str) -> None:
        self.model = DQN.load(path, env=self.env)
        print(f"Model loaded from: {path}")

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action