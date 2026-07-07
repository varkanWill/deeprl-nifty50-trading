from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from settings.config import (
    SAC_LEARNING_RATE,
    SAC_BUFFER_SIZE,
    SAC_LEARNING_STARTS,
    SAC_BATCH_SIZE,
    SAC_TAU,
    SAC_GAMMA,
    SAC_ENT_COEF,
    MODELS_DIR_SAC,
    LOGS_DIR,
    DEVICE,
)


class SACAgent:

    def __init__(self, env: gym.Env, run_name: str = "sac_run_1"):
        self.run_name = run_name
        self.env = Monitor(env)

        MODELS_DIR_SAC.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=SAC_LEARNING_RATE,
            buffer_size=SAC_BUFFER_SIZE,
            learning_starts=SAC_LEARNING_STARTS,
            batch_size=SAC_BATCH_SIZE,
            tau=SAC_TAU,
            gamma=SAC_GAMMA,
            ent_coef=SAC_ENT_COEF,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
            device=DEVICE,
        )

        print(f"\nSAC Agent ready")
        print(f"  Run name        : {run_name}")
        print(f"  Device          : {DEVICE}")
        print(f"  Learning rate   : {SAC_LEARNING_RATE}")
        print(f"  Buffer size     : {SAC_BUFFER_SIZE:,}")
        print(f"  Learning starts : {SAC_LEARNING_STARTS:,}")
        print(f"  Batch size      : {SAC_BATCH_SIZE}")
        print(f"  Ent coef        : {SAC_ENT_COEF}")
        print(f"  Checkpoints     → {MODELS_DIR_SAC / run_name}")
        print(f"  Tensorboard     → {LOGS_DIR}")

    def train(self, total_timesteps: int) -> None:

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODELS_DIR_SAC / self.run_name),
            name_prefix="sac_ckpt",
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
        save_path = str(MODELS_DIR_SAC / f"{self.run_name}_final")
        self.model.save(save_path)
        print(f"Final model saved → {save_path}.zip")
        return save_path

    def load(self, path: str) -> None:
        self.model = SAC.load(path, env=self.env)
        print(f"Model loaded from: {path}")

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action