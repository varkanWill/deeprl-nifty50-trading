from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from settings.config import PROCESSED_DATA_PATH, DQN_TOTAL_TIMESTEPS
from src.envs.stock_env import create_train_env_discrete
from src.agents.dqn_agent import DQNAgent


def main():

    print("=" * 60)
    print("DQN TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Loading processed dataset...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at:\n  {PROCESSED_DATA_PATH}\n\n"
            "Run this first:\n  python -m runners.build_features"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    print(f"  Rows   : {len(df):,}")
    print(f"  Dates  : {df['date'].min()} → {df['date'].max()}")
    print(f"  Stocks : {df['tic'].nunique()}")

    print("\n[2/3] Creating discrete training environment...")
    train_env = create_train_env_discrete(df)
    print(f"  Obs space    : {train_env.observation_space.shape}")
    print(f"  Action space : {train_env.action_space.n} discrete actions")

    print("\n[3/3] Initialising DQN agent...")
    agent = DQNAgent(env=train_env, run_name="dqn_run_1")
    agent.train(total_timesteps=DQN_TOTAL_TIMESTEPS)
    agent.save()

    print("\n" + "=" * 60)
    print("DONE — tensorboard --logdir results/logs --port 6007")
    print("=" * 60)


if __name__ == "__main__":
    main()