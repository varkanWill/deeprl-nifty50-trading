import numpy as np
import pandas as pd

from settings.config import (
    PROCESSED_DATA_PATH,
)

from src.envs.stock_env import create_stock_env


def print_portfolio(env, state):

    stock_dim = env.stock_dim

    print(f"Cash            : {state[0]:,.2f}")

    print(f"Portfolio Value : {env.asset_memory[-1]:,.2f}")

    print(f"Transaction Fee : {env.cost:,.2f}")

    print(f"Shares Held     : {state[1+stock_dim:1+2*stock_dim]}")

    print()


def main():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    df = (
        df.sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    df.index = pd.factorize(df["date"])[0]

    env = create_stock_env(df)

    print("=" * 80)
    print("RESET")
    print("=" * 80)

    state, _ = env.reset()

    print_portfolio(env, state)

    # ======================================================

    print("=" * 80)
    print("BUY")
    print("=" * 80)

    action = np.ones(env.stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward : {reward}")

    print_portfolio(env, state)

    # ======================================================

    print("=" * 80)
    print("HOLD")
    print("=" * 80)

    action = np.zeros(env.stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward : {reward}")

    print_portfolio(env, state)

    # ======================================================

    print("=" * 80)
    print("SELL")
    print("=" * 80)

    action = -np.ones(env.stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward : {reward}")

    print_portfolio(env, state)

    print("=" * 80)
    print("ENVIRONMENT PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()