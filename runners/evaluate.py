
# --- Inlined from src/strategies/buy_and_hold.py ---
import numpy as np


def run_buy_and_hold(env):
    """
    Buy all stocks equally on the first day and hold.

    Returns
    -------
    np.ndarray
        Portfolio value over time.
    """

    obs, info = env.reset()

    stock_dim = env.unwrapped.stock_dim

    portfolio = [env.unwrapped.asset_memory[-1]]

    # Buy all stocks on first step
    first_action = np.ones(stock_dim, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(first_action)

    portfolio.append(env.unwrapped.asset_memory[-1])

    done = terminated or truncated

    # Hold afterwards
    hold_action = np.zeros(stock_dim, dtype=np.float32)

    while not done:

        obs, reward, terminated, truncated, info = env.step(hold_action)

        portfolio.append(env.unwrapped.asset_memory[-1])

        done = terminated or truncated

    return np.array(portfolio)

# --- Inlined from src/strategies/moving_average.py ---
import numpy as np
import pandas as pd


def run_ma_strategy(env, df, window=20):
    """
    Simple Moving Average strategy.

    Buy:
        Close > MA

    Sell:
        Close < MA
    """

    obs, info = env.reset()

    stock_dim = env.unwrapped.stock_dim

    portfolio = [env.unwrapped.asset_memory[-1]]

    done = False

    dates = sorted(df["date"].unique())

    for i in range(len(dates) - 1):

        if done:
            break

        current = df[df["date"] == dates[i]]

        action = np.zeros(stock_dim, dtype=np.float32)

        for j, (_, row) in enumerate(current.iterrows()):

            history = df[df["tic"] == row["tic"]]

            history = history[history["date"] <= row["date"]]

            if len(history) < window:
                continue

            ma = history["close"].tail(window).mean()

            if row["close"] > ma:
                action[j] = 1.0

            elif row["close"] < ma:
                action[j] = -1.0

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio.append(env.unwrapped.asset_memory[-1])

        done = terminated or truncated

    return np.array(portfolio)

# --- Inlined from src/evaluation/backtest.py ---
from stable_baselines3 import DQN, DDPG
import numpy as np


def run_backtest(model, env):

    obs, info = env.reset()

    portfolio = [env.unwrapped.asset_memory[-1]]

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio.append(
            env.unwrapped.asset_memory[-1]
        )

        done = terminated or truncated

    return np.array(portfolio)

# --- Inlined from src/evaluation/metrics.py ---
import numpy as np


def compute_metrics(portfolio_values):
    portfolio_values = np.asarray(portfolio_values)

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    total_return = portfolio_values[-1] / portfolio_values[0] - 1

    years = len(portfolio_values) / 252

    cagr = (
        portfolio_values[-1] /
        portfolio_values[0]
    ) ** (1 / years) - 1

    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        sharpe = 0.0
    else:
        sharpe = (
            np.mean(daily_returns)
            / np.std(daily_returns)
        ) * np.sqrt(252)

    running_max = np.maximum.accumulate(portfolio_values)

    drawdown = (
        portfolio_values
        - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    win_rate = np.mean(daily_returns > 0)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": abs(max_drawdown),
        "Win Rate": win_rate,
    }

# --- Inlined from src/visualization/plots.py ---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_equity_curve(portfolios, filename="results/figures/equity_curve.png"):
    """
    portfolios:
        dict[str, np.ndarray]
    """

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for name, values in portfolios.items():
        plt.plot(values, label=name)

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_metric_bar(results,
                    metric="Sharpe",
                    filename="results/figures/sharpe_bar.png"):

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    values = [results[k][metric] for k in names]

    plt.figure(figsize=(8, 5))

    plt.bar(names, values)

    plt.title(metric)
    plt.ylabel(metric)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_drawdown_curve(portfolio,
                        filename="results/figures/drawdown.png"):

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    running_max = np.maximum.accumulate(portfolio)

    drawdown = (
        portfolio - running_max
    ) / running_max

    plt.figure(figsize=(12, 5))

    plt.plot(drawdown)

    plt.title("Drawdown Curve")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from stable_baselines3 import DQN, DDPG

from src.envs.stock_env import (
    create_test_env,
    create_test_env_discrete,
)



def evaluate_model(model, env, name):
    print(f"\nEvaluating {name}...")

    portfolio = run_backtest(model, env)

    metrics = compute_metrics(portfolio)

    print(f"{name} Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")

    return metrics


def main():

    print("=" * 60)
    print("Loading processed dataset...")
    print("=" * 60)

    df = pd.read_csv("data/processed/stock_processed.csv")

    results = {}
    
    portfolios = {} 

    # ============================================================
    # DQN
    # ============================================================

    # ===========================
# DQN
# ===========================

    dqn_env = create_test_env_discrete(df)

    dqn = DQN.load("models/dqn/dqn_run_1_final.zip")

    dqn_portfolio = run_backtest(dqn, dqn_env)

    portfolios["DQN"] = dqn_portfolio

    results["DQN"] = compute_metrics(dqn_portfolio)

    print(results["DQN"])

    # ============================================================
    # DDPG
    # ============================================================

    # ===========================
# DDPG
# ===========================

    ddpg_env = create_test_env(df)

    ddpg = DDPG.load("models/ddpg/ddpg_run_1_final.zip")

    ddpg_portfolio = run_backtest(ddpg, ddpg_env)

    portfolios["DDPG"] = ddpg_portfolio

    results["DDPG"] = compute_metrics(ddpg_portfolio)

    print(results["DDPG"])

        # ===========================
    # Buy & Hold
    # ===========================

    buy_env = create_test_env(df)

    buy_portfolio = run_buy_and_hold(buy_env)

    results["Buy & Hold"] = compute_metrics(
        buy_portfolio
    )
    portfolios["Buy & Hold"] = buy_portfolio

    # ===========================
    # Moving Average
    # ===========================

    ma_env = create_test_env(df)

    ma_portfolio = run_ma_strategy(
        ma_env,
        df,
    )

    results["MA Strategy"] = compute_metrics(
        ma_portfolio
    )
    portfolios["MA Strategy"] = ma_portfolio
    # ============================================================
    # Comparison table
    # ============================================================

    table = pd.DataFrame(results).T

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(table)

    Path("results").mkdir(exist_ok=True)

    table.to_csv(
        "results/evaluation_metrics.csv"
    )
    
    print("\nSaved metrics to results/evaluation_metrics.csv")
    save_equity_curve(portfolios)

    save_metric_bar(
        results,
        metric="Sharpe",
        filename="results/figures/sharpe_bar.png",
    )

    save_metric_bar(
        results,
        metric="Total Return",
        filename="results/figures/return_bar.png",
    )

    best_name = max(
        results,
        key=lambda x: results[x]["Sharpe"],
    )

    save_drawdown_curve(
        portfolios[best_name],
        filename="results/figures/drawdown.png",
    )

    print("\nFigures saved to results/figures/")

if __name__ == "__main__":
    main()