import numpy as np


def compute_reward(
    begin_total_asset: float,
    end_total_asset: float,
    asset_memory: list,
    step_transaction_cost: float,
    lambda_volatility: float = 0.01,
    mu_transaction: float = 0.01,
    window: int = 20,
):
    """
    Risk-aware reward function.

    Reward =
        Portfolio Profit
        - lambda * rolling volatility
        - mu * current transaction cost
    """

    raw_profit = end_total_asset - begin_total_asset

    volatility_penalty = 0.0

    if len(asset_memory) >= window + 1:

        returns = np.diff(asset_memory[-(window + 1):]) / np.array(
            asset_memory[-(window + 1):-1]
        )

        volatility_penalty = np.std(returns)

    reward = (
        raw_profit
        - lambda_volatility * volatility_penalty
        - mu_transaction * step_transaction_cost
    )

    return reward