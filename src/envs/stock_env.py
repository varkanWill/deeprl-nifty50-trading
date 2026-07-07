import gymnasium as gym
import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from settings.config import (
    INITIAL_CASH,
    HMAX,
    BUY_COST_PCT,
    SELL_COST_PCT,
    REWARD_SCALING,
    TECHNICAL_INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    VOLATILITY_WINDOW,
    LAMBDA_VOLATILITY,
    MU_TRANSACTION,
)

from src.reward.reward_function import compute_reward


# =============================================================================
# Reward Wrapper
# =============================================================================

class RiskAwareRewardWrapper(gym.Wrapper):
    """
    Sits on top of FinRL's StockTradingEnv and replaces
    its default reward with the custom risk-aware reward:

        R = ΔPortfolio - λ*Volatility - μ*TransactionCost

    How it works:
    - Calls the original env.step() as normal
    - Throws away FinRL's reward
    - Reads asset_memory and cost from the inner env
    - Computes and returns our reward instead
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_cost: float = 0.0
        self.stock_dim = env.stock_dim
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_cost = 0.0          # cost resets to 0 on episode start
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # FinRL tracks cumulative cost, so subtract previous to get step cost
        step_cost = self.env.cost - self._prev_cost
        self._prev_cost = self.env.cost

        # asset_memory has at least 2 entries after first step
        if len(self.env.asset_memory) >= 2:
            reward = compute_reward(
                begin_total_asset=self.env.asset_memory[-2],
                end_total_asset=self.env.asset_memory[-1],
                asset_memory=self.env.asset_memory,
                step_transaction_cost=step_cost,
                lambda_volatility=LAMBDA_VOLATILITY,
                mu_transaction=MU_TRANSACTION,
                window=VOLATILITY_WINDOW,
            )
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info


# =============================================================================
# Helpers
# =============================================================================

def _split_dataframe(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter df to date range and re-index for FinRL."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    split_df = (
        df[mask]
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )
    split_df.index = pd.factorize(split_df["date"])[0]
    return split_df


def _build_env(df: pd.DataFrame) -> gym.Env:
    """Core FinRL env factory (no wrapper)."""
    stock_dim = len(df.tic.unique())

    state_space = (
        1
        + 2 * stock_dim
        + stock_dim * len(TECHNICAL_INDICATORS)
    )

    env = StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=HMAX,
        initial_amount=INITIAL_CASH,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[SELL_COST_PCT] * stock_dim,
        reward_scaling=REWARD_SCALING,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=1,
    )

    return env


# =============================================================================
# Public API
# =============================================================================

def create_stock_env(df: pd.DataFrame) -> gym.Env:
    """
    Original factory kept intact so test_environment.py still works.
    Uses the full df as-is, no reward wrapper.
    """
    return _build_env(df)


def create_train_env(df: pd.DataFrame) -> gym.Env:
    """2014–2021 training environment with risk-aware reward."""
    train_df = _split_dataframe(df, TRAIN_START_DATE, TRAIN_END_DATE)
    env = _build_env(train_df)
    env = RiskAwareRewardWrapper(env)
    print(f"  Train env: {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"  Trading days : {train_df.index.max() + 1}")
    return env


def create_test_env(df: pd.DataFrame) -> gym.Env:
    """2022–2025 test environment with risk-aware reward."""
    test_df = _split_dataframe(df, TEST_START_DATE, TEST_END_DATE)
    env = _build_env(test_df)
    env = RiskAwareRewardWrapper(env)
    print(f"  Test env : {test_df['date'].min()} → {test_df['date'].max()}")
    print(f"  Trading days : {test_df.index.max() + 1}")
    return env

def create_train_env_discrete(df: pd.DataFrame) -> gym.Env:
    """2014–2021 training environment with discrete actions for DQN."""
    from src.envs.discrete_wrapper import DiscreteActionWrapper
    train_df = _split_dataframe(df, TRAIN_START_DATE, TRAIN_END_DATE)
    env = _build_env(train_df)
    env = RiskAwareRewardWrapper(env)
    env = DiscreteActionWrapper(env)
    print(f"  Train env (discrete): {train_df['date'].min()} → {train_df['date'].max()}")
    return env
def create_test_env_discrete(df: pd.DataFrame) -> gym.Env:
    """2022–2025 test environment with discrete actions for DQN."""
    from src.envs.discrete_wrapper import DiscreteActionWrapper

    test_df = _split_dataframe(df, TEST_START_DATE, TEST_END_DATE)

    env = _build_env(test_df)
    env = RiskAwareRewardWrapper(env)
    env = DiscreteActionWrapper(env)

    print(f"  Test env (discrete): {test_df['date'].min()} → {test_df['date'].max()}")

    return env
