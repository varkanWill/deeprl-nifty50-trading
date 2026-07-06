from pathlib import Path
import torch

from finrl.config import INDICATORS

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH      = RAW_DATA_DIR / "stock.csv"
RAW_DATA_FILE      = RAW_DATA_PATH
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "stock_processed.csv"

# =============================================================================
# Dataset
# =============================================================================

TICKERS = [
    "^NSEI",
]

START_DATE = "2014-01-01"
END_DATE   = "2025-12-31"

# =============================================================================
# Feature Engineering
# =============================================================================

TECHNICAL_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "close_30_ema"
]

# =============================================================================
# Trading Environment
# =============================================================================

INITIAL_CASH   = 1_000_000
HMAX           = 100
BUY_COST_PCT   = 0.001
SELL_COST_PCT  = 0.001
REWARD_SCALING = 1e-4

# =============================================================================
# Risk-Aware Reward
# =============================================================================

VOLATILITY_WINDOW = 20
LAMBDA_VOLATILITY = 0.01
MU_TRANSACTION    = 0.01

# =============================================================================
# Train / Test Split
# =============================================================================

TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE   = "2021-12-31"
TEST_START_DATE  = "2022-01-01"
TEST_END_DATE    = "2025-12-31"

# =============================================================================
# Paths — Models and Logs
# =============================================================================

MODELS_DIR     = BASE_DIR / "models" / "ppo"
MODELS_DIR_SAC = BASE_DIR / "models" / "sac"
LOGS_DIR       = BASE_DIR / "results" / "logs"
METRICS_DIR    = BASE_DIR / "results" / "metrics"

# =============================================================================
# Device
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PPO Hyperparameters
# =============================================================================

PPO_LEARNING_RATE   = 3e-4
PPO_N_STEPS         = 2048
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_RANGE      = 0.2
PPO_ENT_COEF        = 0.01
PPO_TOTAL_TIMESTEPS = 500_000

# =============================================================================
# SAC Hyperparameters
# =============================================================================

SAC_LEARNING_RATE   = 3e-4
SAC_BUFFER_SIZE     = 100_000
SAC_LEARNING_STARTS = 1_000
SAC_BATCH_SIZE      = 256
SAC_TAU             = 0.005
SAC_GAMMA           = 0.99
SAC_ENT_COEF        = "auto"
SAC_TOTAL_TIMESTEPS = 500_000

# =============================================================================
# DDPG Hyperparameters
# =============================================================================

MODELS_DIR_DDPG      = BASE_DIR / "models" / "ddpg"
DDPG_LEARNING_RATE   = 1e-4
DDPG_BUFFER_SIZE     = 100_000
DDPG_LEARNING_STARTS = 1_000
DDPG_BATCH_SIZE      = 256
DDPG_TAU             = 0.005
DDPG_GAMMA           = 0.99
DDPG_TOTAL_TIMESTEPS = 500_000

# =============================================================================
# DQN Hyperparameters
# =============================================================================

MODELS_DIR_DQN             = BASE_DIR / "models" / "dqn"
DQN_LEARNING_RATE          = 1e-4
DQN_BUFFER_SIZE            = 100_000
DQN_LEARNING_STARTS        = 1_000
DQN_BATCH_SIZE             = 64
DQN_GAMMA                  = 0.99
DQN_TARGET_UPDATE_INTERVAL = 1_000
DQN_EXPLORATION_FRACTION   = 0.3
DQN_EXPLORATION_FINAL_EPS  = 0.05
DQN_TOTAL_TIMESTEPS        = 500_000