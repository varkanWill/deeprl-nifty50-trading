# Deep RL Nifty 50 Trading

Welcome to the **DeepRL Nifty 50 Trading** repository! This project leverages the power of Deep Reinforcement Learning (DRL) to build and train autonomous trading agents specifically designed for the Indian Nifty 50 stock market.

## Overview

The core idea behind this project is to apply advanced reinforcement learning techniques—such as DDPG (Deep Deterministic Policy Gradients), PPO (Proximal Policy Optimization), or other relevant state-of-the-art DRL algorithms—to navigate the complexities of financial markets. By learning from historical market data, our agents aim to discover profitable trading strategies while effectively managing risk.

## Project Structure

Here's a quick look at how the project is organized:
- `data/`: Contains datasets, including processed stock data for training and testing the agents.
- `src/agents/`: Implementations of various deep reinforcement learning agents (e.g., DDPG).
- `src/envs/`: Custom trading environments built (likely compatible with OpenAI Gym/Gymnasium) to simulate the stock market.
- `src/reward/`: Custom reward functions designed to optimize agent behavior for trading.
- `runners/`: Scripts to easily execute training and evaluation loops.
- `settings/`: Configuration files for the models, environments, and training parameters.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/varkanWill/deeprl-nifty50-trading.git
   cd deeprl-nifty50-trading
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   # OR if using pyproject.toml
   pip install -e .
   ```

## Approach

The project approaches stock trading as a Markov Decision Process (MDP).
- **State**: Historical price data, technical indicators, and current portfolio balance.
- **Action**: Decisions to Buy, Hold, or Sell specific quantities of stocks.
- **Reward**: Change in portfolio value, risk-adjusted returns (like Sharpe ratio), or custom metrics defined in `src/reward/`.

## Contributing

Contributions are welcome! If you have ideas for new reward functions, better state representations, or more robust DRL algorithms, feel free to open an issue or submit a pull request.

## License

This project is licensed under the terms of the LICENSE file included in the repository. Happy coding and profitable trading!
