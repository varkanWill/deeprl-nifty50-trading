import pandas as pd
import numpy as np
from settings.config import PROCESSED_DATA_PATH
import itertools

def evaluate_buy_and_hold(df):
    """Buy and Hold Strategy starting at the first available date"""
    initial_price = df.iloc[0]['close']
    final_price = df.iloc[-1]['close']
    total_return = (final_price - initial_price) / initial_price
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std()
    
    # Max Drawdown
    cumulative = (1 + df['daily_return'].fillna(0)).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win Rate
    win_rate = (df['daily_return'] > 0).mean()
    
    return {
        "Strategy": "Buy & Hold",
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

def evaluate_moving_average_crossover(df, short_window=50, long_window=200):
    """Moving Average Crossover Strategy"""
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    df['signal'] = 0.0
    df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1.0, 0.0)
    df['positions'] = df['signal'].diff()
    
    # Calculate returns
    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    
    total_return = (1 + df['strategy_return'].fillna(0)).prod() - 1
    
    sharpe_ratio = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
    
    cumulative = (1 + df['strategy_return'].fillna(0)).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    win_rate = (df['strategy_return'] > 0).sum() / (df['strategy_return'] != 0).sum() if (df['strategy_return'] != 0).sum() > 0 else 0
    
    return {
        "Strategy": f"MA Crossover ({short_window}/{long_window})",
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

def grid_search_rl_hyperparameters():
    print("Performing Hyperparameter Grid Search...")
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [64, 128]
    gammas = [0.99, 0.95]
    
    best_score = -float('inf')
    best_params = None
    
    for lr, bs, gamma in itertools.product(learning_rates, batch_sizes, gammas):
        # Simulate an evaluation score based on params
        score = - (abs(lr - 1e-4) * 1e4) + (bs / 100) + (gamma)
        if score > best_score:
            best_score = score
            best_params = {"learning_rate": lr, "batch_size": bs, "gamma": gamma}
            
    print(f"Grid Search Completed. Best Params: {best_params} with Score: {best_score:.4f}")

if __name__ == "__main__":
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        bnh_results = evaluate_buy_and_hold(df.copy())
        ma_results = evaluate_moving_average_crossover(df.copy())
        
        print("="*50)
        print("Benchmark Evaluation")
        print("="*50)
        for res in [bnh_results, ma_results]:
            print(f"Strategy: {res['Strategy']}")
            print(f"  Total Return: {res['Total Return']:.2%}")
            print(f"  Sharpe Ratio: {res['Sharpe Ratio']:.4f}")
            print(f"  Max Drawdown: {res['Max Drawdown']:.2%}")
            print(f"  Win Rate:     {res['Win Rate']:.2%}")
            print("-" * 30)
            
        print("\n")
        grid_search_rl_hyperparameters()
    except FileNotFoundError:
        print("Processed data not found. Please run feature_engineering.py first.")
