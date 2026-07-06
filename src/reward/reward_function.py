import numpy as np
from settings.config import LAMBDA_VOLATILITY, MU_TRANSACTION

def calculate_risk_aware_reward(portfolio_value_change, portfolio_volatility, transaction_cost):
    """
    Calculate risk-aware reward incorporating transaction costs and volatility penalties.
    
    Args:
        portfolio_value_change: Change in portfolio value from previous step
        portfolio_volatility: Rolling volatility of the portfolio returns
        transaction_cost: Total transaction costs incurred in the step
        
    Returns:
        float: Risk-adjusted reward
    """
    
    base_reward = portfolio_value_change
    volatility_penalty = LAMBDA_VOLATILITY * portfolio_volatility
    transaction_penalty = MU_TRANSACTION * transaction_cost
    
    reward = base_reward - volatility_penalty - transaction_penalty
    
    return float(reward)
