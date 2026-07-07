import numpy as np
import gymnasium as gym


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts continuous multi-stock action space to discrete.

    DQN requires a single integer action. This wrapper maps
    each integer to a meaningful per-stock continuous vector.

    Action layout (stock_dim = 10 → 21 total actions):
        0  .. 9  : buy stock i maximally (+1.0), hold rest
        10 .. 19 : sell stock i maximally (-1.0), hold rest
        20       : hold all positions (zeros)

    Why this design:
    - Avoids 3^10 = 59049 combination explosion
    - Keeps per-stock granularity
    - DQN can learn which individual stock to act on
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.stock_dim = env.stock_dim
        n_actions = 2 * self.stock_dim + 1
        self.action_space = gym.spaces.Discrete(n_actions)

        print(f"  Discrete action space: {n_actions} actions")
        print(f"    0..{self.stock_dim - 1}  → buy stock i")
        print(f"    {self.stock_dim}..{2 * self.stock_dim - 1} → sell stock i")
        print(f"    {2 * self.stock_dim}     → hold all")

    def action(self, discrete_action: int) -> np.ndarray:
        """Map integer action to continuous vector."""
        continuous = np.zeros(self.stock_dim, dtype=np.float32)

        if discrete_action < self.stock_dim:
            # buy stock i maximally
            continuous[discrete_action] = 1.0

        elif discrete_action < 2 * self.stock_dim:
            # sell stock i maximally
            continuous[discrete_action - self.stock_dim] = -1.0

        # else: hold all → zeros

        return continuous