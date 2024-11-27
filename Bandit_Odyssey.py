import numpy as np






class BanditEnvironment:
    def __init__(self, n_arms=10, nonstationary=False, drift=0.01):
    
        # Simulates the k-armed bandit environment.
        # param n_arms: Number of arms (default: 10).
        # param nonstationary: If True, the environment will be nonstationary.
        # param drift: The rate of gradual change in the true values of the arms in a nonstationary environment.
        
        self.n_arms = n_arms
        self.nonstationary = nonstationary
        self.drift = drift
        
        # Initials arms values.
        self.q_star = np.random.normal(0,1,n_arms)

        # Reward history for evaluating algorithms.
        self.reward_history = []

    def pull(self, arm):
        
        # simulates pulling an arm and returns a reward.
        # parm arm: the selected arm number.
        # return: A random reward for the selected arm.
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Arm number must be between 0 and {self.n_arms-1}")
        
        # Generate reward based on actual arm value.
        reward = np.random.normal(self.q_star[arm], 1)
        self.reward_history.append(reward)

        # If the environment is non-stationary, the real arm values changing.
        if self.nonstationary:
            self.q_star += np.random.normal(0, self.drift, self.n_arms)
        return reward
    
    def reset(self):
        # Reset the environment to initial state.
        self.q_star = np.random.normal(0,1, self.n_arms)
        self.reward_history = []

    def get_status(self):
        """
        Returns the current status of the environment.
        :return: A dictionary with the true values of the arms and the reward history.
        """
        return {
            "q_star": self.q_star.copy(),  # To avoid unintentional modifications
            "reward_history": self.reward_history.copy()
        }
    