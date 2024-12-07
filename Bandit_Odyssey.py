import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class MultiArmedBandit:
    def __init__(self, k, true_reward_means):
        """
        Simulates the k-armed bandit environment.
        :param k: Number of arms.
        :param true_reward_means: List of true reward means for each arm.
        """
        self.k = k
        self.true_reward_means = true_reward_means

    def pull(self, arm):
        """
        Simulates pulling an arm.
        :param arm: Index of the selected arm.
        :return: Reward drawn from a Gaussian distribution with the arm's mean.
        """
        return np.random.normal(self.true_reward_means[arm], 1.0)


class EpsilonGreedyAgent:
    def __init__(self, k, epsilon):
        """
        Implements the epsilon-greedy algorithm.
        :param k: Number of arms.
        :param epsilon: Exploration probability.
        """
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = np.zeros(k)  # Estimated rewards for each arm
        self.action_counts = np.zeros(k)  # Count of pulls for each arm

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)  # Random action
        else:
            return np.argmax(self.q_estimates)  # Exploit the best arm

    def update(self, arm, reward):
        self.action_counts[arm] += 1
        self.q_estimates[arm] += (reward - self.q_estimates[arm]) / self.action_counts[arm]
class GreedyOptimisticAgent:
    def __init__(self, k, initial_value):
        """
        Implements greedy with optimistic initialization.
        :param k: Number of arms.
        :param initial_value: Optimistic initial value for each arm.
        """
        self.k = k
        # Initialize estimates with the optimistic value
        self.q_estimates = np.full(k, initial_value)  # Optimistic initialization
        self.action_counts = np.zeros(k)
        self.alpha = 0.1  # Fixed step size as mentioned in the book

    def select_action(self):
        # Pure greedy selection
        return np.argmax(self.q_estimates)

    def update(self, arm, reward):
        # Use constant step size instead of sample average
        self.q_estimates[arm] += self.alpha * (reward - self.q_estimates[arm])

class UCBAgent:
    def __init__(self, k, c):
        """
        Implements the Upper Confidence Bound (UCB) algorithm.
        :param k: Number of arms.
        :param c: Exploration parameter.
        """
        self.k = k
        self.c = c
        self.q_estimates = np.zeros(k)  # Estimated rewards for each arm
        self.action_counts = np.zeros(k)  # Count of pulls for each arm
        self.total_steps = 0

    def select_action(self):
        self.total_steps += 1
        ucb_values = self.q_estimates + self.c * np.sqrt(np.log(self.total_steps) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.action_counts[arm] += 1
        self.q_estimates[arm] += (reward - self.q_estimates[arm]) / self.action_counts[arm]

class GradientBanditAgent:
    def __init__(self, k, alpha):
        """
        Implements the gradient bandit algorithm.
        :param k: Number of arms.
        :param alpha: Step-size parameter.
        """
        self.k = k
        self.alpha = alpha
        self.preferences = np.zeros(k)  # Preferences for each arm
        self.average_reward = 0  # Average reward received
        self.total_steps = 0

    def select_action(self):
        exp_prefs = np.exp(self.preferences)
        probabilities = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.k, p=probabilities)

    def update(self, arm, reward):
        self.total_steps += 1
        self.average_reward += (reward - self.average_reward) / self.total_steps
        exp_prefs = np.exp(self.preferences)
        probabilities = exp_prefs / np.sum(exp_prefs)
        for a in range(self.k):
            if a == arm:
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * probabilities[a]


def simulate_bandit(agent, bandit, steps):
    """
    Simulates the interaction between the agent and the bandit environment.
    :param agent: The agent implementing a bandit algorithm.
    :param bandit: The bandit environment.
    :param steps: Number of steps to simulate.
    :return: Average reward over the simulation.
    """
    rewards = []
    for _ in range(steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards.append(reward)
    return np.mean(rewards)

def run_single_trial(k, params, _=None):
    """
    Runs a single trial for all algorithms.
    :param k: Number of arms
    :param params: Dictionary of parameters for each algorithm
    :param _: Optional parameter to handle the iteration number from pool.imap
    :return: Dictionary of results for this trial
    """
    # Initialize results for this trial
    trial_results = {
        "ε-greedy": np.zeros(len(params["epsilon"])),
        "Optimistic Greedy": np.zeros(len(params["initial_value"])),
        "UCB": np.zeros(len(params["c"])),
        "Gradient Bandit": np.zeros(len(params["alpha"]))
    }
    
    # Create new true means for this trial
    true_means = np.random.normal(0, 1, k)
    bandit = MultiArmedBandit(k, true_means)
    
    # Run all algorithms for this trial
    for i, epsilon in enumerate(params["epsilon"]):
        agent = EpsilonGreedyAgent(k, epsilon)
        trial_results["ε-greedy"][i] = simulate_bandit(agent, bandit, 1000)
    
    for i, initial_value in enumerate(params["initial_value"]):
        agent = GreedyOptimisticAgent(k, initial_value)
        trial_results["Optimistic Greedy"][i] = simulate_bandit(agent, bandit, 1000)
    
    for i, c in enumerate(params["c"]):
        agent = UCBAgent(k, c)
        trial_results["UCB"][i] = simulate_bandit(agent, bandit, 1000)
    
    for i, alpha in enumerate(params["alpha"]):
        agent = GradientBanditAgent(k, alpha)
        trial_results["Gradient Bandit"][i] = simulate_bandit(agent, bandit, 1000)
    
    return trial_results

# Replace the trial loop with this new code
if __name__ == '__main__':
    # Experiment setup
    k = 10  # Number of arms
    n_trials = 1000  # Number of independent trials
    
    # Parameters for each algorithm
    params = {
        "epsilon": [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],  # ε values
        "initial_value": [1/3.3, 1/1.8, 1.2, 2.1, 3.8],  # Q₀ values
        "c": [1/16, 1/3.3, 1/1.5, 1.2, 2, 4],  # c values
        "alpha": [1/32, 1/16, 1/8, 1/4, 1/1.5, 1.2, 2.2, 3.2]  # α values
    }
    
    # Initialize results
    results = {
        "ε-greedy": np.zeros(len(params["epsilon"])),
        "Optimistic Greedy": np.zeros(len(params["initial_value"])),
        "UCB": np.zeros(len(params["c"])),
        "Gradient Bandit": np.zeros(len(params["alpha"]))
    }
    
    # Create a partial function with fixed parameters
    run_trial = partial(run_single_trial, k, params)  # Remove 'steps' parameter
    
    # Use multiprocessing to run trials in parallel
    n_cores = mp.cpu_count()
    with mp.Pool(n_cores) as pool:
        # Run trials with progress bar
        trial_results = list(tqdm(
            pool.imap(run_trial, range(n_trials)),
            total=n_trials,
            desc=f"Running trials using {n_cores} cores"
        ))
    
    # Combine results from all trials
    for trial_result in trial_results:
        for method in results:
            results[method] += trial_result[method]
    
    # Average the results
    for method in results:
        results[method] /= n_trials

    # Add this before plotting
    print("Optimistic Greedy results:", results["Optimistic Greedy"])

    # Plotting
    plt.figure(figsize=(10, 6))

    # Create proper x-axis values that match Figure 2.6
    x_values = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4])

    # Plot each line with their corresponding x-values
    plt.plot([1/128, 1/64, 1/32, 1/16, 1/8, 1/4], results["ε-greedy"], 
             color='red', label="ε-greedy", marker='o')

    plt.plot([1/3.3, 1/1.8, 1.2, 2.1, 3.8], results["Optimistic Greedy"], 
             color='black', label="greedy with\noptimistic initialization\nα = 0.1", marker='o')

    plt.plot([1/16, 1/3.3, 1/1.5, 1.2, 2, 4], results["UCB"], 
             color='blue', label="UCB", marker='o')

    plt.plot([1/32, 1/16, 1/8, 1/4, 1/1.5, 1.2, 2.2, 3.2], results["Gradient Bandit"], 
             color='green', label="gradient\nbandit", marker='o')

    # Set the x-axis to log scale
    plt.xscale('log', base=2)  # Use base 2 since values double each time

    # Set axis limits
    # plt.ylim(1.0, 1.5)
    # plt.xlim(1/128, 4)

    # Set x-ticks to match the book's figure
    plt.xticks(x_values, ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'])

    plt.xlabel("Parameter(ε, α, c, Q₀)")
    plt.ylabel("Average reward\nover first 1000 steps")

    plt.legend(loc='best')
    # plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()