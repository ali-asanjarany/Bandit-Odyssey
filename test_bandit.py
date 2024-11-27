from Bandit_Odyssey import BanditEnvironment

# Create the bandit environment
env = BanditEnvironment(n_arms=5, nonstationary=True)

# Initial status
status = env.get_status()
print("Initial status:", status)

# Pull an arm and get a reward
reward = env.pull(2)
print("Reward from pulling arm 2:", reward)

# Updated status
status = env.get_status()
print("Updated status:", status)