from dqn_agent import DQN_Agent
from itertools import count
import gymnasium as gym
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("WEIGHTS_PATH")
args = parser.parse_args()

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Create the agent
agent = DQN_Agent(n_observations, n_actions)
agent.load(args.WEIGHTS_PATH)

# Run inference
for t in count():
    print(t)

    # Act on environment
    action = agent.apply_policy(state)
    state, reward, terminated, truncated, _ = env.step(action.item())
    
    if terminated or truncated:
        break
