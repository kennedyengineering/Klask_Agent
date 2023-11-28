import torch
import gymnasium as gym
from dqn_agent import DQN_Agent
from itertools import count
from tqdm import tqdm
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("WEIGHTS_PATH", default="weights.torch")
args = parser.parse_args()

# Create the environment
env = gym.make("CartPole-v1")

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Create the agent
agent = DQN_Agent(n_observations, n_actions)

# Determine number of episodes
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# Run training
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get it's state
    state, info = env.reset()

    for t in count():
        # Act on the environment
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # Determine next state
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation

        # Store the transition in memory
        agent.remember(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        if done:
            break

print("Complete")
agent.save(args.WEIGHTS_PATH)
agent.close()
