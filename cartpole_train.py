import torch
import gymnasium as gym
from dqn_agent import DQN_Agent
from itertools import count
from tqdm import tqdm
from plot_train import live_plot
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("WEIGHTS_PATH")
parser.add_argument("FIGURE_PATH", nargs='?', default="train.png")
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

# Create plot
plotter = live_plot("TEST")

# Run training
episode_rewards = []
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get it's state
    state, info = env.reset()

    # Keep track of episode reward
    episode_reward = 0

    for t in count():
        # Act on the environment
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward

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

    episode_rewards.append(episode_reward)
    plotter.update(episode_rewards)

print("Complete")
agent.save(args.WEIGHTS_PATH)
agent.close()

plotter.save(args.FIGURE_PATH)
plotter.close()
