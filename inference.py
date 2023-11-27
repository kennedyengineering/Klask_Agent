import torch
from dqn_agent import DQN_Agent
from itertools import count
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

agent = DQN_Agent(n_observations, n_actions)
agent.load("YO")

# Initialize the environment and get it's state
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
for t in count():
    print(t)
    action = agent.apply_policy(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated

    if terminated:
            next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

    # Move to the next state
        state = next_state

    if done:
        break
