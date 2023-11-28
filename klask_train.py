# Train a DQN agent to play Klask

from dqn_agent import DQN_Agent
from klask_environment import actions, action_to_p1, action_to_p2, states_to_p1, states_to_p2, reward_to_p1, reward_to_p2
from modules.Klask_Simulator.klask_simulator import KlaskSimulator

from itertools import count
from tqdm import tqdm

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("WEIGHTS_PATH", default="weights.torch")
args = parser.parse_args()

# Create the environment
sim = KlaskSimulator(render_mode="headless")

# Get number of actions from sim action space
n_actions = len(actions)

# Get the number of state observations
_, _, agent_states = sim.reset()
n_observations = len(agent_states)

# Create the agent
agent = DQN_Agent(n_observations, n_actions)
assert agent.device.type == "cuda"

# Determine number of episodes
num_episodes = 1000

# Determine episode length
max_episode_length = 600

# Run training
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get it's state
    _, _, agent_states = sim.reset()

    for t in count():
        # Determine action
        p1_states = states_to_p1(agent_states, sim.length_scaler)
        p1_action = action_to_p1(actions[agent.select_action(p1_states).item()])

        p2_states = states_to_p2(agent_states, sim.length_scaler)
        p2_action = action_to_p2(actions[agent.apply_policy(p2_states).item()])
        # Experiment: p2 also uses select_option, and p2 data is added to replay buffer

        # Apply action to environment
        _, game_states, next_agent_states = sim.step(p1_action, p2_action)
        p1_reward = reward_to_p1(game_states)
        p2_reward = reward_to_p2(game_states)
        
        # Determine next state
        terminated = KlaskSimulator.GameStates.PLAYING not in game_states
        truncated = t > max_episode_length
        done = terminated or truncated

        if terminated:
            next_agent_states = None

        # Store the transition in memory
        agent.remember(p1_states, p1_action, states_to_p1(next_agent_states), p1_reward)    # FIXME: states_to_p1 receiveing Nonetype
        # Experiment: p2 data is also added to replay buffer

        # Move to the next state
        agent_states = next_agent_states

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        if done:
            break

print("Complete")
agent.save(args.WEIGHTS_PATH)
agent.close()
sim.close()
