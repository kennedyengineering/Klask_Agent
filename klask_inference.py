# Have a DQN agent play Klask with itself or a human

from dqn_agent import DQN_Agent
from klask_environment import actions, action_to_p1, action_to_p2, states_to_p1, states_to_p2
from modules.Klask_Simulator.klask_simulator import KlaskSimulator

import argparse

# TODO: add keyboard controller as option instead of policy control

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("WEIGHTS_PATH", default="weights.torch")
args = parser.parse_args()

# Create the environment
sim = KlaskSimulator(render_mode="human")

# Get number of actions from sim action space
n_actions = len(actions)

# Get the number of state observations
_, _, agent_states = sim.reset()
n_observations = len(agent_states)

# Create the agent
agent = DQN_Agent(n_observations, n_actions)
assert agent.device.type == "cuda"
agent.load(args.WEIGHTS_PATH)

# Run inference
done = False

while not done:
    # Determine action
    p1_states = states_to_p1(agent_states, sim.length_scaler)
    p1_action = action_to_p1(actions[agent.apply_policy(p1_states).item()])

    p2_states = states_to_p2(agent_states, sim.length_scaler)
    p2_action = action_to_p2(actions[agent.apply_policy(p2_states).item()])

    # Apply action to environment
    _, game_states, agent_states = sim.step(p1_action, p2_action)
    
    # Determine next state
    done = KlaskSimulator.GameStates.PLAYING not in game_states

print("Complete")
agent.close()
sim.close()
