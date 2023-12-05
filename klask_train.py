# Train a DQN agent to play Klask

from dqn_agent import DQN_Agent, DQN
from klask_environment import actions, action_to_p1, action_to_p2, states_to_p1, states_to_p2, reward_to_p1
from modules.Klask_Simulator.klask_simulator import KlaskSimulator

from itertools import count
from tqdm import tqdm

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
def check_positive_nonzero_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue
parser.add_argument("WEIGHTS_PATH")                                                            # Where to save the trained weights
parser.add_argument("--episodes", type=check_positive_nonzero_int, default=1000)               # Number of episodes to train on
parser.add_argument("--checkpoint_interval", type=check_positive_nonzero_int, default=200)     # Save the weights every number of episodes
parser.add_argument("--max_episode_length", type=check_positive_nonzero_int, default=600)      # Maximum number of steps in an episode
parser.add_argument("--render_mode", type=str, default="headless")                             # Can be "human", or "headless"
parser.add_argument("--opponent_update_interval", type=check_positive_nonzero_int, default=20) # Update the opponent's policy every number of episodes
args = parser.parse_args()

# Display configuration
print()
print("Starting training for %d episodes with a max step count of %d" % (args.episodes, args.max_episode_length))
if args.checkpoint_interval == -1:
    print("Checkpoint saving disabled")
else:
    print("Checkpoint saving every %d episodes" % (args.checkpoint_interval))
print("Render mode %s selected" % (args.render_mode))
print("Updating the opponent every %d episodes" % (args.opponent_update_interval))
print()

# Create the environment
sim = KlaskSimulator(render_mode=args.render_mode)

# Get number of actions from sim action space
n_actions = len(actions)

# Get the number of state observations
_, _, agent_states = sim.reset()
n_observations = len(agent_states)

# Create the agent
agent = DQN_Agent(n_observations, n_actions)
assert agent.device.type == "cuda"

# Create the opponent
opponent = DQN_Agent(n_observations, n_actions)
opponent.policy_net.load_state_dict(agent.policy_net.state_dict())

# Run training
for i_episode in tqdm(range(args.episodes)):
    # Initialize the environment and get it's state
    _, _, agent_states = sim.reset()

    for t in count():
        # Determine action
        p1_states = states_to_p1(agent_states, sim.length_scaler)
        p1_action_tensor = agent.select_action(p1_states)
        p1_action = action_to_p1(actions[p1_action_tensor.item()])

        p2_states = states_to_p2(agent_states, sim.length_scaler)
        p2_action_tensor = opponent.apply_policy(p2_states)
        p2_action = action_to_p2(actions[p2_action_tensor.item()])

        # Apply action to environment
        _, game_states, next_agent_states = sim.step(p1_action, p2_action)
        p1_reward = reward_to_p1(game_states)
        
        # Determine next state
        terminated = KlaskSimulator.GameStates.PLAYING not in game_states
        truncated = t > args.max_episode_length
        done = terminated or truncated

        if terminated:
            next_agent_states = None

        # Store the transition in memory
        agent.remember(p1_states, p1_action_tensor, states_to_p1(next_agent_states, sim.length_scaler), p1_reward)

        # Move to the next state
        agent_states = next_agent_states

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        if done:
            break

    # Save checkpoint weights files
    if (i_episode+1) % args.checkpoint_interval == 0:
        agent.save(args.WEIGHTS_PATH+"_checkpoint_"+'{:04d}'.format(i_episode+1))

    # Update the opponent policy network
    if (i_episode+1) % args.opponent_update_interval == 0:
        opponent.policy_net.load_state_dict(agent.policy_net.state_dict())

print("Complete")
agent.save(args.WEIGHTS_PATH+"_final")
agent.close()
sim.close()
