# Use Klask_Simulator to train a DQN agent via policy self-play

from modules.Klask_Simulator.klask_simulator import KlaskSimulator
from modules.Klask_Simulator.klask_constants import KG_BOARD_HEIGHT, KG_BOARD_WIDTH

# import pygame
# def numpy_to_pygame(arr):
#     # Convert image stored in np array to pygame Surface
#     return pygame.surfarray.make_surface(arr.swapaxes(0,1))

# import numpy as np
# def frame_to_p2(arr):
#     # Rotate frame to p2 perspective
#     return np.rot90(arr, -1)

# def frame_to_p1(arr):
#     # Rotate frame to p1 perspective
#     return np.rot90(arr)

action_force = 0.01
actions = ((action_force, 0),   # right
           (-action_force, 0),  # left
           (0, 0),              # noop
           (0, action_force),   # up
           (0, -action_force))  # down

state_names =  ["biscuit1_pos_x",
                "biscuit1_pos_y",
                "biscuit1_vel_x",
                "biscuit1_vel_y",
                "biscuit2_pos_x",
                "biscuit2_pos_y",
                "biscuit2_vel_x",
                "biscuit2_vel_y",
                "biscuit3_pos_x",
                "biscuit3_pos_y",
                "biscuit3_vel_x",
                "biscuit3_vel_y",
                "puck1_pos_x",
                "puck1_pos_y",
                "puck1_vel_x",
                "puck1_vel_y",
                "puck2_pos_x",
                "puck2_pos_y",
                "puck2_vel_x",
                "puck2_vel_y",
                "ball_pos_x",
                "ball_pos_y",
                "ball_vel_x",
                "ball_vel_y"]

def print_states(states):
    # Display agent states in human readable form
    for el in list(zip(state_names, states)):
        print(el)

def action_to_p1(tup):
    # Rotate control vector to p1 perspective
    return (tup[1], -tup[0])

def action_to_p2(tup):
    # Rotate control vector to p2 perspective
    return (-tup[1], tup[0])

def states_to_p1(states, length_scaler):
    # Convert states into p1 perspective

    if states is None:
        return None

    new_states = []

    def convert_state(state):
        # Compute new agent state
        new_pos = (KG_BOARD_HEIGHT * length_scaler - state[1], state[0])
        new_vel = action_to_p2((state[2], state[3]))
        return [new_pos[0], new_pos[1], new_vel[0], new_vel[1]]
    
    # Each agent has 4 associated state values
    list_of_slices = zip(*(iter(states),) * 4)
    for slice in list_of_slices:
        new_states.extend(convert_state(slice))

    return tuple(new_states)

def states_to_p2(states, length_scaler):
    # Convert states into p2 perspective

    if states is None:
        return None

    new_states = []

    def convert_state(state):
        # Compute new agent state
        new_pos = (state[1], KG_BOARD_WIDTH * length_scaler - state[0])
        new_vel = action_to_p1((state[2], state[3]))
        return [new_pos[0], new_pos[1], new_vel[0], new_vel[1]]
    
    # Each agent has 4 associated state values
    list_of_slices = zip(*(iter(states),) * 4)
    for slice in list_of_slices:
        new_states.extend(convert_state(slice))

    # Swap locations of p1 and p2 state values so p2 thinks it is p1
    new_states[12:16], new_states[16:20] = new_states[16:20], new_states[12:16]

    return tuple(new_states)

def reward_to_p1(game_states):
    # Reward of +1 if win, -1 is lose, and 0 if tie

    reward = 0

    if KlaskSimulator.GameStates.P1_WIN in game_states:
        reward += 1

    if KlaskSimulator.GameStates.P2_WIN in game_states:
        reward -= 1

    return reward

def reward_to_p2(game_states):
    # Reward of +1 if win, -1 is lose, and 0 if tie

    reward = 0

    if KlaskSimulator.GameStates.P2_WIN in game_states:
        reward += 1

    if KlaskSimulator.GameStates.P1_WIN in game_states:
        reward -= 1

    return reward
