# Use Klask_Simulator to train a DQN agent via policy self-play

from modules.Klask_Simulator.klask_simulator import KlaskSimulator
from modules.Klask_Simulator.klask_constants import KG_BOARD_HEIGHT, KG_BOARD_WIDTH
from KeyboardController import KeyboardController

import pygame

import numpy as np

# TODO: refactor into an enviornment class
# TODO: define action space

def numpy_to_pygame(arr):
    # Convert image stored in np array to pygame Surface
    return pygame.surfarray.make_surface(arr.swapaxes(0,1))

def frame_to_p1(arr):
    # Rotate frame to p1 perspective
    return np.rot90(frame)

def action_to_p1(tup):
    # Rotate control vector to p1 perspective
    return (tup[1], -tup[0])

def frame_to_p2(arr):
    # Rotate frame to p2 perspective
    return np.rot90(frame, -1)

def action_to_p2(tup):
    # Rotate control vector to p2 perspective
    return (-tup[1], tup[0])

def print_states(states):
    # Display agent states in human readable form
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
                    "puck1_pos_x",      # idx 12
                    "puck1_pos_y",
                    "puck1_vel_x",
                    "puck1_vel_y",
                    "puck2_pos_x",      # idx 16
                    "puck2_pos_y",
                    "puck2_vel_x",
                    "puck2_vel_y",
                    "ball_pos_x",
                    "ball_pos_y",
                    "ball_vel_x",
                    "ball_vel_y"]
    for el in list(zip(state_names, states)):
        print(el)

def states_to_p1(states, length_scaler):
    # Convert states into p1 perspective

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

# Initialize the simulator
sim = KlaskSimulator(render_mode="human")

sim.reset()

# Initialize the controllers
force = 0.005
p1 = KeyboardController(force)
p2 = KeyboardController(force)

# Initialize the simulator
sim = KlaskSimulator(render_mode="frame")

frame, game_states, agent_states = sim.reset()

w, h, _ = frame.shape
screen = pygame.display.set_mode((w*2, h))
pygame.display.set_caption('Klask Simulator')

running = True

while running:
    # Check the event queue (only accessable if render_mode="human", is optional)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

        # Handle inputs for the keyboard controller
        if event.type == pygame.KEYDOWN:
            # P1
            if event.key == pygame.K_a:
                p1.keyLeft_pressed()
            if event.key == pygame.K_d:
                p1.keyRight_pressed()
            if event.key == pygame.K_w:
                p1.keyUp_pressed()
            if event.key == pygame.K_s:
                p1.keyDown_pressed()

            # P2
            if event.key == pygame.K_LEFT:
                p2.keyLeft_pressed()
            if event.key == pygame.K_RIGHT:
                p2.keyRight_pressed()
            if event.key == pygame.K_UP:
                p2.keyUp_pressed()
            if event.key == pygame.K_DOWN:
                p2.keyDown_pressed()
        if event.type == pygame.KEYUP:
            # P1
            if event.key == pygame.K_a:
                p1.keyLeft_released()
            if event.key == pygame.K_d:
                p1.keyRight_released()
            if event.key == pygame.K_w:
                p1.keyUp_released()
            if event.key == pygame.K_s:
                p1.keyDown_released()

            # P2
            if event.key == pygame.K_LEFT:
                p2.keyLeft_released()
            if event.key == pygame.K_RIGHT:
                p2.keyRight_released()
            if event.key == pygame.K_UP:
                p2.keyUp_released()
            if event.key == pygame.K_DOWN:
                p2.keyDown_released()

    frame, game_states, agent_states = sim.step(action_to_p1(p1.getAction()), action_to_p2(p2.getAction()))

    p1_frame = frame_to_p1(frame)
    p1_states = states_to_p1(agent_states, sim.length_scaler)

    p2_frame = frame_to_p2(frame)
    p2_states = states_to_p2(agent_states, sim.length_scaler)

    screen.blit(numpy_to_pygame(p1_frame), (0,0))
    screen.blit(numpy_to_pygame(p2_frame), (w,0))
    pygame.display.flip()

    print(game_states)
    # print_states(agent_states)
    # print_states(p1_states)
    print_states(p2_states)
    print()
    print()

sim.close()
