from pettingzoo import ParallelEnv
from copy import copy
import functools
import numpy as np
from gymnasium.spaces import Discrete, Box
from modules.Klask_Simulator.klask_simulator import KlaskSimulator
from klask_environment import klask_actions

def env(render_mode="frame"):
    env = CustomEnvironment(render_mode)
    return env

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, render_mode="frame", max_timestep=600):
        # Attributes should lnot be changed after initialization
        self.possible_agents = ["agent1", "agent2"]
        self.timestep = None

        self.max_timestep = max_timestep

        self.sim = KlaskSimulator(render_mode=render_mode)
        self.frame = None

    def reset(self, seed=None, options=None):
        # Must setup environment so render(), step(), and observe() can be called without issues.
        
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.frame, _, agent_states = self.sim.reset()

        observations = {
            "agent1": agent_states,
            "agent2": agent_states
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):

        agent1_action = klask_actions[actions["agent1"]]
        agent2_action = klask_actions[actions["agent2"]]

        # Execute actions ...
        self.frame, game_states, agent_states = self.sim.step(agent1_action, agent2_action)

        # Check termination conditions ...
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if KlaskSimulator.GameStates.P1_WIN in game_states:
            rewards = {"agent1": 1, "agent2": -1}
            terminations = {a: True for a in self.agents}
        elif KlaskSimulator.GameStates.P2_WIN in game_states:
            rewards = {"agent1": -1, "agent2": 1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions ... (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_timestep:
            rewards = {"agent1": 0, "agent2": 0}
            truncations = {"agent1": True, "agent2": True}
        self.timestep += 1

        # Get observations
        observations = {
            "agent1": agent_states,
            "agent2": agent_states
        }

        # Get dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # Renders an environment
        return self.frame

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=-np.inf, high=np.inf, shape=[24,])

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
