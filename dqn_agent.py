# PyTorch implementation of a DQN agent and associated datastructures
# Refactored variant of https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import random
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_Agent:
    def __init__(self, state_size, action_size):
        # Required parameters
        self.n_actions = action_size
        self.n_observations = state_size
        
        # Tunable parameters
        self.BATCH_SIZE = 128   # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.99       # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = 0.9    # EPS_START is the starting value of epsilon
        self.EPS_END = 0.05     # EPS_END is the final value of epsilon
        self.EPS_DECAY = 1000   # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005        # TAU is the update rate of the target network
        self.LR = 1e-4          # LR is the learning rate of the ``AdamW`` optimizer

        # Component networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        # Replay buffer
        self.memory = ReplayMemory(5000)

        self.steps_done = 0

    def remember(self, state, action, next_state, reward):
        # Convert to tensor if needed
        if not torch.is_tensor(state) and state is not None:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not torch.is_tensor(action) and action is not None:
            torch.tensor([[action]], device=self.device, dtype=torch.long)
        if not torch.is_tensor(next_state) and next_state is not None:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not torch.is_tensor(reward) and reward is not None:
            reward = torch.tensor([reward], device=self.device)

        # Store transition
        self.memory.push(state, action, next_state, reward)

    def apply_policy(self, state):
        # Used for inference
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)         
            return self.policy_net(state).max(1).indices.view(1, 1)

    def select_action(self, state):
        # Used for training
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.apply_policy(state)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        # Used for training
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, PATH):
        # Save policy to disk for later inference
        torch.save(self.policy_net.state_dict(), PATH)

    def load(self, PATH):
        # Load policy from disk for later inference
        self.policy_net.load_state_dict(torch.load(PATH))

    def close(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
