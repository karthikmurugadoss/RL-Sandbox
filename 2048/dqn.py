from collections import namedtuple, deque
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from engine import GameEngine


class DQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_size, fc2_size, fc3_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.bn3 = nn.BatchNorm1d(fc3_size)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(fc3_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = self.fc1(state)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return self.fc4(x)


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, epsilon):
    # Exploration
    if random.uniform(0, 1) < epsilon:
        return random.randrange(4)  # 4 actions

    # Exploitation
    else:
        state = torch.from_numpy(state).float().unsqueeze(0)
        policy_net.eval()
        with torch.no_grad():
            action = int(policy_net(state).max(1)[1].view(1, 1)[0][0])
        policy_net.train()
        return action


def optimize_model(policy_net, memory, optimizer, batch_size, gamma):
    policy_net.train()

    if len(memory) < batch_size:
        # If there isn't enough memory available, we have nothing to train on
        # Therefore just return.
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state_batch = np.stack(batch.state, axis=0)
    state_batch = torch.from_numpy(state_batch).float()
    action_batch = np.array(batch.action)
    action_batch = torch.from_numpy(action_batch).unsqueeze(1)
    reward_batch = np.array(batch.reward)
    reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).float()

    q_expected = policy_net(state_batch).gather(1, action_batch)
    loss = F.mse_loss(q_expected, reward_batch)

    # Minimize the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def transform_state(state):
    state = state.copy()
    state = np.reshape(state, -1)
    state[state == 0] = 1
    state = np.log2(state)
    state = state.astype(int)
    new_state = np.reshape(np.eye(18)[state], -1)
    return new_state


def untransform_state(state):
    """
    Mainly used for debugging
    """
    reshaped_state = np.reshape(state, (4, 4, 18))
    indices = np.argmax(reshaped_state, axis=2)
    original_state = 2**indices
    original_state[original_state == 1] = 0
    return original_state


def get_log2_reward(reward):
    if reward == 0:
        return 0
    else:
        return np.log2(reward)


NEGATIVE_REWARD = -10

# Hyperparameters
EPISODES = 50000
BATCH_SIZE = 1024
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


policy_net = DQN(
    state_size=4 * 4 * 18,
    action_size=4,
    fc1_size=1024,
    fc2_size=1024,
    fc3_size=1024,
)
optimizer = optim.RMSprop(policy_net.parameters())
# policy_net.to("cpu")

# Memory is limited to storing 50K examples
# Since this is a queue, we will be storing only the most recent games
# This number should probably depend on the number of moves per episode
memory = ReplayMemory(50000)

steps_done = 0
max_val = 0
total_scores = []

# We iterate over a number of episodes
for episode in range(EPISODES):
    # Derive epsilon
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)

    # Initialize the game
    game_engine = GameEngine()

    # Each episode lasts a full game i.e. until the game over state is reached
    while not game_engine.is_complete():
        moved = False
        while not moved:
            state = game_engine.matrix
            state = transform_state(state)

            action = select_action(state, policy_net, epsilon)
            action_ = game_engine.ACTIONS[action]

            next_state, reward, _ = game_engine.action(action_)
            next_state = transform_state(next_state)
            reward = get_log2_reward(reward)

            if not game_engine.is_complete() and np.array_equal(next_state, state):
                reward = NEGATIVE_REWARD
            else:
                moved = True

            if game_engine.is_complete():
                next_state = None

            memory.push(state, action, reward, next_state)

        game_engine.add_new_value()

        if game_engine.is_complete():
            next_state = None
            break

    steps_done += 1

    total_scores.append(game_engine.score)
    max_val = max(max_val, np.max(game_engine.matrix))
    avg_score = np.round(np.mean(total_scores[-50:]), 2)
    print(
        f"EPISODE: {episode}\tSCORE: {game_engine.score}\tAVERAGE SCORE: {avg_score}\tMAX VALUE: {max_val}\tEPSILON: {epsilon}"
    )

    optimize_model(policy_net, memory, optimizer, BATCH_SIZE, GAMMA)
