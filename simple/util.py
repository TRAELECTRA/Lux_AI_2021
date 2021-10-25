# from collections import namedtuple, deque
# import random
# import numpy as np
# import math
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
#
# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 1
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward')
#
#
# def get_inputs(game_state):
#     # Teh shape of the map
#     w, h = game_state.map.width, game_state.map.height
#     # The map of ressources
#     M = [
#         [0 if game_state.map.map[j][i].resource == None else game_state.map.map[j][i].resource.amount for i in range(w)]
#         for j in range(h)]
#
#     M = np.array(M).reshape((h, w, 1))
#
#     # The map of units features
#     U_player = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
#     units = game_state.player.units
#     for i in units:
#         U_player[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]
#     U_player = np.array(U_player)
#
#     U_opponent = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
#     units = game_state.opponent.units
#     for i in units:
#         U_opponent[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]
#
#     U_opponent = np.array(U_opponent)
#
#     # The map of cities featrues
#     e = game_state.player.cities
#     C_player = [[[0, 0, 0] for i in range(w)] for j in range(h)]
#     for k in e:
#         citytiles = e[k].citytiles
#         for i in citytiles:
#             C_player[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
#     C_player = np.array(C_player)
#
#     e = game_state.opponent.cities
#     C_opponent = [[[0, 0, 0] for i in range(w)] for j in range(h)]
#     for k in e:
#         citytiles = e[k].citytiles
#         for i in citytiles:
#             C_opponent[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
#     C_opponent = np.array(C_opponent)
#
#     # stacking all in one array
#     E = np.dstack([M, U_opponent, U_player, C_opponent, C_player])
#     return E
#
# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)
#
#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#
# class DQNModel(nn.Module):
#     def __init__(self, s=12, output_size=9):
#         # ouput_size    == number of actions
#         # input_size(s) == 12, 16, 24, 32 tiles long
#         super(DQNModel, self).__init__()
#         self.conv1 = nn.Conv2d(17, 32, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(64)
#
#         get_size = lambda size, kernel_size=5, stride=2 : (size - (kernel_size - 1) - 1) // stride + 1
#         convs = get_size(get_size(get_size(s)))
#         linear_input_size = convs * convs * 64
#         self.head = nn.Linear(linear_input_size, output_size)
#
#     def forward(self, x):
#         x = x.to(device)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.view(x.size(0), -1)))
#
# def select_action(state):
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
#
#
# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                        if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Debug
#     print("non_final_mask", non_final_mask)
#     print("non_final_next_states", non_final_next_states)
#     print("batch(s,a,r)", state_batch, action_batch, reward_batch)
#     print("state_action_values", state_action_values)
#     print("next_state_values", next_state_values)
#     print("next_state_values(single)", next_state_values[non_final_mask])
#     print("expected_state_action_values", expected_state_action_values)
#     print("loss", loss)
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()
#
