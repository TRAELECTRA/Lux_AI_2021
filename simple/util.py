from collections import namedtuple, deque
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from lux.constants import Constants

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
RESOURCE_TYPES = Constants.RESOURCE_TYPES

def updateMap(nStep: int, \
              nXShift: int, \
              nYShift: int, \
              nTeam: int, \
              nUId: int, \
              updateList: list) -> list:

    # indexing
    # rp  - gameMap[0:2]                  #resource points
    # r   - gameMap[2:5]                  #resource
    # u   - gameMap[5:13]                 #unit
    # c   - ...it only consumes fuels     #city
    # ct  - gameMap[8:12]                 #citytile
    # ccd - gameMap[]                     #roads

    rpStart = 0
    rStart = 2
    uStart = 5
    ctStart = 8

    gameMap: np.ndarray(np.float32) = np.zeros(20, 32, 32)
    cityDict: dict = {}

    for update in updateList:
        cmdList: list[str] = update.split(' ')

        sIdentifier: str = cmdList[0]
        if INPUT_CONSTANTS.RESEARCH_POINTS == sIdentifier:
            team = int(cmdList[1])
            rp = int(cmdList[2])
            idx = rpStart + (team - nTeam) % 2
            value = min(rp, 200) / 200
            gameMap[idx, :] = value

        elif INPUT_CONSTANTS.RESOURCES == sIdentifier:
            rtype = cmdList[1]
            x = int(cmdList[2]) + nXShift
            y = int(cmdList[3]) + nYShift
            amt = int(float(cmdList[4]))
            idx = rStart + {'wood':0, 'coal':1, 'uranium':2}[rtype]
            value = amt / 800
            gameMap[idx, x, y] = value

        elif INPUT_CONSTANTS.UNITS == sIdentifier:
            utype = int(cmdList[1])
            team = int(cmdList[2])
            uid = cmdList[3]
            x = int(cmdList[4])
            y = int(cmdList[5])
            cooldown = float(cmdList[6]) / 6.0
            wood = int(cmdList[7])
            coal = int(cmdList[8])
            uranium = int(cmdList[9])
            resources = (wood + coal + uranium) / 100

            if nUId == uid:
                idx = uStart
                value = (1, resources)
                gameMap[idx:idx+2, x, y] = value
            else:
                idx = uStart + 2
                value = (1, cooldown, resources)
                gameMap[idx:idx+3, x, y] = value

        elif INPUT_CONSTANTS.CITY == sIdentifier:
            team = int(cmdList[1])
            cid: str = cmdList[2]
            fuel = float(cmdList[3])
            lightupkeep = float(cmdList[4])
            cityDict[cid] = min(fuel / lightupkeep, 10) / 10

        elif INPUT_CONSTANTS.CITY_TILES == sIdentifier:
            team = int(cmdList[1])
            cid: str = cmdList[2]
            x = int(cmdList[3]) + nXShift
            y = int(cmdList[4]) + nYShift
            cooldown = float(cmdList[5])
            idx = ctStart + (team - nTeam) % 2 * 2
            value = (1, cityDict[cid])
            gameMap[idx:idx+2, x, y] = value

        elif INPUT_CONSTANTS.ROADS == sIdentifier:
            x = int(cmdList[1])
            y = int(cmdList[2])
            road = float(cmdList[3])

        else:
            print( ':: ERROR :: UPDATEMAP')

    # Day/Night Cycle
    gameMap[17, :] = nStep % 40 / 40
    # Turns
    gameMap[18, :] = nStep / 360
    # Map Size
    gameMap[19, nXShift:32-nXShift, nYShift:32-nYShift] = 1

    return gameMap

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
