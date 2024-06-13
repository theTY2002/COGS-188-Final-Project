from learning.models import MahjongNetwork
from learning.q_learning import DQNTrainer, ReplayBuffer
from simulator.agent import Agent
from simulator.melds import Meld, MeldType
from simulator.tiles import Tile
from simulator.util import *


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

device = torch.device("mps")

DISCARD_CHANNELS = 9
DISCARD_OUTPUT = 34
MELD_CHANNELS = 10
MELD_OUTPUT = 2
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99
TAU = 0.01
EPSILON = 0.1
SEED = 0

#Discard Model

# Inputs:
    # Hand: 34x4
    # Meld: 34x4x4
    # Discard: 34x4x4

# Output:
    # 34: one-hot


# Meld model:

# Inputs:
    # Stolen Tile: 34
    # Rest of meld (in hand): 34x3
    # Hand: 34x4
    # Meld: 34x4x4
    # Discard: 34x4x4

# Output size:
    # 2 outputs: take and not take

# Choose highest probability

class DQNAgent(Agent):
    def __init__(self, train: bool, discard_model: MahjongNetwork, meld_model: MahjongNetwork, seed: int):
        self.train = train
        self.epsilon = EPSILON
        #self.seed = random.seed(seed)
        self.score = 0
        self.discard_network = discard_model.to(device)
        self.discard_trainer = DQNTrainer(self.discard_network, BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, TAU, EPSILON, SEED)
        self.meld_network = meld_model.to(device)
        self.meld_trainer = DQNTrainer(self.meld_network, BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, TAU, EPSILON, SEED)

        self.discard_state = torch.zeros(9, 34, 4)
        self.discard_last_reward = 0
        self.discard_action = 0
        self.meld_state = []
        self.meld_last_reward = 0
        self.meld_action = 0

        # self.qnetwork_local = MahjongNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = MahjongNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # self.memory = ReplayBuffer(action_size, buffer_size=100000, batch_size=64, seed=seed)

    def end_game(self, reward):
        self.discard_trainer.end_step(self.discard_state, self.discard_action, reward, torch.zeros(9, 34, 4), True)
        self.score += self.discard_last_reward
        for i, state in enumerate(self.meld_state):
                self.meld_trainer.step(state, 0 if i == self.meld_action else 1, reward, torch.zeros(10, 34, 4), True)
        self.score += self.meld_last_reward
        self.score += reward

    def discard_reward(self, hand: list[Tile], action: int):
        if (len(hand) == 0):
            return 0
        discard_tile = index_to_tile(action)
        hand_copy = hand.copy()
        hand_copy.remove(discard_tile)
        counts = tiles_as_counts(hand_copy)

        match discard_tile:
            case SuitedTile(suit, rank):
                avail = []

                # Check for sets
                match counts[discard_tile]:
                    case 2:
                        avail.append(Meld(MeldType.PON, [discard_tile] * 2, discard_tile))
                    case 3:
                        avail.append(Meld(MeldType.PON, [discard_tile] * 2, discard_tile))
                        avail.append(Meld(MeldType.KAN, [discard_tile] * 3, discard_tile))
                    
                # Check for chi
                candidates = [ SuitedTile(suit, rank + i) for i in range(-2, 3) ]
                if counts[candidates[3]] > 0:
                    if counts[candidates[4]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[3], candidates[4]], candidates[2]))
                    if counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[1], candidates[3]], candidates[2]))
                if counts[candidates[0]] > 0 and counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[0], candidates[1]], candidates[2]))
                
            case _:
                match counts[discard_tile]:
                    case 2:
                        avail = [
                            Meld(MeldType.PON, [discard_tile] * 2, discard_tile)
                        ]
                    case 3:
                        avail = [
                            Meld(MeldType.PON, [discard_tile] * 2, discard_tile),
                            Meld(MeldType.KAN, [discard_tile] * 3, discard_tile)
                        ]
                    case _:
                        avail = []
        
        match (len(avail)):
            case 0:
                return 5
            case _:
                return -1

    def choose_discard(self, hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int:
        # tensor = torch.empty(34, 4)

        hand_tensor = tiles_to_tensor(hand)
        # tensor = hand_tensor

        ordered_melds = melds[player:] + melds[:player]
        ordered_discards = discards[player:] + discards[:player]

        meld_tensors = []

        for player_melds in ordered_melds:
            player_tiles = []
            for meld in player_melds:
                #Skip flowers
                if meld.type == MeldType.FLOWER:
                    continue
                if meld.discard != None:
                    meld_tiles = meld.tiles + [meld.discard]
                else:
                    meld_tiles = meld.tiles
                player_tiles = player_tiles + meld_tiles
            meld_tensor = tiles_to_tensor(player_tiles)
            meld_tensors.append(meld_tensor)

        discard_tensors = []
        for player_discards in ordered_discards:
            discard_tensor = tiles_to_tensor(player_discards)
            discard_tensors.append(discard_tensor)


        tensor = torch.stack(([hand_tensor] + meld_tensors + discard_tensors)).to(device)

        next_state = tensor.to(device)

        #Train before taking next action
        if (self.train):
            done = False
            self.discard_trainer.step(self.discard_state, self.discard_action, self.discard_last_reward, next_state.detach().cpu(), done)
            self.score += self.discard_last_reward

        # self.discard_hand = hand
        action = self.discard_trainer.act(next_state, hand)
        self.discard_last_reward = self.discard_reward(hand, action)
        self.discard_state = next_state.detach().cpu()
        self.discard_action = action

        # print("Score: ")
        # print(self.score)

        discard_tile = index_to_tile(action)
        return hand.index(discard_tile)

    def meld_reward(self, hand: list[Tile], action_meld: list[Tile]):
        reward = 0
        if (action_meld == None):
            return reward
        meld_tile = action_meld[0]
        hand_copy = hand.copy()
        hand_copy.remove(meld_tile)
        counts = tiles_as_counts(hand_copy)

        match meld_tile:
            case SuitedTile(suit, rank):
                avail = []

                # Check for sets
                match counts[meld_tile]:
                    case 2:
                        avail.append(Meld(MeldType.PON, [meld_tile] * 2, meld_tile))
                    case 3:
                        avail.append(Meld(MeldType.PON, [meld_tile] * 2, meld_tile))
                        avail.append(Meld(MeldType.KAN, [meld_tile] * 3, meld_tile))
                    
                # Check for chi
                candidates = [ SuitedTile(suit, rank + i) for i in range(-2, 3) ]
                if counts[candidates[3]] > 0:
                    if counts[candidates[4]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[3], candidates[4]], candidates[2]))
                    if counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[1], candidates[3]], candidates[2]))
                if counts[candidates[0]] > 0 and counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[0], candidates[1]], candidates[2]))
                
            case _:
                match counts[meld_tile]:
                    case 2:
                        avail = [
                            Meld(MeldType.PON, [meld_tile] * 2, meld_tile)
                        ]
                    case 3:
                        avail = [
                            Meld(MeldType.PON, [meld_tile] * 2, meld_tile),
                            Meld(MeldType.KAN, [meld_tile] * 3, meld_tile)
                        ]
                    case _:
                        avail = []
            
        match (len(avail)):
            case 0:
                reward += 5
            case _:
                reward += -1

        meld_tile = action_meld[1]
        hand_copy = hand.copy()
        hand_copy.remove(meld_tile)
        counts = tiles_as_counts(hand_copy)

        match meld_tile:
            case SuitedTile(suit, rank):
                avail = []

                # Check for sets
                match counts[meld_tile]:
                    case 2:
                        avail.append(Meld(MeldType.PON, [meld_tile] * 2, meld_tile))
                    case 3:
                        avail.append(Meld(MeldType.PON, [meld_tile] * 2, meld_tile))
                        avail.append(Meld(MeldType.KAN, [meld_tile] * 3, meld_tile))
                    
                # Check for chi
                candidates = [ SuitedTile(suit, rank + i) for i in range(-2, 3) ]
                if counts[candidates[3]] > 0:
                    if counts[candidates[4]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[3], candidates[4]], candidates[2]))
                    if counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[1], candidates[3]], candidates[2]))
                if counts[candidates[0]] > 0 and counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[0], candidates[1]], candidates[2]))
                
            case _:
                match counts[meld_tile]:
                    case 2:
                        avail = [
                            Meld(MeldType.PON, [meld_tile] * 2, meld_tile)
                        ]
                    case 3:
                        avail = [
                            Meld(MeldType.PON, [meld_tile] * 2, meld_tile),
                            Meld(MeldType.KAN, [meld_tile] * 3, meld_tile)
                        ]
                    case _:
                        avail = []
            
        match (len(avail)):
            case 0:
                reward += 5
            case _:
                reward += -1

        return reward

    def choose_meld(self, available_melds: list[Meld], hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int | None:
        #To make this easier on myself, I'm stacking the stolen tile and possible meld at the bottom

        hand_tensor = tiles_to_tensor(hand)

        ordered_melds = melds[player:] + melds[:player]
        ordered_discards = discards[player:] + discards[:player]

        meld_tensors = []

        for player_melds in ordered_melds:
            player_tiles = []
            for meld in player_melds:
                if meld.discard != None:
                    meld_tiles = meld.tiles + [meld.discard]
                else:
                    meld_tiles = meld.tiles
                player_tiles.extend(meld_tiles)
            meld_tensor = tiles_to_tensor(player_tiles)
            meld_tensors.append(meld_tensor)
            #tensor = torch.stack((tensor, meld_tensor))

        discard_tensors = []
        for player_discards in ordered_discards:
            discard_tensor = tiles_to_tensor(player_discards)
            discard_tensors.append(discard_tensor)
            #tensor = torch.stack((tensor, discard_tensor))

        #Potential melds
        max_q = -float('inf')
        action_meld = None
        action_meld_index = None
        prob = random.uniform(0, 1)

        avail_meld_tensors = []
        for meld in available_melds:
            stolen_meld = meld.tiles + [meld.discard]
            stolen_meld_tensor = tiles_to_tensor(stolen_meld)

            tensor = torch.stack(([hand_tensor] + meld_tensors + discard_tensors + [stolen_meld_tensor])).to(device)
            next_state = tensor.to(device)
            avail_meld_tensors.append(next_state)

        if (prob < self.epsilon):
            action_meld_index = np.random.choice(len(available_melds))
            random_meld = available_melds[action_meld_index]
            action_meld = random_meld.tiles
            stolen_meld = random_meld.tiles + [random_meld.discard]
        else:
            for i, meld in enumerate(available_melds):
                next_state = avail_meld_tensors[i]

                q_values = self.meld_trainer.act_meld(next_state).squeeze(0)

                # If Yes > No and Yes > current max q
                if (q_values[0] > q_values[1]) and (q_values[0] > max_q):
                    max_q = q_values[0]
                    action_meld_index = i
                    action_meld = meld.tiles

        if (action_meld == None):
            action = None
            next_state = avail_meld_tensors[0]
        else:
            action = action_meld_index
            #Put the optimal move back in the state
            next_state = avail_meld_tensors[action_meld_index]

        if (self.train):
            for i, state in enumerate(self.meld_state):
                self.meld_trainer.step(state, 0 if i == self.meld_action else 1, self.meld_last_reward, next_state.detach().cpu(), False)
            self.score += self.meld_last_reward
        
        self.meld_last_reward = self.meld_reward(hand, action_meld)
        self.meld_state = avail_meld_tensors
        self.meld_action = action

        # print("Score: ")
        # print(self.score)
        return action