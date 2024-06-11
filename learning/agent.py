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
BUFFER_SIZE = 100000
BATCH_SIZE = 64
LR = 0.0005
GAMMA = 0.99
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
    def __init__(self, train: bool, seed: int):
        self.train = train
        self.epsilon = EPSILON
        #self.seed = random.seed(seed)
        self.score = 0
        self.discard_network = MahjongNetwork(DISCARD_CHANNELS, DISCARD_OUTPUT, SEED).to(device)
        self.discard_trainer = DQNTrainer(self.discard_network, BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, EPSILON, SEED)
        self.meld_network = MahjongNetwork(MELD_CHANNELS, MELD_OUTPUT, SEED).to(device)
        self.meld_trainer = DQNTrainer(self.meld_network, BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, EPSILON, SEED)

        self.discard_state = torch.zeros(34, 4, 9)
        self.discard_action = 0
        self.meld_state = torch.zeros(34, 4, 10)
        self.meld_action = 0

        # self.qnetwork_local = MahjongNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = MahjongNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # self.memory = ReplayBuffer(action_size, buffer_size=100000, batch_size=64, seed=seed)

    def end_game(self, reward):
        self.discard_trainer.end_step(self.discard_state, self.discard_action, reward, torch.zeros(34, 4, 9), True)
        self.meld_trainer.end_step(self.meld_state, self.meld_state, reward, torch.zeros(34, 4, 10), True)

    def discard_reward(self, hand: list[Tile], action: int):
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
                return 3
            case 1:
                return 1
            case _:
                return 0

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
            #tensor = torch.stack((tensor, meld_tensor))

        discard_tensors = []
        for player_discards in ordered_discards:
            discard_tensor = tiles_to_tensor(player_discards)
            discard_tensors.append(discard_tensor)

            #tensor = torch.stack((tensor, discard_tensor))

        tensor = torch.stack(([hand_tensor] + meld_tensors + discard_tensors)).to(device)

        next_state = tensor.to(device)

        #Train before taking next action
        if (self.train):
            reward = self.discard_reward(hand, self.discard_action)
            done = False
            self.discard_trainer.step(self.discard_state, self.discard_action, reward, next_state, done)
            self.discard_state = next_state
            self.score += reward

        action = self.discard_trainer.act(next_state, hand)
        self.discard_action = action

        print("Score: ")
        print(self.score)

        discard_tile = index_to_tile(action)
        return hand.index(discard_tile)

    def meld_reward(self, hand, action_meld):
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
                reward += 3
            case 1:
                reward += 1
            case _:
                reward += 0

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
                reward += 3
            case 1:
                reward += 1
            case _:
                reward += 0

        return reward

    def choose_meld(self, available_melds: list[Meld], hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int | None:
        #tensor = torch.empty(34, 4)

        #To make this easier on myself, I'm stacking the stolen tile and possible meld at the bottom

        hand_tensor = tiles_to_tensor(hand)
        #tensor = hand_tensor

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
                player_tiles = player_tiles + meld_tiles
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
        action_meld_tensor = torch.zeros(34, 4)
        prob = random.uniform(0, 1)

        
        if (prob < self.epsilon):
            random_meld = np.random.choice(available_melds)
            stolen_meld_tiles = random_meld.tiles
            stolen_meld = random_meld.tiles + [random_meld.discard]
            stolen_meld_tensor = tiles_to_tensor(stolen_meld)
            action_meld = stolen_meld_tiles
            action_meld_tensor = stolen_meld_tensor
        else:
            for meld in available_melds:
                stolen_meld_tiles = meld.tiles
                stolen_meld = meld.tiles + [meld.discard]
                stolen_meld_tensor = tiles_to_tensor(stolen_meld)
                #tensor = torch.stack((tensor, stolen_meld_tensor))

                tensor = torch.stack(([hand_tensor] + meld_tensors + discard_tensors + [stolen_meld_tensor])).to(device)
                next_state = tensor.to(device)

                q_values = self.meld_trainer.act_meld(next_state).squeeze(0)

                if (q_values[0] > max_q):
                    max_q = q_values[0]
                    action_meld = stolen_meld_tiles
                    action_meld_tensor = stolen_meld_tensor

                tensor = tensor[:9, :, :]

        if (self.train):
            reward = self.meld_reward(hand, action_meld)
            done = False
            self.meld_trainer.step(self.meld_state, self.meld_action, reward, next_state, done)
            self.meld_state = next_state
            self.score += reward

        if (action_meld == None):
            action = 1
        else:
            action = 0
        self.meld_action = action

        
        #Put the optimal move back in the state
        tensor = torch.stack(([hand_tensor] + meld_tensors + discard_tensors + [action_meld_tensor])).to(device)
        next_state = tensor

        print("Score: ")
        print(self.score)
        return action