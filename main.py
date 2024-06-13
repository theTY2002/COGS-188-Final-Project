from collections import deque
import numpy as np
import torch
from agents.random import RandomAgent
from learning.agent import DQNAgent
from learning.models import MahjongNetwork
from simulator.game import Game
import csv

from simulator.melds import MeldType

DISCARD_CHANNELS = 9
DISCARD_OUTPUT = 34
MELD_CHANNELS = 10
MELD_OUTPUT = 2
SEED = 0

discard_model = MahjongNetwork(DISCARD_CHANNELS, DISCARD_OUTPUT, SEED)
meld_model = MahjongNetwork(MELD_CHANNELS, MELD_OUTPUT, SEED)

agent1 = DQNAgent(True, discard_model=discard_model, meld_model=meld_model, seed=0)
agent2 = DQNAgent(True, discard_model=discard_model, meld_model=meld_model, seed=0)
# game = Game([RandomAgent(), RandomAgent(), RandomAgent(), agent])

# while (game.step() == False):
#     continue
# if game.winner == 3:
#     agent.end_game(100)
# else:
#     agent.end_game(0)

# agent1.discard_trainer.qnetwork_local.load_state_dict(torch.load('discard_checkpoint.pth'))
# agent1.meld_trainer.qnetwork_local.load_state_dict(torch.load('meld_checkpoint.pth'))
# agent2.discard_trainer.qnetwork_local.load_state_dict(torch.load('discard_checkpoint.pth'))
# agent2.meld_trainer.qnetwork_local.load_state_dict(torch.load('meld_checkpoint.pth'))

winrate = 0.0
average_steps = []
average_scores = []


myfile = open("data.csv", 'a', newline='')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#wr.writerow(["Episode", "Steps", "Wins", "Scores"])

steps_window = deque(maxlen=100)
wins_window = deque(maxlen=100)
scores_window = deque(maxlen=200)

for i in range(2000):
    game = Game([RandomAgent(), RandomAgent(), agent1, agent2])
    agent1.score = 0
    agent2.score = 0
    curr_steps = 0

    done = False
    while not done:
        done = game.step()
        curr_steps += 1

    if (game.winner == 2):
        agent1.end_game(100)
        agent2.end_game(-2)
        wins_window.append(1)
    elif (game.winner == 3):
        agent2.end_game(100)
        agent1.end_game(-2)
        wins_window.append(1)
    else:
        agent1.end_game(-2)
        agent2.end_game(-2)
        wins_window.append(0)

    steps_window.append(curr_steps)

    scores_window.append(max(agent1.score, agent2.score))

    print(f'\rEpisode {i}\nAverage Steps: {np.mean(steps_window)} \nAverage Wins: {np.mean(wins_window)} \nAverage Score: {np.mean(scores_window)} \n', end="")
    if i % 100 == 0:
        torch.save(agent1.discard_trainer.qnetwork_local.state_dict(), 'discard_checkpoint.pth')
        torch.save(agent1.meld_trainer.qnetwork_local.state_dict(), 'meld_checkpoint.pth')

    wr.writerow([i, np.mean(steps_window), np.mean(wins_window), np.mean(scores_window)])

myfile.close()