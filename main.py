from collections import deque
import numpy as np
import torch
from agents.random import RandomAgent
from learning.agent import DQNAgent
from simulator.game import Game

agent = DQNAgent(True, 0)
# game = Game([RandomAgent(), RandomAgent(), RandomAgent(), agent])

# while (game.step() == False):
#     continue
# if game.winner == 3:
#     agent.end_game(100)
# else:
#     agent.end_game(0)

# agent.discard_trainer.qnetwork_local.load_state_dict(torch.load('discard_checkpoint.pth'))
# agent.meld_trainer.qnetwork_local.load_state_dict(torch.load('meld_checkpoint.pth'))

winrate = 0.0
total_wins = 0
average_steps = 0

steps = []
steps_window = deque(maxlen=100)

wins = []
wins_window = deque(maxlen=100)

scores = []
scores_window = deque(maxlen=100)
for i in range(2000):
    game = Game([RandomAgent(), RandomAgent(), RandomAgent(), agent])
    agent.score = 0
    curr_steps = 0
    while (game.step() == False):
        curr_steps += 1
        continue
        
    if game.winner == 3:
        agent.end_game(100)
        total_wins += 1
    else:
        agent.end_game(0)

    steps_window.append(curr_steps)
    steps.append(curr_steps)

    steps_window.append(total_wins)
    steps.append(total_wins)

    scores_window.append(agent.score)
    scores.append(agent.score)

    print(f'\rEpisode {i}\nAverage Steps: {np.mean(steps_window)} \nAverage Wins: {np.mean(wins_window)} \nAverage Score: {np.mean(scores_window)} \n', end="")
    if i % 100 == 0:
        print(f'\rEpisode {i}\nAverage Steps: {np.mean(steps_window)} \nAverage Wins: {np.mean(wins_window)} \nAverage Score: {np.mean(scores_window)} \n')

    # print(f'\rEpisode {i}\tAverage Wins: {np.mean(wins_window)}', end="")
    # if i % 100 == 0:
    #     print(f'\rEpisode {i}\tAverage Wins: {np.mean(wins_window)}')

    # print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window)}', end="")
    # if i % 100 == 0:
    #     print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window)}')
    # winrate = total_wins / (i + 1)
    # average_steps = curr_steps / (i + 1)
    # print("Avg Steps: " + str(average_steps))
    # print("Avg Winrate: " + str(winrate))
    # print("Avg Score: " + str(agent.score / (i + 1)))

torch.save(agent.discard_trainer.qnetwork_local.state_dict(), 'discard_checkpoint.pth')
torch.save(agent.meld_trainer.qnetwork_local.state_dict(), 'meld_checkpoint.pth')