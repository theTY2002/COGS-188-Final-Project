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

# torch.save(agent.discard_trainer.qnetwork_local.state_dict(), 'discard_checkpoint.pth')
# torch.save(agent.meld_trainer.qnetwork_local.state_dict(), 'meld_checkpoint.pth')

agent.discard_trainer.qnetwork_local.load_state_dict(torch.load('discard_checkpoint.pth'))
agent.meld_trainer.qnetwork_local.load_state_dict(torch.load('meld_checkpoint.pth'))

winrate = 0.0
total_wins = 0
average_steps = 0
for i in range(2000):
    game = Game([RandomAgent(), RandomAgent(), RandomAgent(), agent])
    curr_steps = 0
    while (game.step() == False):
        curr_steps += 1
        continue
        
    if game.winner == 3:
        agent.end_game(100)
        total_wins += 1
    else:
        agent.end_game(0)
    winrate = total_wins / (i + 1)
    average_steps = curr_steps / (i + 1)
    print("Avg Steps: " + str(average_steps))
    print("Avg Winrate: " + str(winrate))
    print("Avg Score: " + str(agent.score / (i + 1)))