import torch
from agents.random import RandomAgent
from learning.agent import DQNAgent
from simulator.game import Game

agent = DQNAgent(True, 0)
game = Game([RandomAgent(), RandomAgent(), RandomAgent(), agent])
while (game.step() == False):
    continue
if game.winner == 3:
    agent.end_game(100)
else:
    agent.end_game(0)

torch.save(agent.discard_trainer.qnetwork_local.state_dict(), 'discard_checkpoint.pth')
torch.save(agent.meld_trainer.qnetwork_local.state_dict(), 'meld_checkpoint.pth')