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