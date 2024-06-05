import random
from simulator.agent import Agent
from simulator.melds import Meld
from simulator.tiles import Tile

class RandomAgent(Agent):
    def choose_discard(self, hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int:
        return random.randrange(len(hand))
    
    def choose_meld(self, available_melds: list[Meld], hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int | None:
        return random.choice([None] + list(range(len(available_melds))))