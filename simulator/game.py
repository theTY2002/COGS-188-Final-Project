import random
from simulator.meld import Meld
from simulator.tile import Tile, bonus_tiles, tiles

class Game:
    wall: list[Tile]
    hands: list[list[Tile]]
    melds: list[list[Meld]]

    player: int
    players: list[int] = list(range(4))

    def begin(self):
        self.melds = [[]]*4
        self.player = 0
        
        self.wall = random.sample(tiles, len(tiles))

        self.hands = [ self.__draw(13) for _ in self.players ]

    def draw(self) -> list[Tile]:
        self.hands[self.player].append(self.__draw())

        return self.hands[self.player]
    
    def discard(self, index: int):
        del self.hands[self.player][index]
        self.player = (self.player + 1) % 4

    def __draw(self, n: int = None) -> Tile | list[Tile] | None:
        n_orig = n
        if not n:
            n = 1

        if n > len(self.wall):
            return None

        drawn = self.wall[:n]
        del self.wall[:n]

        removed_bonus = True
        while removed_bonus:
            removed_bonus = False

            for i in range(len(drawn)):
                if drawn[i] in bonus_tiles:
                    drawn[i] = self.wall[-1]
                    del self.wall[-1]

                    removed_bonus = True

        return drawn if n_orig else drawn[0]