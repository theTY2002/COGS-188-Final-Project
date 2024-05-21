import random
from simulator.meld import Meld, MeldType
from simulator.tile import Tile, BONUS_TILES, TILES

class State:
    wall: list[Tile]
    hands: list[list[Tile]]
    melds: list[list[Meld]]
    players: list[int] = list(range(4))

    curr_player: int
    discarded: Tile
    winner: int

    def begin(self):
        self.melds = [[]]*4
        self.curr_player = 0
        
        self.wall = random.sample(TILES, len(TILES))

        self.hands = [ self.__draw(13) for _ in self.players ]
        self.hands[self.curr_player].append(self.__draw())

    def meld(self, hand_tiles: list[Tile], player: int) -> bool:
        # Check type of meld
        # Check if hand_tiles + discard forms a pon, chi, or kan
        # if not, return false
        # 

        # Each agent will be provided each possible meld they could make
        # And asked which they prefer (if any)
        # Then this function will be called by the simulator
        self.melds[player].append(meld)

        for tile in hand_tiles:
            self.hands[player].remove(tile)
    
    def play(self, discard: int):
        # Discard
        self.discarded = self.hands[self.curr_player][discard]
        del self.hands[self.curr_player][discard]

        # Update current player
        self.curr_player = (self.curr_player + 1) % 4

        # Draw tile for current player
        self.hands[self.curr_player].append(self.__draw())


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
                if drawn[i] in BONUS_TILES:
                    drawn[i] = self.wall[-1]
                    del self.wall[-1]

                    removed_bonus = True

        return drawn if n_orig else drawn[0]