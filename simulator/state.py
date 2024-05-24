from dataclasses import dataclass
import random

from simulator.melds import Meld
from simulator.tiles import SUITED_TILES, Suit, Tile, BONUS_TILES, TILES
from simulator.util import tiles_as_counts


class State:
    players: list[int] = list(range(4))

    wall: list[Tile]
    hands: list[list[Tile]]
    melds: list[list[Meld]] = [[]]*4

    curr_player: int = 0
    winner: int | None

    __discarded: Tile

    def __init__(self):
        # Create the wall as a permutation of the tiles
        self.wall = random.sample(TILES, len(TILES))

        self.hands = [ self.__draw(13) for _ in self.players ]
        self.hands[self.curr_player].append(self.__draw())

    def meld(self, meld: Meld, player: int):
        # Each agent will be provided each possible meld they could make
        # And asked which they prefer (if any)
        # Then this function will be called by the simulator, with the relevant priority

        # Remove tiles from hand
        for tile in meld.tiles:
            self.hands[player].remove(tile)

        # Record the meld
        self.melds[player].append(meld)
    
    def play(self, discard: int):
        # Discard
        self.__discarded = self.hands[self.curr_player][discard]
        del self.hands[self.curr_player][discard]

        # Update current player
        self.curr_player = (self.curr_player + 1) % 4

        # Draw tile for current player
        tile = self.__draw()

        # If we ran out of tiles, we have a draw
        if not tile:
            self.winner = -1 
            return
        
        # Add the tile to their hand
        self.hands[self.curr_player].append(tile)

    @property
    def available_melds(self, player: int):
        pass

    def would_win(self, player: int, tile: Tile) -> bool:
        # Get the count of each tile for efficient use
        counts = tiles_as_counts(self.hands[player])

        # The essence of this algorithm is that any tile that we have only one 
        # of must be a member of a chi, or else we have not won. Thus we can
        # first remove the chis that must exist, then we can deal with the rest
        suit_index = 0
        tile_index = 0
        while suit_index < 3:
            # If we don't have exactly 1, skip
            if counts[SUITED_TILES[suit_index][tile_index]] != 1:
                if tile_index != 8:
                    tile_index += 1
                else:
                    tile_index = 0
                    suit_index += 1
                continue

            # If we have only one, check if we have a valid run for this tile
            # There are three cases; two tiles in front, one tile in front and one behind, and two behind
            # For the behind cases, we decrease the index because this might have changed the previous cases
            if counts[SUITED_TILES[suit_index][tile_index+1]] > 0:
                if tile_index < 7 and counts[SUITED_TILES[suit_index][tile_index+2]] > 0:
                    counts[SUITED_TILES[suit_index][tile_index]] -= 1
                    counts[SUITED_TILES[suit_index][tile_index+1]] -= 1
                    counts[SUITED_TILES[suit_index][tile_index+2]] -= 1
                    tile_index += 1
                elif tile_index >= 1 and counts[SUITED_TILES[suit_index][tile_index-1]] > 0:
                    counts[SUITED_TILES[suit_index][tile_index]] -= 1
                    counts[SUITED_TILES[suit_index][tile_index+1]] -= 1
                    counts[SUITED_TILES[suit_index][tile_index-1]] -= 1
                    tile_index -= 1
                else:
                    return False
            elif tile_index >= 2 and counts[SUITED_TILES[suit_index][tile_index-1]] > 0 and counts[SUITED_TILES[suit_index][tile_index-2]] > 0:
                counts[SUITED_TILES[suit_index][tile_index]] -= 1
                counts[SUITED_TILES[suit_index][tile_index-1]] -= 1
                counts[SUITED_TILES[suit_index][tile_index-2]] -= 1
                tile_index -= 2
            else:
                return False
            
        # Once we've removed the

        return False

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