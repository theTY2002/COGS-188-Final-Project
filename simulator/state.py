import random

from simulator.melds import Meld, MeldType
from simulator.tiles import OTHER_TILES, SUITED_TILES, BonusTile, Suit, SuitedTile, Tile, BONUS_TILES, TILES
from simulator.util import tiles_as_counts


class State:
    players: list[int] = list(range(4))

    wall: list[Tile]
    hands: list[list[Tile]]
    melds: list[list[Meld]] = [ [] for _ in players ]
    discards: list[list[Tile]] = [ [] for _ in players ]

    curr_player: int = 0
    winner: int | None = None

    __discarded: Tile

    def __init__(self):
        # Create the wall as a permutation of the tiles
        self.wall = random.sample(TILES, len(TILES))

        self.hands = [ self.__draw(p, 13) for p in self.players ]
        self.hands[self.curr_player].append(self.__draw(self.curr_player))
    
    def discard(self, discard: int):
        # Discard
        self.__discarded = self.hands[self.curr_player][discard]
        self.discards[self.curr_player].append(self.__discarded)
        del self.hands[self.curr_player][discard]

        # Update current player
        self.curr_player = (self.curr_player + 1) % 4

    def meld(self, player: int, meld: Meld):
        # Each agent will be provided each possible meld they could make
        # And asked which they prefer (if any)
        # Then this function will be called by the simulator, with the relevant priority

        # If this meld makes this player win, do so
        if meld.discard and self.would_win(player, meld.discard):
            self.winner = player

        # Remove tiles from hand
        for tile in meld.tiles:
            self.hands[player].remove(tile)

        # Record the meld
        self.melds[player].append(meld)

        # Set the current player, since they need to draw now
        self.curr_player = player

        # If it was a kan, they're missing a tile, so draw
        if meld.type == MeldType.KAN:
            self.draw()

    def draw(self):
        # Draw tile for current player
        tile = self.__draw(self.curr_player)

        # If we ran out of tiles, we have a draw
        if not tile:
            self.winner = -1 
            return
        
        if self.would_win(self.curr_player, tile):
            self.winner = self.curr_player

        # Add the tile to their hand
        self.hands[self.curr_player].append(tile)

    def available_melds(self, player: int):
        counts = tiles_as_counts(self.hands[player])

        match self.__discarded:
            case SuitedTile(suit, rank):
                avail = []

                # Check for sets
                match counts[self.__discarded]:
                    case 2:
                        avail.append(Meld(MeldType.PON, [self.__discarded] * 2, self.__discarded))
                    case 3:
                        avail.append(Meld(MeldType.PON, [self.__discarded] * 2, self.__discarded))
                        avail.append(Meld(MeldType.KAN, [self.__discarded] * 3, self.__discarded))
                    
                # Check for chi
                candidates = [ SuitedTile(suit, rank + i) for i in range(-2, 3) ]
                if counts[candidates[3]] > 0:
                    if counts[candidates[4]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[3], candidates[4]], candidates[2]))
                    if counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[1], candidates[3]], candidates[2]))
                if counts[candidates[0]] > 0 and counts[candidates[1]] > 0:
                        avail.append(Meld(MeldType.CHI, [candidates[0], candidates[1]], candidates[2]))
                
                return avail
            case _:
                match counts[self.__discarded]:
                    case 2:
                        return [
                            Meld(MeldType.PON, [self.__discarded] * 2, self.__discarded)
                        ]
                    case 3:
                        return [
                            Meld(MeldType.PON, [self.__discarded] * 2, self.__discarded),
                            Meld(MeldType.KAN, [self.__discarded] * 3, self.__discarded)
                        ]
                    case _:
                        return []
                    
    def concealed_kans(self):
        avail = []
        counts = tiles_as_counts(self.hands[self.curr_player])
        for tile in counts:
            if counts[tile] == 4:
                avail.append(Meld(MeldType.KAN, [tile]*4, None))

        return avail

    def would_win(self, player: int, tile: Tile) -> bool:
        # Get the count of each tile for efficient use
        counts = tiles_as_counts(self.hands[player] + [tile])

        # This will keep track of our pair; we must have exactly one
        pair = False

        # Fail if we have an individual or a group of 4 of a single tile
        # Ignore if we have a 0 or a 3
        # If we have 2, that can be our one pair, but only one of those are allowed
        for t in OTHER_TILES:
            match counts[t]:
                case 1 | 4:
                    return False
                case 2:
                    if pair:
                        return False
                    else:
                        pair = True

        # The essence of this algorithm is that any tile that we have only one of
        # must be a member of a chi, or else we have not won. Thus we can first
        # remove the chis that must exist, then we can deal with the rest
        suit_index = 0
        tile_index = 0
        revisit: list[set[int]] = [ set() for _ in Suit ]
        while suit_index < 3:
            match counts[SUITED_TILES[suit_index][tile_index]]:
                case 1:
                    # If we have only one, check if we have a valid run for this tile
                    # There are three cases; two tiles in front, one tile in front and one behind, and two behind
                    # For the behind cases, we decrease the index because this might have changed the previous cases
                    if tile_index < 8 and counts[SUITED_TILES[suit_index][tile_index+1]] > 0:
                        if tile_index < 7 and counts[SUITED_TILES[suit_index][tile_index+2]] > 0:
                            counts[SUITED_TILES[suit_index][tile_index]] = 0
                            counts[SUITED_TILES[suit_index][tile_index+1]] -= 1
                            counts[SUITED_TILES[suit_index][tile_index+2]] -= 1
                            tile_index += 1
                        elif tile_index >= 1 and counts[SUITED_TILES[suit_index][tile_index-1]] > 0:
                            counts[SUITED_TILES[suit_index][tile_index]] = 0
                            counts[SUITED_TILES[suit_index][tile_index+1]] -= 1
                            counts[SUITED_TILES[suit_index][tile_index-1]] -= 1
                            tile_index -= 1
                        else:
                            return False
                    elif tile_index >= 2 and counts[SUITED_TILES[suit_index][tile_index-1]] > 0 and counts[SUITED_TILES[suit_index][tile_index-2]] > 0:
                        counts[SUITED_TILES[suit_index][tile_index]] = 0
                        counts[SUITED_TILES[suit_index][tile_index-1]] -= 1
                        counts[SUITED_TILES[suit_index][tile_index-2]] -= 1
                        tile_index -= 2
                    else:
                        return False
                    
                    # If this tile was previously a problem, it isn't anymore
                    # This can happen due to backtracking
                    revisit[suit_index].discard(tile_index)
                    continue
                case 2 | 4:
                    # If this is a problem, we'll need to look at it later
                    revisit[suit_index].add(tile_index)
                case _:
                    # Again, if this was a problem, it isn't anymore
                    revisit[suit_index].discard(tile_index)

            # Go to the next tile
            if tile_index != 8:
                tile_index += 1
            else:
                tile_index = 0
                suit_index += 1

        # Go through each tile we marked for revisiting
        for suit_index, s in enumerate(revisit):
            # Until we've dealt with each tile
            while len(s) > 0:
                # Get the current tile
                tile_index = s.pop()

                # Can't win with 4 in hand unless its part of a chi
                if counts[SUITED_TILES[suit_index][tile_index]] == 4:
                    return False

                # Check if it has 2 neighbors, if not, it can either be
                # the one pair or we would have to fair if we have already
                # No need for negative checks since it's either in the set or not
                if (tile_index + 1) in s:
                    if (tile_index + 2) in s:
                        s -= { tile_index + 1, tile_index + 2 }
                    elif (tile_index - 1) in s:
                        s -= { tile_index + 1, tile_index + 2 }
                    elif not pair:
                        pair = True
                    else:
                        return False
                elif (tile_index - 1) in s and (tile_index - 2) in s:
                        s -= { tile_index - 1, tile_index - 2 }
                elif not pair:
                    pair = True
                else: 
                    return False

        # If we made it through all that and didn't fail, this is a winning hand
        return True

    def __draw(self, player: int, n: int = None) -> Tile | list[Tile] | None:
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
                match drawn[i]:
                    case BonusTile(_):
                        self.melds[player].append(Meld(MeldType.FLOWER, [drawn[i]], None))

                        if len(self.wall) == 0:
                            return None

                        drawn[i] = self.wall[-1]
                        del self.wall[-1]

                        removed_bonus = True

        return drawn if n_orig else drawn[0]