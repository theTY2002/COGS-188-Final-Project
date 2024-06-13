from collections import defaultdict

import torch
from simulator.tiles import DragonTile, SuitedTile, Tile, WindTile
from simulator.tiles import *

def tiles_as_counts(tiles: list[Tile]) -> dict[Tile, int]:
    counts = defaultdict(int)
    for tile in tiles:
        counts[tile] += 1

    return counts

def tile_to_index(tile: Tile) -> int:
    match tile:
        case SuitedTile(suit, rank):
            return (suit.value - 1) * 9 + rank - 1
        case WindTile(wind):
            return 26 + (wind.value)
        case DragonTile() as dragon:
            return 30 + (dragon.value)
        
def tiles_to_tensor(tiles: list[Tile]) -> torch.Tensor:
    tensor = torch.zeros(34, 4)
    if (len(tiles) == 0):
        return tensor
    counts = tiles_as_counts(tiles)
    for tile in counts:
        index = tile_to_index(tile)
        tensor[index, :counts[tile]] = 1
    return tensor

def index_to_tile(index: int):
    if 0 <= index <= 26:
        rank = index % 9 + 1
        suit = Suit(int((index - (index % 9)) / 9) + 1)
        return SuitedTile(suit, rank)
    elif 27 <= index <= 30:
        return WindTile(Wind(index - 26))
    else:
        return DragonTile(index - 30)