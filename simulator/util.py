from collections import defaultdict
from simulator.tiles import Tile

def tiles_as_counts(tiles: list[Tile]) -> dict[Tile, int]:
    counts = defaultdict(int)
    for tile in tiles:
        counts[tile] += 1

    return counts