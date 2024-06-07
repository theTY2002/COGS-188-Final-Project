from dataclasses import dataclass
from enum import Enum, auto
import itertools
from typing import Literal


class Suit(Enum):
    DOTS = auto()
    BAMBOO = auto()
    NUMBERS = auto()

class Wind(Enum):
    EAST = auto()
    SOUTH = auto()
    WEST = auto()
    NORTH = auto()

class Tile:
    pass

@dataclass(frozen=True)
class SuitedTile(Tile):
    suit: Suit
    rank: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]

@dataclass(frozen=True)
class WindTile(Tile):
    wind: Wind

class DragonTile(Tile, Enum):
    RED = auto()
    GREEN = auto()
    WHITE = auto()

@dataclass(frozen=True)
class BonusTile(Tile):
    wind: Wind

SUITED_TILES: list[list[Tile]] = [ [ SuitedTile(suit, i) for i in range(1, 10) ] for suit in Suit ]
OTHER_TILES: list[Tile] = [ WindTile(wind) for wind in Wind ] + list(DragonTile)
STANDARD_TILES: list[Tile] = list(itertools.chain(*SUITED_TILES)) + OTHER_TILES
BONUS_TILES: list[Tile] = [ BonusTile(wind) for wind in Wind ]

TILES: list[Tile] = STANDARD_TILES * 4 + BONUS_TILES * 2
