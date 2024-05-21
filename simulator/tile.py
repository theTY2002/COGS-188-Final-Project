from dataclasses import dataclass
from enum import Enum, auto
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
    value: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]

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

STANDARD_TILES: set[Tile] = { SuitedTile(suit, i) for suit in Suit for i in range(1, 10) } | { WindTile(wind) for wind in Wind } | set(DragonTile)
BONUS_TILES: set[Tile] = { BonusTile(wind) for wind in Wind }

TILES: list[Tile] = list(STANDARD_TILES) * 4 + list(BONUS_TILES) * 2
