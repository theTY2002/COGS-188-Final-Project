from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

class Suit(Enum):
    DOTS = auto()
    STICKS = auto()
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

standard_tiles: set[Tile] = { SuitedTile(suit, i) for suit in Suit for i in range(1, 10) } | { WindTile(wind) for wind in Wind } | set(DragonTile)
bonus_tiles: set[Tile] = { BonusTile(wind) for wind in Wind }

tiles: list[Tile] = list(standard_tiles) * 4 + list(bonus_tiles) * 2