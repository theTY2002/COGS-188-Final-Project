from dataclasses import dataclass
from enum import Enum, auto

from simulator.tile import Tile

class MeldType(Enum):
    Pong = auto()
    Chow = auto()
    Kong = auto()
    Flower = auto()

@dataclass(frozen=True)
class Meld:
    type: MeldType
    tiles: list[Tile]