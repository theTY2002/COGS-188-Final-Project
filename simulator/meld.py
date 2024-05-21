from dataclasses import dataclass
from enum import Enum, auto

from simulator.tile import Tile

class MeldType(Enum):
    PON = auto()
    CHI = auto()
    KAN = auto()
    FLOWER = auto()

@dataclass(frozen=True)
class Meld:
    type: MeldType
    tiles: list[Tile]