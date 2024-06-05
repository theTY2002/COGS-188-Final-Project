from abc import ABC, abstractmethod

from simulator.melds import Meld
from simulator.tiles import Tile


class Agent(ABC):
    @abstractmethod
    def choose_discard(self, hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int:
        """
        Chooses a tile to discard.

        Args:
            hand: This agent's hand.
            melds: The melds of each player.
            discards: The discards of each player.
            player: The index for this player in the list.

        Returns:
            An int, representing the index of the tile to discard.
        """
        pass
        
    @abstractmethod
    def choose_meld(self, available_melds: list[Meld], hand: list[Tile], melds: list[list[Meld]], discards: list[list[Tile]], player: int) -> int | None:
        """
        Chooses a meld to take, if any.

        Args:
            available_melds: The melds this player can choose to take.
            hand: This agent's hand.
            melds: The melds of each player.
            discards: The discards of each player.
            player: The index for this player in the list.

        Returns:
            An int, representing the index of the meld to take, or None if they would not like to meld.
        """
        pass