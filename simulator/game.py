import random
from simulator.agent import Agent
from simulator.melds import Meld, MeldType
from simulator.state import State


class Game:
    state: State
    agents: list[Agent]
    winner: int | None = None

    def __init__(self, agents: list[Agent]):
        # Save a copy of the players
        assert len(agents) == 4
        self.agents = agents.copy()

        # Generate the starting state
        self.state = State()

    def step(self) -> bool:
        if self.winner:
            return True

        # Ask the current agent to discard a tile (its action for the turn)
        self.state.discard(self.agents[self.state.curr_player].choose_discard(
            self.state.hands[self.state.curr_player],
            self.state.melds,
            self.state.discards,
            self.state.curr_player
        ))

        # Ask each agent what meld it would like to make, in play order
        preferred_melds: list[tuple[int, Meld]] = []
        enum_agents = list(enumerate(self.agents))
        for i, agent in enum_agents[self.state.curr_player+1:] + enum_agents[:self.state.curr_player]:
            avail = self.state.available_melds(i)

            if len(avail) > 0:
                m = agent.choose_meld(
                    avail,
                    self.state.hands[i],
                    self.state.melds,
                    self.state.discards,
                    i
                )
                if m != None:
                    preferred_melds.append((i, avail[m]))

        # If any melds were proposed, select the prioritized one
        if len(preferred_melds) > 0:
            best_player, best_meld = None, None

            for player, meld in preferred_melds:
                # If this meld would win, this is the earliest meld that would
                # do so, so it has priority, and nothing can trump it
                if self.state.would_win(player, meld.discard):
                    self.state.meld(player, meld)
                    self.winner = self.state.winner
                    return True

                # Otherwise, the meld has priority if this is the first meld
                # or if the current is a chi and the next is a pon or kan
                match (best_meld, meld):
                    case (None, _) | (Meld(MeldType.CHI), Meld(MeldType.PON | MeldType.KAN)):
                        best_player, best_meld = player, meld
            
            # Actually meld the chosen option
            self.state.meld(best_player, best_meld)
        else:
            # If no meld occurred, have the next player draw a tile
            self.state.draw()

        # Return done if we have winner
        if self.state.winner:
            self.winner = self.state.winner
            return True

        # Process concealed kans, as they can be declared right after
        # drawing/taking but before discarding
        kans = self.state.concealed_kans()
        while len(kans) > 0:
            meld = self.agents[self.state.curr_player].choose_meld(
                kans,
                self.state.hands[self.state.curr_player],
                self.state.melds,
                self.state.discards,
                self.state.curr_player
            )

            # If it chose something, take it and then recalculate
            if meld:
                self.state.meld(self.state.curr_player, meld)

                # Return done if we have a winner (have to recheck after meld)
                if self.state.winner:
                    self.winner = self.state.winner
                    return True

                kans = self.state.concealed_kans()
            else:
                break

        return False
