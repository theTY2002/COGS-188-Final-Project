import random
from simulator.agent import Agent
from simulator.melds import Meld
from simulator.state import State


class Game:
    state: State
    agents: list[Agent]

    def __init__(self, agents: list[Agent]):
        # Save a shuffled version of the players
        assert len(agents) == 4
        self.agents = random.sample(agents, 4)

        # Generate the starting state
        self.state = State()

    def step(self):
        # Get the relevant agent
        curr_agent = self.agents[self.state.curr_player]

        # Ask it to discard a tile (its action for the turn)
        self.state.discard(curr_agent.choose_discard(
            self.state.hands[self.state.curr_player],
            self.state.melds,
            self.state.curr_player
        ))

        # Ask each agent what meld it would like to make
        preferred_melds: list[tuple[int, Meld]] = []
        for i, agent in enumerate(self.agents):
            # The player who just discarded cannot meld
            if agent == curr_agent:
                continue
            preferred_melds.append((i, agent.choose_meld(
                self.state.available_melds(i),
                self.state.hands[i],
                self.state.melds,
                i
            )))

        self.state.draw()
        kans = self.state.concealed_kans()
        
        while len(kans) > 0:
            meld = agent.choose_meld(
                self.state.available_melds(i),
                self.state.hands[i],
                self.state.melds,
                i
            )

            # If it chose something, take it and then recalculate
            if meld:
                self.state.meld(self.state.curr_player, meld)
                kans = self.state.concealed_kans()
            else:
                break
