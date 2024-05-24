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
        self.state.play(curr_agent.choose_discard(
            self.state.hands[self.state.curr_player],
            self.state.melds,
            self.state.curr_player
        ))

        # Ask each agent what meld it would like to make
        preferred_melds: list[tuple[Agent, Meld]] = []
        for i, agent in enumerate(self.agents):
            if agent == curr_agent:
                continue
            preferred_melds.append(agent.choose_meld(
                self.state.available_melds,
                self.state.hands[i],
                self.state.melds,
                i
            ))
        

        