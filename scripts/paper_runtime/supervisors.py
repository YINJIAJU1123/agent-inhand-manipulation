"""Local baseline adapters. Only instruction and observed RGB/proprio enter agents.

The shared external pose completion head supplies their autonomous stop signal.
Neither historical local supervisor natively predicts a stop action.
"""
from async_agent_v7 import AsyncAgent
from semantic_agent_v3 import SemanticAgent, SemanticVLMClient
from latent_supervisor_v4 import LatentAgent
from latent_service_v7 import RemotePredictor
from agent import language_symbol
from common import SYMBOLS


def build_agents(kind, url, instructions, budget):
    predictor=RemotePredictor(url) if kind=='latent' else None
    agents=[]
    for instruction in instructions:
        if kind=='rule':
            agent=SemanticAgent('periodic',SemanticVLMClient(url=url),instruction,budget)
            agent.goal_embedding=None
            agent.declared_success=False
        else:
            agent=LatentAgent(predictor,instruction,budget)
            requested=language_symbol(instruction)
            agent.goal_symbol=SYMBOLS[requested] if requested is not None else None
        agents.append(agent)
    return agents


class LocalAsyncAgent(AsyncAgent):
    def submit(self,rgb,proprio,step):
        # The rule agent uses two actual model calls on its first tick.
        if self.agent.calls>=self.agent.budget:
            return False
        return super().submit(rgb,proprio,step)
