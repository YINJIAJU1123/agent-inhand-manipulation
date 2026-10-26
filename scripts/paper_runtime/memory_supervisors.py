"""Fine-tuned latent policy with immutable per-request observed context."""
from latent_supervisor_v4 import LatentAgent
from latent_service_v7 import RemotePredictor
from async_agent_v7 import AsyncAgent
from agent import language_symbol
from common import SYMBOLS
from memory_context import ContextPredictor

class ContextLatentAgent(LatentAgent):
    def __init__(self,backend,instruction,budget):
        proxy=ContextPredictor(backend)
        super().__init__(proxy,instruction,budget)
        proxy.owner=self
        requested=language_symbol(instruction)
        self.goal_symbol=SYMBOLS[requested] if requested is not None else None

class ContextAsyncAgent(AsyncAgent):
    def __init__(self,agent,pool,mode,context_source):
        super().__init__(agent,pool,mode);self.context_source=context_source
    def submit(self,rgb,proprio,step):
        # Capture before the worker starts. Never mutate an in-flight request.
        if self.closed or self.future is not None or step<self.due or self.submitted>=self.agent.budget or self.agent.calls>=self.agent.budget:return False
        self.agent.predictor.capture(self.context_source.context)
        return super().submit(rgb,proprio,step)

def build_context_agents(url,instructions,budget):
    backend=RemotePredictor(url)
    return [ContextLatentAgent(backend,text,budget) for text in instructions]
