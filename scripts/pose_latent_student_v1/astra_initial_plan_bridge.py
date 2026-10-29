"""Same authorized Astra transport, restricted to the single-initial-plan schema."""
from functools import partial
from pathlib import Path
import runpy
import sys
from initial_plan import INITIAL_PLAN_SYSTEM,plan_tools_schema,validate_initial_plan

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/astra_transport_v2'),str(ROOT/'scripts/agentic')]
import bridge as transport
from subscription_v2 import AstraSubscriptionClient
if __name__=='__main__':
    transport.AstraSubscriptionClient=partial(AstraSubscriptionClient,
        system_prompt=INITIAL_PLAN_SYSTEM,tool_schema=plan_tools_schema,action_validator=validate_initial_plan)
    if '--frame-size' not in sys.argv:sys.argv+=['--frame-size','120']
    runpy.run_path(str(ROOT/'scripts/semantic_completion_v1/bridge.py'),run_name='__main__')
