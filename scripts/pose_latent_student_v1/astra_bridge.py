"""Run the existing identity-pinned Astra bridge with the RSLCube contract.

Credentials stay in the separately consented local profile. The simulator sends
the same 120x120 RGB used by the fine-tuned VLM, with the same image-only crop.
"""
from functools import partial
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
from semantic import SYSTEM, tools_schema, validate_action

sys.path[:0] = [str(ROOT/'scripts/astra_transport_v2'), str(ROOT/'scripts/agentic')]
import bridge as transport
from subscription_v2 import AstraSubscriptionClient

if __name__ == '__main__':
    transport.AstraSubscriptionClient = partial(AstraSubscriptionClient,
        system_prompt=SYSTEM, tool_schema=tools_schema, action_validator=validate_action)
    if '--frame-size' not in sys.argv:
        sys.argv += ['--frame-size', '120']
    # Reuse the exact consent/identity pinning and no-fallback rules.
    runpy.run_path(str(ROOT/'scripts/semantic_completion_v1/bridge.py'), run_name='__main__')
