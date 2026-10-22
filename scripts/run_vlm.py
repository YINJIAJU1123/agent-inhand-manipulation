"""Qwen learned-latent baseline using the separate loopback VLM service.

This retained baseline uses evaluator termination on reaching the correct goal.
Its metric differs from run_astra.py's autonomous-stop metric.
"""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    command=[os.environ.get('ISAACLAB_PYTHON',sys.executable),str(ROOT/'scripts/agentic/evaluate_async_v7.py'),
        '--mode','latent','--waiting','continue','--seed','110101','--headless','--device','cuda:0']+sys.argv[1:]
    raise SystemExit(subprocess.call(command,cwd=ROOT))
