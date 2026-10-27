"""One fixed-actor six-scene job from the paper's three-arm stop comparison."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    command=[os.environ.get('ISAACLAB_PYTHON',sys.executable),str(ROOT/'scripts/paper_stop/evaluate.py'),
        '--completion-policy','shared','--maintenance','original','--actor-choice','mixed','--actor-mode','candidate',
        '--asset-manifest',str(ROOT/'configs/runtime_assets.json'),
        '--stop-assets',str(ROOT/'configs/paper_stop_assets.json'),'--supervisor','astra',
        '--initialization','collision_clearance','--startup-mode','fixed_green','--memory-mode','memory',
        '--retention-scope','none','--arm','verify','--evidence-model',str(ROOT/'checkpoints/runtime/pose_verifier.pt'),
        '--template-set','canonical','--max-steps','3600','--budget','6','--headless','--device','cuda:0',
        *sys.argv[1:]]
    raise SystemExit(subprocess.call(command,cwd=ROOT))
