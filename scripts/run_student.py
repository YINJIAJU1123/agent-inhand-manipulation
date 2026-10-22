"""Frozen RGB-D + SigLIP2 text student, with the published exact-text cache."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    command=[os.environ.get('ISAACLAB_PYTHON',sys.executable),str(ROOT/'scripts/vision/evaluate_feature_student.py'),
        '--checkpoint',str(ROOT/'checkpoints/student/viserdex_rgbd_student.pt'),
        '--prompt-adapter',str(ROOT/'checkpoints/student/siglip2_text_adapter.pt'),
        '--text-feature-cache',str(ROOT/'checkpoints/student/siglip2_text_cache.pt'),
        '--semantic-camera','--semantic-grid-size','16','--semantic-depth-grid-size','16',
        '--language-mode','vlm','--vision-stride','2','--headless','--device','cuda:0']+sys.argv[1:]
    raise SystemExit(subprocess.call(command,cwd=ROOT))
