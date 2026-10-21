"""Reuse frozen assets without importing executable v1 entry points."""
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('completion_v1_assets',ROOT/'scripts/semantic_completion_v1/common.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
STUDENT,ADAPTER,CONTROLLER,PYTHON=v1.STUDENT,v1.ADAPTER,v1.CONTROLLER,v1.PYTHON
COLORS,SYMBOLS=v1.COLORS,v1.SYMBOLS
save,sha=v1.save,v1.sha
