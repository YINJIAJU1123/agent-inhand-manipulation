"""Relocate copied dependencies in an isolated development environment only."""
import argparse,hashlib,json,os,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();project=a.project.resolve();old='/home/brainco/agentic_rotation_newton'
site=root/'.venv-newton/lib/python3.12/site-packages';changes=[]
for path in list(site.glob('__editable__*.py'))+list(site.glob('*.pth')):
 text=path.read_text()
 if old in text:path.write_text(text.replace(old,str(root)));changes.append(str(path.relative_to(root)))
for directory,dirs,files in os.walk(root,followlinks=False):
 for name in dirs+files:
  path=Path(directory)/name
  if path.is_symlink():
   target=os.readlink(path)
   if target.startswith(old+'/'):path.unlink();path.symlink_to(str(root)+target[len(old):]);changes.append(str(path.relative_to(root)))
for name in ('tmp','cache','bin','outputs','.venv-newton/lib/cuda-build'):(root/name).mkdir(parents=True,exist_ok=True)
links={root/'VLMmanipulation':project,
 root/'outputs/goal_study_release_20261029':project/'outputs/pose_latent_student_v1/goal_study_release_20261029',
 root/'.venv-newton/lib/cuda-build/libcudart.so':site/'nvidia/cu13/lib/libcudart.so.13'}
for path,target in links.items():
 if not path.exists() and not path.is_symlink():path.symlink_to(target)
 elif path.resolve()!=target.resolve():raise ValueError('Unexpected path already exists: '+str(path))
q=shlex.quote
settings={'ISAACLAB_PATH':str(root/'IsaacLab-e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4'),
 'VIRTUAL_ENV':str(root/'.venv-newton'),'VISERDEX_PYTHON':str(root/'.venv-newton/bin/python'),
 'UV_BIN':str(root/'bin/uv'),'UV_PROJECT_ENVIRONMENT':str(root/'.venv-newton'),
 'UV_PYTHON_PREFERENCE':'only-system','UV_PYTHON_DOWNLOADS':'never','PYTHONNOUSERSITE':'1',
 'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache'),
 'CUDA_HOME':str(site/'nvidia/cu13'),'MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'8.6',
 'CPATH':'/home/yin/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12'}
text='\n'.join('export '+k+'='+q(v) for k,v in settings.items())+'\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\nunset PYTHONPATH\n'
(root/'bin/activate_runtime.sh').write_text(text)
for name,asset in json.loads((root/'relocation.json').read_text())['assets'].items():
 path=Path(asset['path'].replace(old,str(root),1));assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256'],name
(root/'local_setup.json').write_text(json.dumps({'status':'paths_prepared_import_check_pending','scope':'local_development_only','path_changes':changes,'isaacsim_graphical_package_copied':False,'asset_hashes_verified':True},indent=2)+'\n')
print({'relocated_paths':len(changes),'asset_hashes_verified':True})
