import ast,importlib.util,json,os,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff_launch',ROOT/'scripts/handoff/launch.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
class HandoffTests(unittest.TestCase):
 def test_all_python_parses(self):
  for p in (ROOT/'scripts').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 def test_six_object_configuration(self):
  cfg=json.loads((ROOT/'configs/objects.json').read_text());self.assertEqual(set(cfg),{'cube','die','duck','hedgehog','pig','milk_carton'})
  self.assertEqual(cfg['cube']['frame_size'],120)
  for key in set(cfg)-{'cube'}:self.assertEqual(cfg[key]['frame_size'],512)
 @unittest.skipUnless(os.environ.get('REVEAL_TEST_BUNDLE'),'Set REVEAL_TEST_BUNDLE to validate full delivery')
 def test_every_method_command_and_bundle(self):
  from argparse import Namespace
  bundle=Path(os.environ['REVEAL_TEST_BUNDLE']);config=json.loads((ROOT/'configs/objects.json').read_text())
  with tempfile.TemporaryDirectory() as td:
   for obj in config:
    for method in ['astra','vlm','rule']:
     a=Namespace(bundle=bundle,object=obj,mode='eval',method=method,gpu=4,port=19081,model=None,output=Path(td)/(obj+'_'+method),seed=990601,episodes=1,execute=False)
     argv,env=launch.command(a);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4');self.assertEqual(env['PYGLET_HEADLESS_DEVICE'],'4')
     self.assertEqual('--actor_checkpoint' in argv,method=='astra')
     self.assertEqual('--stop_checkpoint' in argv,method=='astra' and obj!='cube')
     self.assertEqual('--execution_actor_checkpoint' in argv,method=='astra' and obj=='cube')
     if obj=='cube':self.assertEqual(argv[argv.index('--stop_mode')+1],'measured')
     else:self.assertIn('--development',argv)
     for i,value in enumerate(argv):
      if value in ['--checkpoint','--source_checkpoint','--student_checkpoint','--actor_checkpoint','--execution_actor_checkpoint','--stop_checkpoint','--env_config','--relocation','--assets_manifest','--pose_checkpoint','--original_asset','--text_cache','--camera_calibration','--bank']:
       self.assertTrue(Path(argv[i+1]).exists(),argv[i+1])
     if '--assets_manifest' in argv:
      m=json.loads(Path(argv[argv.index('--assets_manifest')+1]).read_text());self.assertTrue(Path(m['source']).exists())
      for entry in m['conditions'].values():self.assertEqual(launch.sha(Path(entry['asset'])),entry['sha256'])
   # Resolving a venv's interpreter symlink loses its pyvenv.cfg and packages.
   fake=Path(td)/'venv/bin/python';fake.parent.mkdir(parents=True);fake.symlink_to(sys.executable)
   with patch.dict(os.environ,{'VISERDEX_PYTHON':str(fake)}):
    _,env=launch.command(a);self.assertEqual(env['VISERDEX_PYTHON'],str(fake.absolute()))
   a.output=Path(td);self.assertRaises(FileExistsError,launch.command,a)
if __name__=='__main__':unittest.main()
