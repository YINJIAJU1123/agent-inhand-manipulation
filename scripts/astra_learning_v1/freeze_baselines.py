"""Verify and snapshot the historical live benchmark, excluding credentials."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/async_v7/local_20261016'
DEST=ROOT/'outputs/astra_learning_v1/baseline_freeze_20261019'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((SOURCE/'protocol.json').read_text());summary=json.loads((SOURCE/'summary.json').read_text())
    expected={'astra/continue':60,'astra/slow':68,'astra/learned':61,'latent/continue':59,'latent/slow':47,'latent/learned':43}
    files=[SOURCE/n for n in ['protocol.json','summary.json','final_diagnostics.json']]
    counts={k:[] for k in expected}
    for p in sorted(SOURCE.glob('eval_*/report.json')):
        r=json.loads(p.read_text());m=r['metadata'];key=m['mode']+'/'+m['waiting']
        assert m['seed'] in protocol['seeds'] and m['template_set'] in protocol['templates']
        assert len(r['episodes'])==6
        counts[key].extend(r['episodes']);files.extend([p,p.parent/'initial.json',p.parent/'manifest.json'])
    for k,rows in counts.items():
        assert len(rows)==120 and sum(x['success'] for x in rows)==expected[k],k
        assert len({(r['seed'],r['template_set'],r['case']) for r in rows})==120
    for name,h in protocol['source_sha256'].items():
        p=SOURCE/'sources'/Path(name).name
        assert sha(p)==h,str(p)
        files.append(p)
    for name,h in protocol['weight_sha256'].items():
        assert sha(ROOT/name)==h,name
        files.append(ROOT/name)
    DEST.mkdir(parents=True,exist_ok=False)
    for p in files:
        q=DEST/'artifacts'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    result=dict(created=time.time(),version='historical_live_baseline_v1',source=str(SOURCE),episodes=720,
        arms={k:dict(success=expected[k],episodes=120,rate=expected[k]/120) for k in expected},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        interpretation='Historical live Astra/VLM benchmark. Correct delivered semantic binding and orientation <=0.16 rad, no drop. Immediate evaluator termination, not a six-frame hold. Old learned means v1, not V3 or the new representation.',
        caveats=['Astra and VLM have different inference latency distributions.','There was one discarded late Astra inference error.','Known-goal controller diagnostics are a separate benchmark.','New learning pilot uses fresh matched baselines, not a direct comparison of different seeds.'])
    (DEST/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(frozen=str(DEST),files=len(files),arms=result['arms'])))
if __name__=='__main__':main()
