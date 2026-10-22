"""Frozen RGB reader plus explicit historical evidence for an external agent."""
import hashlib
import importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'checkpoints/runtime/semantic_reader.pt'
CHECKPOINT_SHA256='ff61173f56b3ba0c26481e1b9037f8e1a1307241077488415cae74dbd82a1f87'
COLORS=('red','green','blue','yellow','magenta','cyan')
SYMBOLS=('star','circle','triangle','cross','diamond','bars')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class OnlineMemory:
    def __init__(self,device,cases=6,checkpoint=CHECKPOINT):
        checkpoint=Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
            raise RuntimeError('Frozen semantic reader checksum mismatch')
        pack=torch.load(checkpoint,map_location=device,weights_only=True)
        if pack.get('architecture')!='verified' or not pack.get('offline_gate_passed') or pack['threshold']!=.85:
            raise RuntimeError('Unexpected reader/calibration')
        module=load_module('online_frozen_verified_reader',ROOT/'scripts/astra_symbol_verified_v1/model.py')
        self.model=module.StickerReader().to(device).eval();self.model.load_state_dict(pack['model']);self.model.requires_grad_(False)
        memory_module=load_module('online_frozen_semantic_memory',ROOT/'scripts/astra_semantic_memory_v1/memory.py')
        self.memory=memory_module.SemanticMemory(cases,pack['threshold'],device)
        self.device=device;self.records={};self.last=None

    @torch.inference_mode()
    def update(self,rgb,step):
        if rgb.ndim!=4 or rgb.shape[-1]!=3 or rgb.dtype!=torch.uint8:
            raise ValueError('Expected measured NHWC uint8 RGB')
        probability=self.model(rgb.to(self.device).permute(0,3,1,2))['probability']
        snapshot=self.memory.update(probability,step)
        self.last={k:v.cpu() for k,v in snapshot.items()}
        self.records[step]=dict(probability=probability.cpu(),memory=self.last)

    def summary(self,case,step):
        if self.last is None:raise RuntimeError('No RGB evidence received')
        if step<self.memory.last_step:raise ValueError('Feedback precedes observation')
        observations={}
        for color,name in enumerate(COLORS):
            symbol=int(self.last['symbol'][case,color]);observed=int(self.last['observed_step'][case,color])
            observations[name]=dict(symbol=SYMBOLS[symbol] if symbol>=0 else 'unknown',
                confidence=float(self.last['confidence'][case,color]),
                observed_sim_seconds=observed/30 if observed>=0 else None,
                age_sim_seconds=(step-observed)/30 if observed>=0 else None,
                accepted_in_latest_frame=bool(self.last['fresh_accepted'][case,color]))
        known=[r['symbol'] for r in observations.values() if r['symbol']!='unknown']
        duplicates=sorted({name for name in known if known.count(name)>1})
        return dict(source='Frozen learned RGB sticker reader, threshold 0.85; latest accepted observation per color',
            colors=observations,ambiguous_symbols=duplicates,
            interpretation='Historical sticker identity on this unchanged object, not current visibility or orientation. Confidence is a model score, not verified truth. Unknown and duplicate bindings need inspection. Do not infer unobserved identities from a permutation. Use current visual pose evidence before stopping.')
