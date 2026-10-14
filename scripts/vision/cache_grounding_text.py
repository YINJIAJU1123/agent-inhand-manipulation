"""Cache frozen encoder outputs keyed by complete text, with no goal labels."""
import argparse
import hashlib
from pathlib import Path
import torch
from transformers import AutoModel, AutoProcessor
from text_grounding import TRAIN_TEMPLATES, HELDOUT_TEMPLATES, FACE_NAMES, encode_texts
p=argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
torch.set_num_threads(4)
processor=AutoProcessor.from_pretrained(a.model,use_fast=False)
model=AutoModel.from_pretrained(a.model).eval()
texts=list(dict.fromkeys(t.format(color=c) for t in TRAIN_TEMPLATES+HELDOUT_TEMPLATES for c in FACE_NAMES))
features=encode_texts(model,processor,texts,'cpu')
weights=Path(a.model)/'model.safetensors'
h=hashlib.sha256()
with weights.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
payload=dict(version='siglip2_full_text_cache_v1',max_length=64,model=a.model,
             model_sha256=h.hexdigest(),embeddings=dict(zip(texts,features.cpu())))
torch.save(payload,a.output)
print('cached',len(texts),'complete texts; model_sha256',h.hexdigest(),flush=True)
