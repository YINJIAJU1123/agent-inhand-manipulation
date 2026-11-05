"""Cache frozen SigLIP2 vectors for six fixed physical motor goals, using local files only."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer
import transformers
from goals import PROMPTS
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
model=AutoModel.from_pretrained(a.model,local_files_only=True).eval()
with torch.inference_mode():
    inputs=tokenizer(list(PROMPTS),padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    features=model.get_text_features(**inputs)
    if not isinstance(features,torch.Tensor):features=features.pooler_output
    vectors=torch.nn.functional.normalize(features.float(),dim=-1)
if vectors.shape!=(6,768):raise ValueError(vectors.shape)
a.output.parent.mkdir(parents=True,exist_ok=True)
meta=dict(model=str(a.model.resolve()),model_sha256=hashlib.sha256((a.model/'model.safetensors').read_bytes()).hexdigest(),prompts=list(PROMPTS),vectors=vectors,transformers_version=transformers.__version__,
    tokenization=dict(use_fast=True,max_length=64,padding='max_length',truncation=True),l2_normalized=True,
    tokenizer_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.glob('tokenizer*') if p.is_file() and p.suffix!='.partial'})
torch.save(meta,a.output)
a.output.with_suffix('.json').write_text(json.dumps({k:v for k,v in meta.items() if k!='vectors'}|{'cosine_similarity':(vectors@vectors.T).tolist()},indent=2)+'\n')
print(json.dumps({'output':str(a.output),'shape':list(vectors.shape)}))
