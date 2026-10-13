"""Build a frozen SigLIP2 prompt canonicalizer.

The controller is deliberately untouched: all equivalent language templates are
encoded once and stored with their canonical target embedding.  At evaluation
time a nearest-template lookup maps a paraphrase to the canonical embedding
used by the stable RGB-D student.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoProcessor

FACE_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]
TEMPLATES = [
    "show the {color} marker",
    "reorient to the {color} marker",
    "turn toward the {color} marker",
    "bring the {color} marker to the top",
    "move the {color} marker into view",
    "place the {color} marker upward",
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    device=torch.device(args.device)
    processor=AutoProcessor.from_pretrained(args.model,use_fast=False)
    model=AutoModel.from_pretrained(args.model).to(device).eval()
    out=[]
    with torch.inference_mode():
        for template in TEMPLATES:
            texts=[template.format(color=color) for color in FACE_NAMES]
            inp=processor(text=texts,return_tensors="pt",padding=True,truncation=True)
            inp={k:v.to(device) for k,v in inp.items() if torch.is_tensor(v)}
            if hasattr(model,"get_text_features"): feat=model.get_text_features(**inp)
            else:
                enc=model.text_model(input_ids=inp["input_ids"],attention_mask=inp.get("attention_mask"))
                feat=enc.pooler_output if hasattr(enc,"pooler_output") else enc.last_hidden_state[:,0]
                if hasattr(model,"text_projection"): feat=model.text_projection(feat)
            out.append(torch.nn.functional.normalize(feat.float(),dim=-1).cpu())
    table=torch.stack(out, dim=0)
    canonical=table[0].clone()
    payload={"version":"siglip2_prompt_canonicalizer_v1","language_model":args.model,
             "templates":TEMPLATES,"face_names":FACE_NAMES,
             "prompt_embeddings":table,"canonical_embeddings":canonical,
             "language_model_sha256":hashlib.sha256(json.dumps({"model":args.model},sort_keys=True).encode()).hexdigest()}
    out_path=Path(args.output); out_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,out_path)
    report={"output":str(out_path),"version":payload["version"],"language_model":args.model,
            "templates":TEMPLATES,"face_names":FACE_NAMES,"embedding_dim":int(table.shape[-1]),
            "prompt_count":int(table.shape[0]*table.shape[1])}
    out_path.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()

