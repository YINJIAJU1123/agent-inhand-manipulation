"""Ground arbitrary English text without an environment, goal ID, or parser."""
import argparse
import json
import torch
from transformers import AutoModel, AutoProcessor
from text_grounding import encode_texts, ground_features

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--text',action='append',required=True)
p.add_argument('--device',default='cpu')
a=p.parse_args()
torch.set_num_threads(4)
adapter=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
processor=AutoProcessor.from_pretrained(a.model,use_fast=False)
model=AutoModel.from_pretrained(a.model).to(a.device).eval()
features=encode_texts(model,processor,a.text,a.device,adapter['max_length'])
_,predictions,confidences=ground_features(features,adapter)
print(json.dumps([dict(text=t,predicted_face=int(i),predicted_color=adapter['face_names'][int(i)],
                       softmax_confidence=float(c)) for t,i,c in zip(a.text,predictions,confidences)],indent=2))
