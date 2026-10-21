"""Frozen-skill rollouts for sensor-only completion and observation prediction.

Simulation geometry is used only to produce offline labels. These are random
skill rollouts, not a claim of same-state counterfactual action supervision.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"source/BrainCo_DexHand"),str(ROOT/"scripts/agentic")]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--output",required=True)
p.add_argument("--seed",type=int,required=True)
p.add_argument("--split",choices=["train","val"],required=True)
p.add_argument("--max-steps",type=int,default=900)
p.add_argument("--stride",type=int,default=10)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(dict(vars(a),multi_gpu=False)).app

import torch
import numpy as np
from common import *
from latent_data_v4 import split_layouts, visibility_labels
from latency_controller_v1 import normalized_target
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch


def all_errors(quaternions):
    s = 2.**-.5
    target = torch.tensor([[s,0,-s,0],[s,0,s,0],[s,s,0,0],
                           [s,-s,0,0],[1,0,0,0],[0,1,0,0]],
                          device=quaternions.device)
    dots = (quaternions[:,None]*target[None]).sum(-1).abs().clamp(0,1)
    return 2*torch.acos(dots)


def readable_labels(raw, symbols, camera, rgb, layout):
    """Match the existing corrected symbol-visibility labeling protocol."""
    labels = visibility_labels(raw,camera)
    symbols.markers.set_visibility(False)
    without = raw.capture_camera(refresh=True)["rgb"][...,:3].to(torch.uint8)
    symbols.markers.set_visibility(True)
    raw.capture_camera(refresh=True)
    difference = (rgb.float()-without.float()).abs().mean(-1)>20
    achromatic = (rgb.max(-1).values-rgb.min(-1).values)<45
    ink = (rgb.max(-1).values<80)|(rgb.min(-1).values>190)
    changed = difference & achromatic & ink
    labels["usable"] = [[False]*6 for _ in layout]
    fractions = [.315,.78,.3825,.4375,.4,.3825]
    for case in range(len(layout)):
        for color, points in enumerate(labels["projected_points"][case]):
            xx,yy=zip(*points)
            x0,x1=max(0,int(min(xx))-2),min(512,int(max(xx))+3)
            y0,y1=max(0,int(min(yy))-2),min(512,int(max(yy))+3)
            symbol=layout[case][color]
            pixels=int(changed[case,y0:y1,x0:x1].sum()) if x1>x0 and y1>y0 else 0
            expected=labels["projected_area"][case][color]*fractions[symbol]*(.0075/.009)**2
            labels["readable"][case][color]=bool(labels["facing"][case][color]>=.5 and pixels>=max(35,.65*expected))
            labels["usable"][case][color]=bool(labels["readable"][case][color] or pixels<=3 or labels["facing"][case][color]<=0)
    return labels


out = Path(a.output)
out.mkdir(parents=True,exist_ok=False)
env = None
try:
    student,text,pack = load_student(a.device)
    env,raw,symbols = environment(a.seed,a.max_steps,a.device)
    layout = split_layouts(a.seed,a.split)
    symbols.set_layout(layout)
    rng = np.random.default_rng(a.seed+97)
    schedule = rng.integers(0,6,size=(a.max_steps//90+1,6))
    # Half of slots persist on a skill for 270 steps, exposing convergence,
    # drift after reach and successful/failed sustained attempts.
    for j in range(len(schedule)):
        schedule[j,:3] = schedule[j-j%3,:3]
    active = torch.ones(6,dtype=torch.bool,device=a.device)
    ih = ph = None
    rows, arrays = [], []
    prefix = []
    start = time.time()
    with torch.inference_mode():
        for step in range(a.max_steps):
            if step%2 == 0:
                camera,rgb,feat,compact,prop = sensor_frame(raw,symbols)
                ih = history_append(ih,feat,pack["history"])
                ph = history_append(ph,prop,pack["history"])
            ids = torch.as_tensor(schedule[step//90],device=a.device)
            if step%a.stride == 0:
                patches,geometry = color_patches(rgb)
                labels = readable_labels(raw,symbols,camera,rgb,layout)
                errors = all_errors(raw.object_rot)
                for i in range(6):
                    if not active[i]:
                        continue
                    arrays.append(dict(patches=(patches[i]*255).byte().cpu(),
                                       geometry=geometry[i].cpu(),features=ih[i].cpu(),
                                       proprio=ph[i].cpu(),error=errors[i].cpu(),
                                       readable=torch.tensor(labels["readable"][i]),
                                       usable=torch.tensor(labels["usable"][i]),
                                       layout=torch.tensor(layout[i])))
                    rows.append(dict(seed=a.seed,case=i,step=step,skill=int(ids[i]),
                                     segment=step//90,split=a.split))
            proposal = student(VisualStudentBatch(ih,text[ids],ph))["action"]
            actions = torch.where(active[:,None],proposal,normalized_target(raw))
            prefix.append(actions.cpu())
            _,_,term,trunc,info = env.step(actions)
            active &= ~(term|trunc|info["semantic_metrics"]["dropped"].bool())
            if step%100 == 0:
                save(out/"progress.json",dict(step=step,frames=len(rows),active=int(active.sum()),
                                              wall_s=time.time()-start))
                print(f"step={step} frames={len(rows)} active={int(active.sum())}",flush=True)
            if not active.any():
                break
    if not arrays:
        raise RuntimeError("No training frames")
    data = {k:torch.stack([x[k] for x in arrays]) for k in arrays[0]}
    data["rows"] = rows
    # Future view quality is supervised only over a full 90-step segment.
    by_case = {i:[j for j,r in enumerate(rows) if r["case"]==i] for i in range(6)}
    future, valid, survived = [], [], []
    for j,row in enumerate(rows):
        candidates = [k for k in by_case[row["case"]]
                      if row["step"] <= rows[k]["step"] <= row["step"]+80 and
                      rows[k]["segment"] == row["segment"]]
        whole = row["step"]%90 == 0
        end_observed = any(rows[k]["step"]==row["step"]+80 for k in candidates)
        future.append(data["readable"][candidates].float().amax(0))
        valid.append(whole)
        # Missing suffix is failure only when episode actually terminated.
        survived.append(end_observed)
    data.update(future_readable=torch.stack(future),
                future_valid=torch.tensor(valid),
                survived=torch.tensor(survived))
    torch.save(data,out/"data.pt")
    torch.save(torch.stack(prefix),out/"actions.pt")
    save(out/"manifest.json",dict(status="completed",seed=a.seed,split=a.split,
        frames=len(rows),steps=step+1,student_sha256=sha(STUDENT),adapter_sha256=sha(ADAPTER),
        source_sha256=sha(__file__),data_sha256=sha(out/"data.pt"),
        labels="GT per-color full orientation error and same-state symbol-hidden render difference, actual ink-radius area correction and ambiguous-label masking; labels only",
        supervision_limit="Random skill transitions; action-dependent observation prediction, not counterfactual Q or final-success value",
        max_steps=a.max_steps,stride=a.stride))
    print(f"completed frames={len(rows)}",flush=True)
except BaseException:
    import traceback
    save(out/"failure.json",dict(error=traceback.format_exc()))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
