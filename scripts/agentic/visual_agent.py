"""Bounded evidence-driven supervisor. Public inputs are RGB, instruction, time only."""
from __future__ import annotations
import base64,hashlib,io,json,time,urllib.request
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps

COLORS=('red','green','blue','yellow','magenta','cyan')
PROTOTYPES=np.asarray([[.85,.05,.05],[.05,.75,.15],[.05,.25,.9],[.95,.75,.05],[.8,.05,.75],[.05,.8,.85]])

def image_evidence(rgb):
    x=np.asarray(rgb,dtype=np.float32)/255
    saturation=x.max(-1)-x.min(-1)
    mask=(saturation>.3)&(x.max(-1)>.3)
    yy,xx=np.where(mask)
    if len(xx)>8:
        box=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(x.shape[1],int(xx.max())+9),min(x.shape[0],int(yy.max())+9))
    else: box=(0,0,x.shape[1],x.shape[0])
    distance=((x[:,:,None,:]-PROTOTYPES)**2).mean(-1)
    mass=(np.exp(-distance/.025)*mask[:,:,None]).mean((0,1))
    return box,mass

def encode_png(image):
    b=io.BytesIO(); image.save(b,format='PNG'); return base64.b64encode(b.getvalue()).decode()

class VLMClient:
    def __init__(self,url='http://127.0.0.1:18741',grounding='inventory'):
        self.url=url; self.grounding=grounding
    def ask(self,rgb,prompt,max_tokens=256):
        frame=Image.fromarray(np.asarray(rgb,dtype=np.uint8)); box,_=image_evidence(rgb)
        crop=ImageOps.pad(frame.crop(box),(336,336),color=(32,32,32))
        request={'images':[encode_png(frame),encode_png(crop)],'prompt':prompt,'max_tokens':max_tokens}
        start=time.monotonic()
        req=urllib.request.Request(self.url,data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as f: result=json.load(f)
        raw=result['text']; parsed=None
        decoder=json.JSONDecoder()
        for i,c in enumerate(raw):
            if c in '{[':
                try: parsed=decoder.raw_decode(raw[i:])[0]; break
                except json.JSONDecodeError: pass
        if isinstance(parsed,list): parsed={'stickers':parsed}
        if not isinstance(parsed,dict): parsed={}
        color=parsed.get('color','unknown')
        if color not in COLORS: color='unknown'
        visible=parsed.get('visible') is True
        return {'color':color,'visible':visible,'raw':raw,'elapsed_s':time.monotonic()-start,
                'image_sha256':hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),'crop_box':box,
                'parsed':parsed}
    def ground(self,rgb,instruction):
        # This first benchmark has a single semantic predicate. Reject unsupported
        # tasks instead of silently pretending to understand arbitrary language.
        if 'star' not in instruction.lower():
            raise ValueError('v1 symbol benchmark supports star instructions only')
        if self.grounding == 'inventory':
            result=self.ask(rgb, """Describe only the small square colored stickers on this robot-held cube. The second image is an enlarged crop of the first.
Ignore the LARGE colored cube face backgrounds and printed letters. Each SMALL sticker may carry a black geometric mark with a white outline. Inspect the mark's actual outline, not what you expect.
List visible small stickers, specifying sticker color and symbol. Symbol choices: star (five pointed tips), circle (round), triangle (three corners), cross (plus sign), diamond (four corners), bars (two separate parallel rectangles), unknown.
Use unknown for marks you cannot see distinctly. Do not infer hidden stickers. No sticker is guaranteed to show a star.
JSON only: {"stickers":[{"color":"red|green|blue|yellow|magenta|cyan|unknown","symbol":"star|circle|triangle|cross|diamond|bars|unknown"}]}""")
            stickers=result['parsed'].get('stickers',[])
            if not isinstance(stickers,list): stickers=[]
            candidates={s.get('color') for s in stickers if isinstance(s,dict) and s.get('symbol')=='star' and s.get('color') in COLORS}
            result['color']=next(iter(candidates)) if len(candidates)==1 else 'unknown'
            result['visible']=len(candidates)==1
            result['grounding']='inventory'
            return result
        return self.ask(rgb, f'''Task: {instruction}
The images show the same view: a robot-held cube and an enlarged crop.
Each face has a SMALL COLORED SQUARE STICKER with a black and white geometric symbol.
Find the FIVE-POINTED STAR symbol. Report the color of the SMALL sticker immediately underneath it, not the larger cube face background. Ignore letters on the cube.
Allowed sticker colors: red, green, blue, yellow, magenta, cyan.
If the star is hidden, occluded, or not recognizable, say unknown. Do not guess using a fixed layout: the symbol assignment changes every episode.
Reply only JSON: {{"color":"red|green|blue|yellow|magenta|cyan|unknown","visible":true|false,"evidence":"brief visible detail"}}''')
    def verify(self,rgb,color):
        return self.ask(rgb,f'''The two images are a scene and its enlarged crop. Inspect the SMALL {color} colored sticker on the cube, ignoring the larger face background and letters.
Is that small sticker on the cube's upward-facing TOP surface, rather than a side? If hidden or ambiguous say unknown. This is a coarse visual check, not a measurement of rotation error.
Reply only JSON: {{"color":"{color}","visible":true|false,"upward":"yes|no|unknown","evidence":"brief visible detail"}}''')

@dataclass
class Decision:
    color: int | None
    state: str
    reason: str

class VisualAgent:
    def __init__(self,mode,client,instruction='Turn the cube so the star sticker faces upward.',budget=6,enable_recovery=True):
        if mode not in ('once','periodic','event_memory'): raise ValueError(mode)
        self.mode=mode; self.client=client; self.instruction=instruction; self.budget=budget
        self.goal=None; self.last_call=-1000; self.last_mass=None; self.last_rgb=None
        self.pending=None; self.pending_votes=0; self.pending_frame=None
        self.inspected=[0]*6; self.inspect_color=None; self.inspect_since=-1000
        self.trace=[]; self.calls=0; self.stale=0; self.recover_until=-1; self.recovery_used=False
        self.enable_recovery=enable_recovery
        self.symbol_evidence={}; self.coarse_verified=False

    def tick(self,rgb,step):
        _,mass=image_evidence(rgb)
        if self.last_rgb is not None:
            delta=float(np.abs(np.asarray(rgb,dtype=float)-self.last_rgb).mean()/255)
            self.stale=self.stale+1 if delta<.004 else 0
        else: delta=1.0
        self.last_rgb=np.asarray(rgb,dtype=float)
        changed=self.last_mass is None or float(np.abs(mass-self.last_mass).sum())>.005
        reason=None
        if self.calls==0: reason='initial_grounding'
        elif self.mode=='periodic' and step-self.last_call>=90: reason='periodic'
        elif self.mode=='event_memory' and step-self.last_call>=60:
            if self.pending is not None: reason='confirm_hypothesis'
            elif self.goal is None and (changed or step-self.last_call>=120): reason='new_view_unresolved'
            elif self.stale>=4 and self.goal is not None: reason='visual_stall'
            elif self.goal is not None and self.last_mass is not None and mass[self.goal]<self.last_mass[self.goal]*.35: reason='visibility_loss'
        if reason and self.calls<self.budget:
            is_verify=reason=='visual_stall' and self.goal is not None
            result=self.client.verify(rgb,COLORS[self.goal]) if is_verify else self.client.ground(rgb,self.instruction)
            self.calls+=1; self.last_call=step; self.last_mass=mass.copy()
            previous=self.goal
            if is_verify:
                verdict=result['parsed'].get('upward','unknown')
                self.coarse_verified=verdict=='yes'
                # Verification is coarse and never declares exact task success.
                # A visible side-facing verdict permits one bounded recovery.
                if verdict=='no' and result['visible'] and not self.recovery_used and self.enable_recovery:
                    self.recovery_used=True; self.recover_until=step+20
                    alternatives=[i for i in range(6) if i!=self.goal]
                    self.inspect_color=max(alternatives,key=lambda i:mass[i])
            else:
                inventory=result.get('parsed',{}).get('stickers',[])
                if isinstance(inventory,list):
                    for sticker in inventory:
                        if not isinstance(sticker,dict): continue
                        color=sticker.get('color'); symbol=sticker.get('symbol')
                        if color in COLORS and symbol in ('star','circle','triangle','cross','diamond','bars'):
                            self.symbol_evidence.setdefault(color,{}).setdefault(symbol,set()).add(result['image_sha256'])
                candidate=COLORS.index(result['color']) if result['visible'] and result['color'] in COLORS else None
                if candidate is not None:
                    if self.mode!='event_memory' or candidate==self.goal:
                        self.goal=candidate; self.pending=None; self.pending_votes=0
                    else:
                        # Both initial binding and goal changes require support
                        # from two distinct observations. A first detection becomes
                        # an inspection hypothesis, not an accepted task goal.
                        if candidate!=self.pending:
                            self.pending=candidate; self.pending_votes=1; self.pending_frame=result['image_sha256']
                        elif self.pending_frame!=result['image_sha256']:
                            self.pending_votes+=1; self.pending_frame=result['image_sha256']
                        if self.pending_votes>=2:
                            self.goal=candidate; self.pending=None; self.pending_votes=0
                # Unknown never erases established visual evidence.
            self.trace.append(dict(step=step,reason=reason,operation='verify' if is_verify else 'ground',previous_goal=previous,accepted_goal=self.goal,pending=self.pending,**result))
        if self.goal is not None:
            if step<self.recover_until:
                return Decision(self.inspect_color,'recover_inspect','bounded_20_step_reorientation_then_retry')
            return Decision(self.goal,'execute','retain_visual_goal')
        if self.mode=='once': return Decision(None,'abstain','no_visual_evidence')
        if self.pending is not None:
            return Decision(self.pending,'inspect','turn_candidate_up_for_confirmation')
        if self.inspect_color is None or step-self.inspect_since>=90:
            # Prefer an unvisited but currently visible surface before a fully
            # hidden one; pixel mass is only an observation-based motion-cost proxy.
            def resolved_nonstar(i):
                votes=self.symbol_evidence.get(COLORS[i],{})
                return any(len(frames)>=2 for symbol,frames in votes.items() if symbol!='star') and not votes.get('star')
            candidates=[i for i in range(6) if not resolved_nonstar(i)] or list(range(6))
            self.inspect_color=min(candidates,key=lambda i:(self.inspected[i],-mass[i]))
            self.inspected[self.inspect_color]+=1; self.inspect_since=step
        return Decision(self.inspect_color,'inspect','unvisited_visible_surface_first')
