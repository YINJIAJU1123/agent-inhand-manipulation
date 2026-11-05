"""Continuous diagnostic video of the native hand and actual policy RGB.

The native camera is a spectator. Its pixels never enter the student or agent.
Frame timestamps refer to simulation time, and every sampled frame is retained.
"""
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class EpisodeVideo:
    def __init__(self, directory, title, stride):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.title=title
        self.stride=stride
        self.started=time.monotonic()
        self.frames=0
        self.last_step=None
        self.writer=imageio.get_writer(str(self.directory/'continuous.mp4'),
            fps=30/stride,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=16)
        self.log=(self.directory/'video_frames.jsonl').open('w')
        font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.font=ImageFont.truetype(str(font),18) if font.exists() else ImageFont.load_default()
        self.large=ImageFont.truetype(str(font),24) if font.exists() else ImageFont.load_default()
        self.closed=False

    def append(self,step,native,policy_rgb,context):
        if step%self.stride or step==self.last_step:return
        canvas=Image.new('RGB',(1280,800),'#101820')
        draw=ImageDraw.Draw(canvas)
        draw.text((20,12),self.title,font=self.large,fill='white')
        draw.text((20,48),f"Simulation {step/30:.2f}s | wall {time.monotonic()-self.started:.2f}s | continuous physics",
                  font=self.font,fill='#b6c5d4')
        draw.text((20,84),'Native hand view - spectator only',font=self.font,fill='#6ae0d0')
        draw.text((830,84),'Actual policy RGB (120 x 120)',font=self.font,fill='#6ae0d0')
        canvas.paste(Image.fromarray(native).resize((800,640)),(16,112))
        canvas.paste(Image.fromarray(policy_rgb).resize((384,384),Image.Resampling.NEAREST),(832,112))
        rows=[
            'Mode: '+str(context['mode']),
            'Waiting for inference: '+str(context['pending']),
            'Goal bound: '+str(context['known']),
            'Active color: '+str(context['active_color']),
            'Bound color: '+str(context.get('bound_color')),
            'Calls submitted: '+str(context['calls']),
            f"Measured active error: {context['measured_error_rad']:.3f} rad",
            '109 proprio + 7 pose + 768 latent',
        ]
        for i,line in enumerate(rows):
            draw.text((832,518+i*28),line,font=self.font,fill='white')
        draw.text((20,766),'Video time = simulation time. Shared GPU throughput is not a real-time claim.',
                  font=self.font,fill='#b6c5d4')
        frame=np.asarray(canvas)
        self.writer.append_data(frame)
        self.log.write(json.dumps(dict(frame=self.frames,step=step,simulation_s=step/30,**context))+'\n')
        if self.frames==0:canvas.save(self.directory/'preview.png')
        self.frames+=1
        self.last_step=step

    def close(self,reason,terminal_step):
        if self.closed:return
        self.writer.close()
        self.log.close()
        self.closed=True
        (self.directory/'video_audit.json').write_text(json.dumps(dict(
            title=self.title,frames=self.frames,fps=30/self.stride,frame_stride=self.stride,
            first_frame_step=0 if self.frames else None,last_frame_step=self.last_step,
            terminal_step=terminal_step,reason=reason,
            scope='qualitative continuous recording; not a formal evaluation condition',
            sampled_frames_preserved=True,policy_input_unchanged=True,
            terminal_frame='pre-action samples only; no post-reset state is shown on drop or timeout',
            native_camera='spectator only; not supplied to student or upper model'),indent=2)+'\n')
