"""Optional streaming recorder. Reuses existing RGB; never renders or steps."""
from pathlib import Path
import numpy as np

class RolloutVideo:
    def __init__(self,path,width=512,height=512,fps=15):
        import imageio_ffmpeg
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.frames=0;self.fps=fps;self.closed=False
        self.writer=imageio_ffmpeg.write_frames(str(self.path),(width,height),fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','21','-movflags','+faststart'],ffmpeg_log_level='error')
        self.writer.send(None)
    def append(self,rgb):
        if self.closed: raise RuntimeError('recorder already closed')
        self.writer.send(np.ascontiguousarray(rgb,dtype=np.uint8));self.frames+=1
    def close(self):
        if not self.closed:self.writer.close();self.closed=True
