"""RGB capture without a pose estimator, using the official Newton GL backend."""
import ctypes
import numpy as np

EYE=(-.146,.033,.878)
LOOKAT=(.02,-.01,.565)
SIZE=512

def configure_capture(cfg):
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    visual=NewtonGLVisualizerCfg(headless=True,window_width=SIZE,window_height=SIZE,
        eye=EYE,lookat=LOOKAT,streaming_view=False,enable_picking=False,background_color=(.08,.11,.16))
    cfg.sim.visualizer_cfgs=[visual];cfg.sim.default_visualizer_cfg=visual

class Capture:
    def __init__(self,env):
        self.env=env
        self.visualizer=next(v for v in env.sim.visualizers if v.cfg.visualizer_type=='newton_gl')
    def read(self,env_id=0,depth=False):
        origin=self.env.scene.env_origins[env_id].cpu().numpy()
        self.env.sim.set_camera_view(tuple(origin+EYE),tuple(origin+LOOKAT))
        self.env.sim.render()
        rgb=self.visualizer.render_rgb_array()[...,:3].copy()
        result=dict(rgb=rgb)
        if depth:
            viewer=self.visualizer._viewer;renderer=viewer.renderer;gl=renderer.gl
            w,h=rgb.shape[1],rgb.shape[0]
            values=np.empty((h,w),dtype=np.float32)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,renderer._frame_fbo)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER,0)
            gl.glReadPixels(0,0,w,h,gl.GL_DEPTH_COMPONENT,gl.GL_FLOAT,ctypes.c_void_p(values.ctypes.data))
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,0)
            values=np.flipud(values).copy()
            if not np.isfinite(values).all() or values.min()<0 or values.max()>1:raise ValueError('Invalid depth buffer')
            view=np.asarray(viewer.camera.get_view_matrix()).reshape(4,4).T
            projection=np.asarray(viewer.camera.get_projection_matrix()).reshape(4,4).T
            result.update(depth=values,view=view,projection=projection,camera_position=np.asarray(tuple(viewer.camera.pos)))
        return result
