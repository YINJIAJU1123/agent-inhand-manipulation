"""Request/response state pairs for diagnosis, never policy inputs or feedback."""
import copy,math
import numpy as np

def state_drift(before,after):
    q0=np.asarray(before['object_root'][3:7],dtype=float);q1=np.asarray(after['object_root'][3:7],dtype=float)
    cosine=abs(float(q0@q1))/(max(np.linalg.norm(q0)*np.linalg.norm(q1),1e-12))
    return dict(object_rotation_rad=2*math.acos(float(np.clip(cosine,0,1))),
                object_translation_m=float(np.linalg.norm(np.asarray(before['object_root'][:3])-after['object_root'][:3])),
                joint_rms=float(np.sqrt(np.mean((np.asarray(before['joint_pos'])-after['joint_pos'])**2))))

def sensor_drift(before,after):
    return {k+'_rms':float(np.sqrt(np.mean((np.asarray(before[k])-after[k])**2))) for k in ['visual','proprio','command']}

class ResponseAudit:
    def __init__(self):self.rows=[];self.latest={};self.images=[]
    def request(self,case,sequence,step,sensors,physical,snapshot,image,image_step):
        row=dict(case=case,sequence=sequence,request_step=step,request_image_step=image_step,
                 request_sensors=copy.deepcopy(sensors),request_privileged_diagnostics=copy.deepcopy(physical),
                 previous_decision=copy.deepcopy(snapshot),status='pending')
        self.rows.append(row);self.latest[case]=row
        self.images.append((f'case{case}_request{sequence}.png',image.copy()))
    def response(self,case,step,sensors,physical,snapshot,image,image_step,event):
        row=self.latest[case]
        row.update(status='delivered',response_step=step,response_image_step=image_step,
            elapsed_physics_s=(step-row['request_step'])/30,request_image_age_s=(step-row['request_image_step'])/30,
            response_sensors=copy.deepcopy(sensors),response_privileged_diagnostics=copy.deepcopy(physical),
            returned_decision=copy.deepcopy(snapshot),sensor_drift=sensor_drift(row['request_sensors'],sensors),
            privileged_state_drift=state_drift(row['request_privileged_diagnostics'],physical),async_event=copy.deepcopy(event),
            drop_within_2s=False,success_within_2s=False,followup_steps=0)
        self.images.append((f'case{case}_response{row["sequence"]}.png',image.copy()))
    def physical_step(self,case,step,drop,success):
        for row in self.rows:
            if row['case']!=case or row['status']!='delivered':continue
            delta=step-row['response_step']
            if 0<delta<=60:
                row['followup_steps']+=1;row['drop_within_2s']|=bool(drop);row['success_within_2s']|=bool(success)
    def finish(self,episodes):
        es={e['case']:e for e in episodes}
        for row in self.rows:
            e=es[row['case']];row['episode_success']=e['success'];row['episode_drop']=e['drop']
            if row['status']=='pending':row['status']='not_delivered_before_episode_end'
            else:row['followup_censored']=row['followup_steps']<60 and not row['drop_within_2s'] and not row['success_within_2s']
        return self.rows
