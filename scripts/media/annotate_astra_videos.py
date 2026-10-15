"""Add factual tool-call captions to saved RGB recordings; preserve raw videos."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS={n:ImageFont.truetype(FONT,n) for n in [16,19,23,29]}
W,H=1120,704
LABELS={'inspect_face':'主动观察：转动所选颜色面，获得新视角','reorient_target':'执行目标：调用冻结的 RGB-D student','verify_target':'再次验证：维持该技能 20 步后重新看图','stop':'模型提出停止；是否成功由独立判分器判断','abstain':'模型放弃：当前证据不足'}
SYMS={'star':'星形','circle':'圆形','triangle':'三角形','cross':'十字','diamond':'菱形','bars':'双横杠'}
COLORS={'red':'红色','green':'绿色','blue':'蓝色','yellow':'黄色','magenta':'品红','cyan':'青色'}

def put(draw,xy,text,size=19,fill='#dbe4ee'):
    draw.text(xy,str(text),font=FONTS[size],fill=fill)

def wrap(draw,text,width,size):
    lines=[];line=''
    for word in str(text).split(' '):
        candidate=(line+' '+word).strip()
        if draw.textlength(candidate,font=FONTS[size])>width and line:lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    return lines

def annotate(report,e,dest,label):
    source=report.parent/e['video']['path'];output=dest/(report.parent.name+f"_case{e['case']}.mp4")
    if output.exists():return output
    reader=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24');meta=next(reader)
    width,height=meta['size'];fps=e['video']['fps'];trace=e['trace'];last=None
    writer=imageio_ffmpeg.write_frames(str(output),(W,H),fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','21','-movflags','+faststart'],ffmpeg_log_level='error');writer.send(None)
    try:
      for idx,frame in enumerate(reader):
        step=idx*2;seen=[x for x in trace if x['step']<=step];current=seen[-1] if seen else None
        canvas=Image.new('RGB',(W,H),'#0c1420');d=ImageDraw.Draw(canvas)
        put(d,(24,16),f"Astra 闭环机器人 | {SYMS[e['target_symbol']]}朝上",29,'#ffffff')
        put(d,(24,58),f"{label} · seed {e['seed']} · case {e['case']} · {report.parent.name.split('_')[-2]}",16,'#96abc2')
        for j,line in enumerate(wrap(d,e['instruction'],1060,19)[:2]):put(d,(24,85+j*25),line)
        rgb=np.frombuffer(frame,dtype=np.uint8).reshape(height,width,3)
        canvas.paste(Image.fromarray(rgb),(24,143))
        put(d,(566,143),f"仿真时间 {step/30:.2f}s  |  step {step}",23)
        put(d,(566,180),f"已调用 Astra {len(seen)} / 6 次",23,'#63d4c6')
        if current:
          args=current['arguments'];tool=current['tool']
          put(d,(566,222),tool,23,'#ffffff')
          put(d,(566,258),LABELS[tool],16)
          put(d,(566,289),f"选择面：{COLORS.get(args.get('color'),'—')}    执行步数：{args.get('steps',20 if tool=='verify_target' else '—')}",19)
          put(d,(566,326),'Astra 的原始视觉依据：',16,'#96abc2')
          for j,line in enumerate(wrap(d,args.get('evidence',''),524,16)[:6]):put(d,(566,352+j*23),line,16)
        put(d,(566,502),'实际调用轨迹（仿真时间）',16,'#96abc2')
        for j,t in enumerate(seen[-6:]):put(d,(566,526+j*21),f"{t['step']/30:5.2f}s   {t['tool']}  {t['arguments'].get('color','')}",16)
        put(d,(24,668),'原始相机画面；15 fps / 仿真时间；视频省略同步 API 等待，末帧为动作前画面。',16,'#96abc2')
        writer.send(np.asarray(canvas));last=canvas
      if last:
        card=Image.new('RGB',(W,H),'#0c1420');d=ImageDraw.Draw(card)
        outcome='成功' if e['success'] else '提前停止，判分未达标' if e['false_stop'] else '掉落' if e['drop'] else '未达标 / 放弃'
        put(d,(45,100),f"独立判分：{outcome}",29,'#63d4c6' if e['success'] else '#f3b37c')
        put(d,(45,165),f"目标绑定正确：{'是' if e['grounding_correct'] else '否'}   |   Astra 调用：{e['calls']} 次",23)
        put(d,(45,215),'成功：正确语义绑定 + 执行该目标 + 全姿态误差 ≤ 0.16 rad + 未掉落',19)
        put(d,(45,260),'成功由评测器停止；不等价于模型自主确认完成。',19)
        put(d,(45,320),label,19,'#96abc2')
        for _ in range(round(fps*2)):writer.send(np.asarray(card))
    finally:writer.close();reader.close()
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    root=Path(a.input).resolve();dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=True);items=[]
    for report in sorted(root.glob('seed*/report.json')):
      data=json.loads(report.read_text())
      for e in data['episodes']:
        if not e.get('video'):continue
        path=annotate(report,e,dest,a.label)
        items.append(dict(file=path.name,source_report=str(report),case=e['case'],symbol=e['target_symbol'],instruction=e['instruction'],success=e['success'],drop=e['drop'],false_stop=e['false_stop'],calls=e['calls']))
    (dest/'index.json').write_text(json.dumps(items,indent=2,ensure_ascii=False)+'\n')
    if items:
      listing=dest/'concat.txt';listing.write_text(''.join("file '"+x['file']+"'\n" for x in items))
      subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(dest/'all_cases.mp4')],check=True)
    print(json.dumps({'videos':len(items),'output':str(dest)},ensure_ascii=False))

if __name__=='__main__':main()
