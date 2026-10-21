"""Finite local collection and training pipeline. Does not access any API."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT,PYTHON,save,sha


def environment_variables(out):
    env=dict(os.environ,PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",
             CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0",
             ASTRA_RENDER_GPU="0",OMNI_KIT_ACCEPT_EULA="YES",ACCEPT_EULA="Y",
             TMPDIR=str(out/"tmp"))
    env["PYTHONPATH"]=":".join(str(ROOT/x) for x in
        ["outputs/vlm_dependencies","outputs/learning_dependencies",
         "outputs/video_dependencies","scripts/agentic"])
    return env


KIT = ["--kit_args","--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ("data","logs","tmp"):
        (out/name).mkdir()
    state=dict(pid=os.getpid(),status="collecting",completed_shards=0)
    def progress(**kw):
        state.update(kw,updated=time.time());save(out/"progress.json",state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path("/tmp/agent_inhand_paired_gpu_0.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment_variables(out);child=None
    schedule=[("train",s) for s in range(96111,96119)]+[("val",96211),("val",96212)]
    frozen={str(ROOT/"scripts/semantic_completion_v1"/f):sha(ROOT/"scripts/semantic_completion_v1"/f)
            for f in ("common.py","collect.py","model.py","train.py","prepare.py")}
    save(out/"protocol.json",dict(schedule=schedule,formal_eval_seeds=[86101,86102],
         source_sha256=frozen,steps_per_shard=900,training_api_calls=0,
         first_smoke_excluded="pilot_20261020/data/train_96101 used the old depth proxy; not included",
         selection="Best validation loss only; no test-seed checkpoint selection"))
    def run(command,name,timeout=2400):
        nonlocal child
        for path,digest in frozen.items():
            if sha(path)!=digest:raise RuntimeError("Training source changed after freeze")
        save(out/"logs"/(name+"_command.json"),command)
        with (out/"logs"/(name+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.time()
            while child.poll() is None:
                progress(current=name,child_pid=child.pid,wall_s=time.time()-start)
                failure=out/"data"/name/"failure.json"
                if failure.exists():raise RuntimeError(f"{name}: collection failure")
                if time.time()-start>timeout:raise TimeoutError(name)
                time.sleep(5)
            if child.returncode:raise RuntimeError(f"{name}: exit {child.returncode}")
    try:
        progress()
        for split,seed in schedule:
            name=f"{split}_{seed}"
            run([PYTHON,str(ROOT/"scripts/semantic_completion_v1/collect.py"),
                 "--headless","--device","cuda:0","--output",str(out/"data"/name),
                 "--seed",str(seed),"--split",split,"--max-steps","900"]+KIT,name)
            progress(completed_shards=state["completed_shards"]+1)
        progress(status="training")
        run([PYTHON,str(ROOT/"scripts/semantic_completion_v1/train.py"),"--data",str(out/"data"),
             "--output",str(out/"train"),"--epochs","35"],"train")
        result=json.loads((out/"train/result.json").read_text())
        progress(status="completed",child_pid=None,training_result=result)
    except BaseException as e:
        progress(status="interrupted" if isinstance(e,KeyboardInterrupt) else "failed",error=str(e))
        raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        lock.close()


if __name__=="__main__":
    main()
