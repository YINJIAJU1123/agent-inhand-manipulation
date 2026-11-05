"""Run a real simulation job and reject solver capacity warnings immediately."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

def capacity_warning(line):
    text=line.lower()
    return (("overflow" in text and ("broadphase" in text or "constraint" in text or "contact" in text))
            or ("please increase" in text and ("nconmax" in text or "njmax" in text or "naconmax" in text))
            or ("exceeded" in text and ("mjwarp limit" in text or "contact capacity" in text)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:parser.error("A simulation command is required")
    args.log.parent.mkdir(parents=True,exist_ok=True)
    warning=None
    process=None
    code=1
    try:
        with args.log.open("x") as output:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,errors="replace",bufsize=1,start_new_session=True,
                env={**os.environ,"PYTHONUNBUFFERED":"1"})
            for line in process.stdout:
                output.write(line)
                output.flush()
                if capacity_warning(line):
                    warning=line.strip()
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    break
            code=process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
        record=dict(command=command,exit_code=code,capacity_warning=warning,
                    physical_log_valid=code==0 and warning is None)
        args.log.with_suffix(".validity.json").write_text(json.dumps(record,indent=2)+"\n")
    if warning:print("INVALID_PHYSICS: "+warning,file=sys.stderr)
    return 86 if warning else (code if code>=0 else 128-code)

if __name__=="__main__":sys.exit(main())
