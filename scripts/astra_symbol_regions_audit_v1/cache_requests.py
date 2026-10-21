"""Cache recorded request RGB separately from labels for sensor-only shadow evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/video_dependencies'))
import imageio_ffmpeg


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames, rows, scores, sources, omissions = [], [], [], [], []
    started = time.time()
    videos = 0
    for report in sorted(args.source.glob('eval_*/report.json')):
        data = json.loads(report.read_text())
        initial = report.parent / 'initial.json'
        layouts = json.loads(initial.read_text())['layout']
        for episode in data['episodes']:
            case = episode['case']
            video = report.parent / f'case{case}.mp4'
            if not video.exists():
                omissions.append(dict(source=str(report.parent), case=case, reason='no_recorded_video'))
                continue
            requests = {}
            for event in episode['async_events']:
                step = event['request_step']
                if step % 2 or step >= episode['finish_step']:
                    continue
                if step // 2 in requests:
                    raise ValueError('Multiple recorded requests on the same frame')
                requests[step // 2] = event
            tools = {row['step']: row for row in episode['trace']}
            found = set()
            reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
            metadata = next(reader)
            width, height = metadata['size']
            if (width, height) != (512, 512) or abs(metadata['fps'] - 15) > .01:
                raise ValueError('Unexpected recording geometry or cadence')
            try:
                for index, frame in enumerate(reader):
                    if index not in requests:
                        continue
                    event = requests[index]
                    step = event['request_step']
                    frames.append(torch.from_numpy(np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3).copy()).permute(2, 0, 1))
                    rows.append(dict(source=str(report.parent.resolve()), case=case, step=step, frame_index=index, request_sequence=event['sequence']))
                    tool = tools.get(step, {})
                    arguments = tool.get('arguments', {})
                    # Scoring-only fields are stored in another file, never in model inputs.
                    scores.append(dict(layout=layouts[case], target_symbol=episode['target_symbol'], target_color=episode['target_color'],
                                       tool=tool.get('tool'), color=arguments.get('color'), goal_symbol=arguments.get('goal_symbol'),
                                       request_status=event['status'], delivery_step=event.get('delivery_step')))
                    found.add(index)
                    if found == set(requests):
                        break
            finally:
                reader.close()
            for index in sorted(set(requests) - found):
                omissions.append(dict(source=str(report.parent), case=case, reason='request_frame_not_recorded', frame_index=index))
            sources.append(dict(video=str(video.resolve()), video_sha256=sha(video), report_sha256=sha(report), initial_sha256=sha(initial)))
            videos += 1
            save(args.output / 'progress.json', dict(status='caching', videos=videos, frames=len(frames), elapsed_s=time.time() - started))
    if not frames:
        raise ValueError('No recorded requests found')
    torch.save(dict(rgb=torch.stack(frames)), args.output / 'inputs.pt')
    save(args.output / 'rows.json', rows)
    save(args.output / 'scoring.json', scores)
    save(args.output / 'manifest.json', dict(videos=videos, request_frames=len(frames), sources=sources, omissions=omissions,
         inputs_sha256=sha(args.output / 'inputs.pt'), rows_sha256=sha(args.output / 'rows.json'), scoring_sha256=sha(args.output / 'scoring.json'),
         source_sha256=sha(__file__), cloud_calls=0,
         note='Previously inspected development recordings. Video step 2*k is captured before the physics action, matching request RGB time. Compression and the recording overlay can differ from raw request PNGs. No closed-loop outcomes changed.'))
    save(args.output / 'progress.json', dict(status='completed', videos=videos, frames=len(frames), elapsed_s=time.time() - started))
    print(json.dumps(dict(videos=videos, request_frames=len(frames), omissions=len(omissions), output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
