"""Conservative geometric visibility labels for rendered letter-grounding data.

These are supervision proxies, not human readability ground truth. Keep the
audit contact sheet and inspect it before admitting a trained grounding model.
Simulation poses/depth are used only to label examples, never as VLM inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from goals import COLORS, SYMBOLS, NORMALS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--require_image_signal', action='store_true',
                        help='Omit geometric positives whose face pixels are too dark, flat or clipped.')
    args = parser.parse_args()
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    if not manifest['completed'] or manifest['object'] != 'RSLCube':
        raise ValueError('Require completed RSLCube capture')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    intrinsics = np.asarray(manifest['camera_intrinsics'])
    camera_position = np.asarray(manifest['camera_position_env'])
    camera_rotation = Rotation.from_quat(manifest['camera_quaternion_xyzw_ros']).as_matrix()
    keypoints = np.asarray(manifest['object_keypoints'])[1:]
    low, high = keypoints.min(0), keypoints.max(0)
    center = (low + high) / 2
    extent = (high - low) / 2
    records, counts = [], [[0, 0] for _ in COLORS]
    omitted_signal = [0 for _ in COLORS]
    with h5py.File(args.dataset / 'samples.h5', 'r') as data:
        for frame in range(len(data['rgb'])):
            position = data['object_pose'][frame, :3].astype(float)
            rotation = Rotation.from_quat(data['object_pose'][frame, 3:]).as_matrix()
            hand_depth = data['hand_depth'][frame]
            rgb = np.asarray(data['rgb'][frame]) if args.require_image_signal else None
            height, width = hand_depth.shape
            toward_camera = camera_position - position
            toward_camera /= np.linalg.norm(toward_camera)
            for goal, normal in enumerate(NORMALS):
                normal = np.asarray(normal, dtype=float)
                axis = int(np.argmax(np.abs(normal)))
                tangents = [i for i in range(3) if i != axis]
                grid = np.stack(np.meshgrid(np.linspace(-.6, .6, 9),
                                           np.linspace(-.6, .6, 9)), -1).reshape(-1, 2)
                points = np.broadcast_to(center, (len(grid), 3)).copy()
                points[:, axis] += normal[axis] * extent[axis]
                points[:, tangents] += grid * extent[tangents]
                world = points @ rotation.T + position
                camera = (world - camera_position) @ camera_rotation
                pixels = camera @ intrinsics.T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                xy = np.rint(pixels).astype(int)
                inside = ((camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                          & (xy[:, 1] >= 0) & (xy[:, 1] < height))
                visible = np.zeros(len(grid), dtype=bool)
                visible[inside] = hand_depth[xy[inside, 1], xy[inside, 0]] >= camera[inside, 2] - .001
                fraction = float(visible.mean())
                corners = pixels[[0, 8, 80, 72]]
                area = float(abs(np.dot(corners[:, 0], np.roll(corners[:, 1], 1))
                                 - np.dot(corners[:, 1], np.roll(corners[:, 0], 1))) / 2)
                facing = float(np.dot(rotation @ normal, toward_camera))
                positive = facing >= .55 and fraction >= .95 and area >= 36
                negative = facing <= -.1 or (facing >= .55 and fraction <= .05)
                if not (positive or negative):
                    continue
                signal = None
                if positive and args.require_image_signal:
                    mask = Image.new('1', (width, height))
                    ImageDraw.Draw(mask).polygon([tuple(point) for point in corners], fill=1)
                    pixels_rgb = rgb[np.asarray(mask, dtype=bool)].astype(float)
                    if not len(pixels_rgb):
                        omitted_signal[goal] += 1
                        continue
                    luminance = pixels_rgb @ np.array([.2126, .7152, .0722])
                    signal = dict(brightness_p95=float(np.quantile(pixels_rgb.max(-1), .95)),
                        luminance_span=float(np.quantile(luminance, .95)-np.quantile(luminance, .05)),
                        clipped_white_fraction=float((pixels_rgb.min(-1)>=245).mean()))
                    if (signal['brightness_p95']<48 or signal['luminance_span']<12
                            or signal['clipped_white_fraction']>=.8):
                        omitted_signal[goal] += 1
                        continue
                row = dict(frame=frame, goal_index=goal, symbol=SYMBOLS[goal], color=COLORS[goal],
                           readable=positive, image_signal=signal, split=args.split, seed=manifest['seed'],
                           env_id=int(data['env_id'][frame]), episode_id=int(data['episode_id'][frame]),
                           sim_step=int(data['sim_step'][frame]), face_cosine=facing,
                           central_region_visible_fraction=fraction, central_region_area_px2=area,
                           instruction=f"Turn the cube so the letter {SYMBOLS[goal]} faces upward.")
                records.append(row)
                counts[goal][int(positive)] += 1

    if any(min(pair) == 0 for pair in counts):
        raise ValueError(f'Missing positive/negative examples: {counts}')
    with (args.output / 'frames.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    metadata = dict(format='rslcube_letter_visibility_proxy_v1', split=args.split,
                    dataset=str(args.dataset.resolve()), dataset_manifest=manifest,
                    source_manifest_sha256=hashlib.sha256((args.dataset / 'manifest.json').read_bytes()).hexdigest(),
                    labels=len(records), negative_positive_counts=dict(zip(COLORS, counts)),
                    label_source='conservative projected central-face region, depth occlusion and front-facing geometry',
                    label_limit='Readability proxy only. Human audit and separate model validation are still required.',
                    inference_inputs='RGB plus instruction only; no poses/depth/visibility labels',
                    human_audit_status='pending', require_image_signal=args.require_image_signal,
                    image_signal_thresholds=dict(brightness_p95_min=48,luminance_span_min=12,clipped_white_fraction_max=.8) if args.require_image_signal else None,
                    omitted_positive_signal_counts=dict(zip(COLORS,omitted_signal)), thresholds=dict(front_cosine=.55, visible_fraction=.95, region_area_px2=36))
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    previews = []
    rng = np.random.default_rng(182913)
    with h5py.File(args.dataset / 'samples.h5', 'r') as data:
        for goal in range(6):
            for readable in (False, True):
                group = [row for row in records if row['goal_index'] == goal and row['readable'] == readable]
                for index in rng.choice(len(group), size=min(4, len(group)), replace=False):
                    row = group[index]
                    previews.append((row, np.asarray(data['rgb'][row['frame']])))
    canvas = Image.new('RGB', (6 * 180, 8 * 205), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (row, pixels) in enumerate(previews):
        x, y = (index % 6) * 180, (index // 6) * 205
        image = Image.fromarray(pixels)
        nonblack = np.max(pixels, axis=-1) > 20
        yy, xx = np.where(nonblack)
        if len(xx):
            image = image.crop((max(0, xx.min()-3), max(0, yy.min()-3),
                                min(pixels.shape[1], xx.max()+4), min(pixels.shape[0], yy.max()+4)))
        scale = min(176 / image.width, 170 / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        draw.text((x+2, y+172), f"{index}: {row['symbol']} {'POS' if row['readable'] else 'NEG'} f{row['frame']}", fill='black')
    canvas.save(args.output / 'audit.png')
    (args.output / 'audit_rows.json').write_text(json.dumps([row for row, _ in previews], indent=2)+'\n')
    print(json.dumps({'labels': len(records), 'counts': counts, 'human_audit_status': 'pending'}))


if __name__ == '__main__':
    main()
