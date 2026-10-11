"""Closed-loop evaluation for the frozen-VLM Revo3 student.

The SigLIP2 encoder is frozen and runs only as an observation encoder.  The
GRU/action head remains the single low-level controller and emits the same
21-dimensional bounded command as the Revo3 teacher interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

# Keep the evaluator runnable from a clean IsaacLab shell, as used on the
# remote GPU host, without relying on an externally exported PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEXHAND_SOURCE = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(DEXHAND_SOURCE) not in sys.path:
    sys.path.insert(0, str(DEXHAND_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default=None, help="Override the registered evaluation task.")
parser.add_argument("--model", default="google/siglip2-base-patch16-224")
parser.add_argument("--cached-features", default=None,
                    help="Deprecated for closed-loop evaluation; cached features are offline-only.")
parser.add_argument("--freeze-manifest", default=str(REPO_ROOT / "configs" / "visual_student_freeze.json"))
parser.add_argument("--structured-camera", action="store_true",
                    help="Use the deterministic 12-D RGB statistics encoder on live camera frames.")
parser.add_argument("--semantic-camera", action="store_true",
                    help="Use deterministic marker-aware RGB features on live camera frames.")
parser.add_argument("--semantic-grid-size", type=int, default=0,
                    help="Append a low-resolution RGB grid before the semantic marker slots.")
parser.add_argument("--semantic-depth-grid-size", type=int, default=0,
                    help="Append a low-resolution normalized depth grid before the semantic marker slots.")
parser.add_argument("--zero-language", action="store_true",
                    help="Zero the language feature while evaluating a control student.")
parser.add_argument("--language-mode", choices=("onehot", "hash", "vlm"), default="onehot",
                    help="Structured face one-hot or deterministic text-string features.")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--vision-stride", type=int, default=4, help="Run the frozen VLM every N control steps.")
parser.add_argument("--action-scale", type=float, default=None,
                    help="Override the frozen action scale; normally read from the freeze manifest.")
parser.add_argument("--report", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
freeze_manifest = json.loads(Path(args.freeze_manifest).read_text())
if args.cached_features:
    raise ValueError("cached features are offline-only and cannot be used for closed-loop evaluation; use --semantic-camera or a live VLM")
manifest_scale = float(freeze_manifest["action_contract"]["action_scale"])
if args.action_scale is None:
    args.action_scale = manifest_scale
elif abs(float(args.action_scale) - manifest_scale) > 1e-8:
    raise ValueError(f"action scale {args.action_scale} disagrees with frozen manifest {manifest_scale}")
# Camera-enabled teacher tasks still need Isaac's camera extension even when
# the policy consumes cached features; otherwise the environment cannot spawn.
args.enable_cameras = args.structured_camera or args.semantic_camera or args.semantic_grid_size > 0 or args.semantic_depth_grid_size > 0 or args.cached_features is None or "Visual" in (args.task or "")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
try:
    from transformers import AutoModel, AutoProcessor  # noqa: E402
except ImportError:  # pragma: no cover - only needed for the SigLIP path
    AutoModel = AutoProcessor = None

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import BrainCo_DexHand  # noqa: F401, E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_feature_student import VisualLanguageStudent, VisualStudentBatch  # noqa: E402
from BrainCo_DexHand.algo.agentic.language_goal import FACE_NAMES  # noqa: E402
from BrainCo_DexHand.algo.agentic.text_features import hashed_text_features  # noqa: E402


def _features(model, processor, images, device, text_features):
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items() if torch.is_tensor(v)}
    image_features = model.get_image_features(**image_inputs)
    return (
        torch.nn.functional.normalize(image_features.float(), dim=-1),
        text_features,
    )


def _text_features(model, processor, texts, device):
    inputs = processor(text=[str(text) for text in texts], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    if hasattr(model, "get_text_features"):
        output = model.get_text_features(**inputs)
    else:
        encoded = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        output = encoded.pooler_output if hasattr(encoded, "pooler_output") else encoded.last_hidden_state[:, 0]
        if hasattr(model, "text_projection"):
            output = model.text_projection(output)
    return torch.nn.functional.normalize(output.float(), dim=-1)


def _structured_rgb_features(rgb: torch.Tensor) -> torch.Tensor:
    """Match ``cache_structured_features.py`` for live RGB frames."""
    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    _, height, width, _ = rgb.shape
    center = rgb[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return torch.cat((rgb.mean((1, 2)), rgb.std((1, 2)),
                      center.mean((1, 2)), center.std((1, 2))), dim=-1)


def _semantic_rgb_features(rgb: torch.Tensor, grid_size: int = 0, depth: torch.Tensor | None = None, depth_grid_size: int = 0) -> torch.Tensor:
    """Match ``cache_semantic_features.py`` for live RGB frames."""
    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    n, height, width, _ = rgb.shape
    pixels = rgb.reshape(n, height * width, 3)
    colors = torch.tensor(
        [[0.85, 0.05, 0.05], [0.05, 0.75, 0.15], [0.05, 0.25, 0.90],
         [0.95, 0.75, 0.05], [0.80, 0.05, 0.75], [0.05, 0.80, 0.85]],
        device=rgb.device, dtype=rgb.dtype,
    )
    distance = (pixels[:, None] - colors[None, :, None]).square().mean(dim=-1)
    weights = torch.exp(-distance / 0.025) * (pixels.mean(dim=-1)[:, None] > 0.06)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype),
        torch.linspace(-1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype), indexing="ij")
    xx, yy = xx.reshape(1, 1, -1), yy.reshape(1, 1, -1)
    mass = weights.mean(dim=-1)
    denom = weights.sum(dim=-1).clamp_min(1e-6)
    cx = (weights * xx).sum(dim=-1) / denom
    cy = (weights * yy).sum(dim=-1) / denom
    sx = torch.sqrt((weights * (xx - cx[..., None]).square()).sum(dim=-1) / denom)
    sy = torch.sqrt((weights * (yy - cy[..., None]).square()).sum(dim=-1) / denom)
    marker = torch.stack((mass, cx, cy, sx + sy), dim=-1).reshape(n, -1)
    center = rgb[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    global_stats = torch.cat((rgb.mean((1, 2)), rgb.std((1, 2)),
                              center.mean((1, 2)), center.std((1, 2))), dim=-1)
    if grid_size < 0:
        raise ValueError("grid_size must be non-negative")
    pieces = [global_stats]
    if grid_size:
        grid = torch.nn.functional.interpolate(
            rgb.permute(0, 3, 1, 2), size=(grid_size, grid_size),
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(n, -1)
        pieces.append(grid)
    if depth_grid_size:
        if depth is None:
            raise ValueError("depth is required when depth_grid_size is nonzero")
        depth = torch.nan_to_num(depth.to(rgb.dtype), nan=2.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0) / 2.0
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        dgrid = torch.nn.functional.interpolate(
            depth.permute(0, 3, 1, 2), size=(depth_grid_size, depth_grid_size),
            mode="bilinear", align_corners=False).reshape(n, -1)
        pieces.append(dgrid)
    pieces.append(marker)
    return torch.cat(pieces, dim=-1)


def _instruction_faces(raw, device):
    """Parse the external language command without reading target_face in the actor path."""
    names = {name: index for index, name in enumerate(FACE_NAMES)}
    values = []
    for text in raw.current_instructions():
        tokens = str(text).lower().split()
        matches = [names[name] for name in names if name in tokens]
        if len(matches) != 1:
            raise ValueError(f"cannot parse exactly one face from instruction: {text!r}")
        values.append(matches[0])
    return torch.as_tensor(values, dtype=torch.long, device=device)


def main():
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    task = args.task or (
        "BrainCo-Direct-Revo3-SemanticReorient-Cube-v0"
        if args.cached_features and not args.structured_camera and not args.semantic_camera
        else "BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0"
    )
    cfg = parse_env_cfg(task, device=str(device), num_envs=args.num_envs)
    cfg.max_consecutive_success = 1
    cfg.record_eval_metrics = True
    cfg.seed = 123
    if hasattr(cfg, "goal_yaw"):
        # Collection and live evaluation share the deterministic yaw contract;
        # language names the requested surface without hiding an extra yaw.
        cfg.goal_yaw = 0.0
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("language_mode") and ckpt["language_mode"] != args.language_mode:
        raise ValueError(f"checkpoint language mode {ckpt['language_mode']} disagrees with --language-mode {args.language_mode}")
    if ckpt.get("freeze_id") and ckpt["freeze_id"] != freeze_manifest["freeze_id"]:
        raise ValueError(f"student checkpoint freeze_id {ckpt['freeze_id']} does not match {freeze_manifest['freeze_id']}")
    if ckpt.get("action_scale") is not None and abs(float(ckpt["action_scale"]) - float(args.action_scale)) > 1e-8:
        raise ValueError("checkpoint action scale disagrees with the frozen evaluation scale")
    policy = VisualLanguageStudent(
        ckpt["rgb_dim"], ckpt["language_dim"], ckpt["proprio_dim"], ckpt["action_dim"],
        memory_mode=ckpt.get("memory_mode", "plain")
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    face_names = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    text_prompts = [f"show the {x} marker" for x in face_names]
    processor = None
    vlm = None
    text_features = None
    # ``vlm`` mode uses the actual instruction string for every episode.
    # The image path remains the existing RGB-D semantic frontend when
    # --semantic-camera is supplied; SigLIP is used only for language here.
    if args.language_mode == "vlm" or not (args.structured_camera or args.semantic_camera):
        if AutoProcessor is None or AutoModel is None:
            raise RuntimeError("transformers is required for SigLIP evaluation")
        processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
        vlm = AutoModel.from_pretrained(args.model).to(device).eval()
        if args.language_mode != "vlm":
            text_features = _text_features(vlm, processor, text_prompts, device)
    else:
        prompts = [f"show the {x} marker" for x in face_names]
        if args.language_mode == "hash":
            text_features = hashed_text_features(prompts, dim=ckpt["language_dim"]).to(device)
        else:
            text_features = torch.eye(len(face_names), device=device)
        if args.zero_language:
            text_features.zero_()

    obs, _ = env.reset()
    if args.language_mode == "vlm":
        current_instruction_texts = [str(text) for text in raw.current_instructions()]
        text_features = _text_features(vlm, processor, current_instruction_texts, device)
        if args.zero_language:
            text_features.zero_()
        last_instruction_texts = current_instruction_texts
    else:
        last_instruction_texts = None
    n = raw.num_envs
    history = int(ckpt.get("history", 8))
    image_hist = torch.zeros((n, history, ckpt["rgb_dim"]), device=device)
    prop_hist = torch.zeros((n, history, ckpt["proprio_dim"]), device=device)
    image_features = torch.zeros((n, ckpt["rgb_dim"]), device=device)
    history_initialized = torch.zeros(n, dtype=torch.bool, device=device)
    done_count = 0
    step_count = 0
    episode_steps = torch.zeros(n, dtype=torch.long, device=device)
    episode_success = torch.zeros(n, dtype=torch.bool, device=device)
    episode_ever_reach = torch.zeros(n, dtype=torch.bool, device=device)
    episode_first_reach = torch.full((n,), -1, dtype=torch.long, device=device)
    episode_drop = torch.zeros(n, dtype=torch.bool, device=device)
    episode_min_error = torch.full((n,), float("inf"), device=device)
    instruction_face = _instruction_faces(raw, device)
    episode_face = instruction_face.clone()
    records = []
    max_vector_steps = args.max_steps * math.ceil(args.episodes / max(n, 1))
    while app.is_running() and done_count < args.episodes and step_count < max_vector_steps:
        # Keep state tensors mutable across episode resets.
        with torch.no_grad():
            capture_step = step_count % max(args.vision_stride, 1) == 0
            if capture_step:
                camera = raw.capture_camera()
                rgb = camera["rgb"][..., :3].to(torch.uint8)
                if args.semantic_camera:
                    image_features = _semantic_rgb_features(rgb, args.semantic_grid_size, camera.get("depth"), args.semantic_depth_grid_size)
                elif args.structured_camera:
                    image_features = _structured_rgb_features(rgb)
                else:
                    images = [Image.fromarray(x.cpu().numpy()) for x in rgb]
                    image_features, _ = _features(vlm, processor, images, device, text_features)
            instruction_face = _instruction_faces(raw, device)
            if args.language_mode == "vlm":
                current_instruction_texts = [str(text) for text in raw.current_instructions()]
                if current_instruction_texts != last_instruction_texts:
                    text_features = _text_features(vlm, processor, current_instruction_texts, device)
                    if args.zero_language:
                        text_features.zero_()
                    last_instruction_texts = current_instruction_texts
                face_text = text_features
            else:
                face_text = text_features[instruction_face]
            if capture_step:
                # Training samples are recorded every vision_stride control steps.
                # Advance both histories on that same clock and pad a new episode
                # with its first observation, as FeatureSequenceDataset does.
                proprio = raw.compute_student_proprio().float()
                image_hist = torch.cat((image_hist[:, 1:], image_features[:, None]), dim=1)
                prop_hist = torch.cat((prop_hist[:, 1:], proprio[:, None]), dim=1)
                new_episode = ~history_initialized
                if new_episode.any():
                    image_hist[new_episode] = image_features[new_episode, None, :]
                    prop_hist[new_episode] = proprio[new_episode, None, :]
                    history_initialized[new_episode] = True
            out = policy(VisualStudentBatch(image_hist, face_text, prop_hist))
            # BrainCo direct tasks consume normalized [-1, 1] commands.
            # The environment maps them to joint targets internally.
            action = out["action"]
            _, _, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).to(device=device, dtype=torch.bool)
            metrics = info["semantic_metrics"]
            episode_steps += 1
            episode_min_error = torch.minimum(episode_min_error, metrics["orientation_error"])
            reached = metrics["goal_reached"] & ~metrics["dropped"]
            episode_ever_reach |= reached
            first_reach = reached & (episode_first_reach < 0)
            episode_first_reach = torch.where(first_reach, episode_steps, episode_first_reach)
            episode_success |= reached
            episode_drop |= metrics["dropped"]
            # Apply an evaluator-side horizon even when the environment does
            # not emit a terminal signal.  This keeps each rollout bounded
            # and lets us compare success/timeout rates fairly.
            timeout = episode_steps >= args.max_steps
            done_eval = done | timeout
        timeout_ids = torch.nonzero(timeout & ~done, as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            raw._reset_idx(timeout_ids)
        for idx_t in torch.nonzero(done_eval, as_tuple=False).flatten():
            idx = int(idx_t.item())
            if done_count >= args.episodes:
                break
            records.append({
                "face": int(episode_face[idx].item()),
                "success": bool(episode_success[idx].item()) and not bool(episode_drop[idx].item()),
                "ever_reach": bool(episode_ever_reach[idx].item()),
                "time_to_first_reach_steps": int(episode_first_reach[idx].item()),
                "drop": bool(episode_drop[idx].item()),
                "timeout": bool(timeout[idx].item()),
                "min_orientation_error_rad": float(episode_min_error[idx].item()),
                "steps": int(episode_steps[idx].item()),
            })
            done_count += 1
            episode_steps[idx] = 0
            episode_success[idx] = False
            episode_ever_reach[idx] = False
            episode_first_reach[idx] = -1
            episode_drop[idx] = False
            episode_min_error[idx] = float("inf")
            instruction_face = _instruction_faces(raw, device)
            episode_face[idx] = instruction_face[idx]
            image_hist[idx] = 0
            prop_hist[idx] = 0
            image_features[idx] = 0
            history_initialized[idx] = False
        step_count += 1
        if step_count % 50 == 0:
            print(f"[eval] vector_steps={step_count} episodes={done_count}", flush=True)

    # A visual student can fail to trigger the environment's terminal condition
    # (for example, it may neither reach the pose nor drop the object).  Keep
    # those partial rollouts in the report as explicit timeouts instead of
    # silently returning ``episodes: 0``.
    if done_count < args.episodes:
        for idx in range(n):
            if done_count >= args.episodes or episode_steps[idx].item() == 0:
                continue
            records.append({
                "face": int(episode_face[idx].item()),
                "success": False,
                "ever_reach": bool(episode_ever_reach[idx].item()),
                "time_to_first_reach_steps": int(episode_first_reach[idx].item()),
                "drop": bool(episode_drop[idx].item()),
                "timeout": True,
                "min_orientation_error_rad": float(episode_min_error[idx].item()),
                "steps": int(episode_steps[idx].item()),
            })
            done_count += 1

    per_face = defaultdict(list)
    for record in records:
        per_face[record["face"]].append(record)
    report = {
        "task": task, "checkpoint": os.path.abspath(args.checkpoint), "model": args.model,
        "vision_stride": args.vision_stride,
        "action_scale": args.action_scale,
        "cached_features": None,
        "freeze_id": freeze_manifest["freeze_id"],
        "protocol_id": freeze_manifest["visual_protocol"]["protocol_id"],
        "stage": freeze_manifest["visual_protocol"]["stage"],
        "teacher_checkpoint_sha256": freeze_manifest["teacher"]["sha256"],
        "structured_camera": args.structured_camera,
        "semantic_camera": args.semantic_camera,
        "semantic_grid_size": args.semantic_grid_size,
        "semantic_depth_grid_size": args.semantic_depth_grid_size,
        "zero_language": args.zero_language,
        "language_mode": args.language_mode,
        "language_model": args.model if args.language_mode == "vlm" else None,
        "episodes": len(records), "vector_steps": step_count,
        "success_rate": sum(x["success"] for x in records) / max(len(records), 1),
        "drop_rate": sum(x["drop"] for x in records) / max(len(records), 1),
        "per_face": {str(k): {"episodes": len(v), "success_rate": sum(x["success"] for x in v) / len(v),
                              "drop_rate": sum(x["drop"] for x in v) / len(v)} for k, v in sorted(per_face.items())},
        "records": records,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
