"""Frozen RGB-D frontend, copied verbatim from the validated v2 evaluator."""
import torch

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
