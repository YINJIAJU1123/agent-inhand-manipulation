"""Read glyphs independently of color and verify each predicted crop's color.

The deployed forward takes RGB only. The color head sees the measured crop, not
its query id. This lets it reject a red query that localized a magenta sticker.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('invariant_sticker_reader', ROOT/'scripts/astra_symbol_invariant_v1/model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
crop_regions = base.crop_regions
losses = base.losses


class StickerReader(base.StickerReader):
    def __init__(self):
        super().__init__()
        self.crop_color = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(4, 48), nn.SiLU(),
            nn.Flatten(), nn.Linear(48*8*8, 64), nn.SiLU(), nn.Linear(64, 7))

    def forward(self, rgb):
        output = super().forward(rgb)
        color = self.crop_color(output['crops']).reshape(-1, 6, 7)
        query_color = color.softmax(-1).diagonal(dim1=1, dim2=2)
        present = output['visible_logit'].sigmoid() * query_color
        output['crop_color_logits'] = color
        output['query_color_probability'] = query_color
        output['probability'] = torch.cat((present[..., None]*output['symbol_logits'].softmax(-1),
                                           (1-present)[..., None]), -1)
        return output


def matched_targets(predicted_boxes, boxes, labels):
    """Training only: label a predicted crop with the region it actually overlaps.

    Do not force the requested color's glyph onto a crop of a different color.
    Ambiguous overlaps receive no crop-level supervision.
    """
    a, b = predicted_boxes[:, :, None], boxes[:, None, :]
    lo = torch.maximum(a[..., :2]-a[..., 2:]/2, b[..., :2]-b[..., 2:]/2)
    hi = torch.minimum(a[..., :2]+a[..., 2:]/2, b[..., :2]+b[..., 2:]/2)
    intersection = (hi-lo).clamp_min(0).prod(-1)
    overlap = intersection/(a[..., 2:].prod(-1)+b[..., 2:].prod(-1)-intersection).clamp_min(1e-8)
    known = (labels >= 0) & (labels < 6)
    overlap = overlap.masked_fill(~known[:, None], 0.)
    best, color = overlap.max(-1)
    matched = best >= .5
    glyph = labels.gather(1, color)
    color_labels = torch.where(matched, color, torch.full_like(color, -100))
    # With no overlap with any readable symbol, reject this crop as evidence.
    # Unknown means insufficient readable evidence, not absence of all color.
    color_labels[best <= .1] = 6
    glyph_labels = torch.where(matched, glyph, torch.full_like(glyph, -100))
    return glyph_labels, color_labels


def crop_losses(model, output, rgb, boxes, labels):
    """Shared matched-crop supervision, plus color verification when present."""
    known = (labels >= 0) & (labels < 6)
    glyph, color = matched_targets(output['boxes'].detach(), boxes, labels)
    matched = glyph >= 0
    zero = output['symbol_logits'].sum()*0
    prediction = F.cross_entropy(output['symbol_logits'][matched], glyph[matched]) if matched.any() else zero
    color_loss = zero
    if hasattr(model, 'crop_color'):
        gt = crop_regions(rgb, boxes).reshape(len(rgb), 6, 3, 64, 64)
        colors = torch.arange(6, device=rgb.device)[None].expand(len(rgb), -1)
        target = model.crop_color(gt[known])
        color_loss = F.cross_entropy(target, colors[known])
        usable = color >= 0
        if usable.any():
            color_loss = color_loss + F.cross_entropy(output['crop_color_logits'][usable], color[usable])
    return prediction, color_loss
