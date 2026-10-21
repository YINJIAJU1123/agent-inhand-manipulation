"""Color-independent glyph reader with the frozen RGB sticker localizer."""
import importlib.util
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('region_reader_base', ROOT / 'scripts/astra_symbol_regions_v1/model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
crop_regions = base.crop_regions
losses = base.losses


class StickerReader(base.StickerReader):
    def glyph_logits(self, crops):
        # Shape identity is shared across sticker colors. RGB remains available
        # to the localizer, but neither color codes nor hue enter this branch.
        gray = crops.mean(1, keepdim=True)
        mean = gray.mean((-2, -1), keepdim=True)
        scale = gray.std((-2, -1), keepdim=True).clamp_min(.05)
        gray = ((gray - mean) / scale).clamp(-3, 3) / 6 + .5
        feature = self.glyph(gray.expand(-1, 3, -1, -1))
        return self.symbol(torch.cat((feature, feature.new_zeros(len(feature), 6)), -1))

    def classify(self, crops):
        return self.glyph_logits(crops).reshape(-1, 6, 6)
