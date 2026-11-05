"""Shared RGB-only crop for both tracks; no target or geometry metadata."""
import numpy as np


def rgb_crop_box(rgb):
    image = np.asarray(rgb, dtype=np.uint8)
    small = image[::4, ::4].astype(np.float32) / 255.
    mask = ((small.max(-1) - small.min(-1)) > .3) & (small.max(-1) > .3)
    seen = np.zeros(mask.shape, dtype=bool)
    best = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        component = []
        while stack:
            yy, xx = stack.pop()
            component.append((yy, xx))
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    if len(best) < 8:
        return (0, 0, image.shape[1], image.shape[0])
    yy, xx = zip(*best)
    pad = 16
    return (max(0, min(xx) * 4 - pad), max(0, min(yy) * 4 - pad),
            min(image.shape[1], (max(xx) + 1) * 4 + pad), min(image.shape[0], (max(yy) + 1) * 4 + pad))
