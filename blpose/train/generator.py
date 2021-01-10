"""BLPose Train: Generator

Author: Bo Lin (@linbo0518)
Date: 2020-09-11
"""

import json
import math
import numpy as np

__all__ = ["dict2json", "json2dict", "TargetGenerator"]


def dict2json(dict_obj: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(dict_obj, f, indent=2)


def json2dict(filename: str) -> dict:
    dict_obj: dict
    with open(filename, "r") as f:
        dict_obj = json.load(f)
    return dict_obj


def _gaussian_kernel(height, width, x, y, sigma=1.75):
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    return np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma ** 2))


def _part_affinity_field(height, width, x1, y1, x2, y2, thickness=1):
    pafmap = np.zeros((2, height, width), dtype=np.float32)
    countmap = np.zeros_like(pafmap, dtype=np.uint)

    limb_vec_x = x2 - x1
    limb_vec_y = y2 - y1
    limb_len = math.sqrt(limb_vec_x ** 2 + limb_vec_y ** 2)

    if limb_len < 1e-7:
        return pafmap, countmap

    limb_unit_x = limb_vec_x / limb_len
    limb_unit_y = limb_vec_y / limb_len

    min_x = max(min(x1, x2) - thickness, 0)
    max_x = min(max(x1, x2) + thickness, width)
    min_y = max(min(y1, y2) - thickness, 0)
    max_y = min(max(y1, y2) + thickness, height)

    grid_y, grid_x = np.mgrid[min_y:max_y, min_x:max_x]
    x_12 = grid_x - x1
    y_12 = grid_y - y1

    limb_width = np.abs(x_12 * limb_unit_y - y_12 * limb_unit_x)
    limb_mask = limb_width < thickness

    pafmap[:, grid_y, grid_x] = np.repeat(
        limb_mask[np.newaxis, :, :], repeats=2, axis=0
    )
    pafmap[0, grid_y, grid_x] *= limb_unit_x
    pafmap[1, grid_y, grid_x] *= limb_unit_y

    limb_mask = (pafmap[0] != 0) | (pafmap[1] != 0)
    countmap[:, limb_mask] += 1
    return pafmap, countmap


class TargetGenerator:
    def __init__(
        self, n_keypoints, limbs, image_shape, stride, sigma=1.75, thickness=1
    ):
        self.n_keypoints = n_keypoints
        self.n_limbs = len(limbs)
        self.limbs = limbs
        height, width, _ = image_shape
        self.height = height // stride  # or round
        self.width = width // stride  # or round
        self.stride = stride
        self.sigma = sigma
        self.thickness = thickness

    def __call__(self, annotation):
        heatmap = np.zeros(
            (self.n_keypoints + 1, self.height, self.width), dtype=np.float32
        )
        pafmap = np.zeros((self.n_limbs * 2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros_like(pafmap, dtype=np.uint)
        for keypoints in annotation:
            heat = self.gen_heatmap(keypoints)
            heatmap = np.maximum(heatmap, heat)

            paf, count = self.gen_pafmap(keypoints)
            pafmap += paf
            countmap += count

        countmap[countmap == 0] = 1
        pafmap /= countmap
        heatmap[-1] = 1.0 - np.max(heatmap[:-1], axis=0)
        return heatmap, pafmap

    def gen_heatmap(self, keypoints):
        heatmap = np.zeros(
            (self.n_keypoints + 1, self.height, self.width), dtype=np.float32
        )
        for idx, keypoint in enumerate(keypoints):
            x, y, v = keypoint
            if v == 0:
                continue
            x, y = round(x / self.stride), round(y / self.stride)

            heatmap[idx] = _gaussian_kernel(self.height, self.width, x, y, self.sigma)
            heatmap[idx][heatmap[idx] > 1.0] = 1
            heatmap[idx][heatmap[idx] < 0.01] = 0
        return heatmap

    def gen_pafmap(self, keypoints):
        pafmap = np.zeros((self.n_limbs * 2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros_like(pafmap, dtype=np.uint)
        for idx, limb in zip(range(0, self.n_limbs * 2, 2), self.limbs):
            x1, y1, v1 = keypoints[limb[0]]
            x2, y2, v2 = keypoints[limb[1]]
            if v1 == 0 or v2 == 0:
                continue
            x1, y1 = round(x1 / self.stride), round(y1 / self.stride)
            x2, y2 = round(x2 / self.stride), round(y2 / self.stride)

            paf, count = _part_affinity_field(
                self.height, self.width, x1, y1, x2, y2, self.thickness
            )
            pafmap[idx : idx + 2] = paf
            countmap[idx : idx + 2] = count
        return pafmap, countmap
