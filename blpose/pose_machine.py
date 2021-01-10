"""BLPose Pose Machine

Author: Bo Lin (@linbo0518)
Date: 2020-12-13
"""

import cv2
import torch
from . import libpose as lib
from .utils.profiler import profiling

__all__ = ["PoseMachine"]


class PoseMachine:
    def __init__(self, model, weights, device="cuda"):
        self.model = model
        self.model.load_state_dict(torch.load(weights))
        self.model.to(device)
        self.model.eval()
        self.device = device

    @profiling("model inference", n_divider=0)
    @torch.no_grad()
    def _inference(self, inputs):
        inputs = torch.from_numpy(inputs).to(self.device)
        pafmap, heatmap = self.model(inputs)  # TODO: maybe modify in the future
        pafmap = pafmap.cpu().numpy()
        heatmap = heatmap.cpu().numpy()
        return pafmap, heatmap

    def image_predict(self, ori_img):
        box_size = 368
        stride = 8
        pad_value = 128
        sigma = 3
        threshold1 = 0.1
        threshold2 = 0.05
        num_sample = 10

        ori_h, ori_w, _ = ori_img.shape

        scale = box_size / ori_h
        rsz_img = cv2.resize(ori_img, (0, 0), fx=scale, fy=scale)

        rsz_h, rsz_w, _ = rsz_img.shape

        pad_img = lib.pad_right_down(rsz_img, rsz_h, rsz_w, stride, pad_value)

        pad_h, pad_w, _ = pad_img.shape

        psd_img = lib.preprocess(pad_img, pad_h, pad_w)

        pafmap, heatmap = self._inference(psd_img)

        heatmap = lib.calibrate_output(heatmap, rsz_h, rsz_w, stride)
        heatmap = cv2.resize(heatmap, (ori_w, ori_h))
        pafmap = lib.calibrate_output(pafmap, rsz_h, rsz_w, stride)
        pafmap = cv2.resize(pafmap, (ori_w, ori_h))

        peaks = lib.find_peaks(heatmap, ori_h, ori_w, sigma, threshold1)

        paired_limbs, unpaired_limb_indices = lib.pair_points(
            pafmap, peaks, 0.5 * ori_h, num_sample, threshold2
        )

        candidate, bodies = lib.pair_limbs(peaks, paired_limbs, unpaired_limb_indices)
        return candidate, bodies

    @staticmethod
    def draw_pose(ori_img, candidate, bodies):
        return lib.draw_pose(ori_img, candidate, bodies)
