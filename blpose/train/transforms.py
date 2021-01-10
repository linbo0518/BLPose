"""BLPose Train: Transforms

Author: Bo Lin (@linbo0518)
Date: 2020-01-04
"Crop",
"""

import cv2
import numpy as np
import albumentations as A

__all__ = [
    "PadIfNeeded",
    "LongestMaxSize",
    "Resize",
    "RandomRotate90",
    "RandomCrop",
    "CenterCrop",
    "Crop",
    "Rotate",
    "VerticalFlip",
    "HorizontalFlip",
    "Flip",
    "Transpose",
]


# helper function
def skip_invisible_keypoint(func):
    def wrapper(self, keypoint, *args, **kwargs):
        x, y, v = keypoint
        if v == 0:
            return x, y, v
        x, y, v = func(self, keypoint, *args, **kwargs)
        return x, y, v

    return wrapper


def blpose_keypoints(swap_keypoints: bool = False):
    def wrapper(func):
        def inner_wrapper(self, keypoints, *args, **kwargs):
            transformed_keypoints = []
            for body in keypoints:
                transformed_body = func(self, body, *args, **kwargs)
                if isinstance(self, A.Flip) and kwargs["d"] == -1:
                    pass
                elif swap_keypoints:
                    transformed_body = [transformed_body[i] for i in self.index_map]
                transformed_keypoints.append(transformed_body)
            return transformed_keypoints

        return inner_wrapper

    return wrapper


def filtering_keypoint(x, y, v, x_start, y_start, x_end, y_end):
    if x < x_start or y < y_start or x >= x_end or y >= y_end:
        return 0, 0, 0
    return x, y, v


def _crop_keypoint_by_coords(keypoint, x1, y1, x2, y2):
    x, y, v = keypoint
    x -= x1
    y -= y1
    return filtering_keypoint(x, y, v, 0, 0, x2, y2)


# transform function
def _keypoint_scale(keypoint, scale_x, scale_y):
    x, y, v = keypoint
    return round(x * scale_x), round(y * scale_y), v


def _keypoint_random_crop(
    keypoint,
    crop_height,
    crop_width,
    h_start,
    w_start,
    rows,
    cols,
):
    y1 = int((rows - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((cols - crop_width) * w_start)
    x2 = x1 + crop_width
    return _crop_keypoint_by_coords(keypoint, x1, y1, x2, y2)


def _keypoint_center_crop(keypoint, crop_height, crop_width, rows, cols):
    y1 = (rows - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (cols - crop_width) // 2
    x2 = x1 + crop_width
    return _crop_keypoint_by_coords(keypoint, x1, y1, x2, y2)


def _keypoint_rotate90(keypoint, factor, rows, cols, **params):
    x, y, v = keypoint

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y = y, (cols - 1) - x
    elif factor == 2:
        x, y = (cols - 1) - x, (rows - 1) - y
    elif factor == 3:
        x, y = (rows - 1) - y, x

    return x, y, v


def _keypoint_rotate(keypoint, angle, rows, cols, **params):
    x, y, v = keypoint
    matrix = cv2.getRotationMatrix2D(((cols - 1) * 0.5, (rows - 1) * 0.5), angle, 1.0)
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()

    return filtering_keypoint(x, y, v, 0, 0, cols, rows)


def _keypoint_vflip(keypoint, rows, cols):
    x, y, v = keypoint

    return x, (rows - 1) - y, v


def _keypoint_hflip(keypoint, rows, cols):
    x, y, v = keypoint

    return (cols - 1) - x, y, v


def _keypoint_flip(keypoint, d, rows, cols):
    if d == 0:
        keypoint = _keypoint_vflip(keypoint, rows, cols)
    elif d == 1:
        keypoint = _keypoint_hflip(keypoint, rows, cols)
    elif d == -1:
        keypoint = _keypoint_hflip(keypoint, rows, cols)
        keypoint = _keypoint_vflip(keypoint, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))

    return keypoint


def _keypoint_transpose(keypoint):
    x, y, v = keypoint

    return y, x, v


# basic
class PadIfNeeded(A.PadIfNeeded):
    def __init__(
        self,
        min_height=1024,
        min_width=1024,
        pad_height_divisor=None,
        pad_width_divisor=None,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(
            min_height,
            min_width,
            pad_height_divisor,
            pad_width_divisor,
            border_mode,
            value,
            mask_value,
            always_apply,
            p,
        )

    @skip_invisible_keypoint
    def apply_to_keypoint(
        self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params
    ):
        x, y, v = keypoint
        return x + pad_left, y + pad_top, v

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class LongestMaxSize(A.LongestMaxSize):
    def __init__(
        self, max_size=1024, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0
    ):
        super().__init__(max_size, interpolation, always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale = self.max_size / max([height, width])
        return _keypoint_scale(keypoint, scale, scale)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class Resize(A.Resize):
    def __init__(
        self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0
    ):
        super().__init__(height, width, interpolation, always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return _keypoint_scale(keypoint, scale_x, scale_y)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class RandomRotate90(A.RandomRotate90):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return _keypoint_rotate90(keypoint, factor, **params)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


# filtering keypoints
class RandomCrop(A.RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(height, width, always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_random_crop(keypoint, self.height, self.width, **params)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class CenterCrop(A.CenterCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(height, width, always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_center_crop(keypoint, self.height, self.width, **params)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class Crop(A.Crop):
    def __init__(
        self, x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0
    ):
        super().__init__(x_min, y_min, x_max, y_max, always_apply, p)

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _crop_keypoint_by_coords(
            keypoint, self.x_min, self.y_min, self.x_max, self.y_max
        )

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class Rotate(A.Rotate):
    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(
            limit, interpolation, border_mode, value, mask_value, always_apply, p
        )

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return _keypoint_rotate(keypoint, angle, **params)

    @blpose_keypoints(swap_keypoints=False)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


# swap keypoints
class VerticalFlip(A.VerticalFlip):
    def __init__(self, index_map, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.index_map = index_map

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_vflip(keypoint, **params)

    @blpose_keypoints(swap_keypoints=True)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class HorizontalFlip(A.HorizontalFlip):
    def __init__(self, index_map, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.index_map = index_map

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_hflip(keypoint, **params)

    @blpose_keypoints(swap_keypoints=True)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class Flip(A.Flip):
    def __init__(self, index_map, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.index_map = index_map

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_flip(keypoint, **params)

    @blpose_keypoints(swap_keypoints=True)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)


class Transpose(A.Transpose):
    def __init__(self, index_map, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.index_map = index_map

    @skip_invisible_keypoint
    def apply_to_keypoint(self, keypoint, **params):
        return _keypoint_transpose(keypoint)

    @blpose_keypoints(swap_keypoints=True)
    def apply_to_keypoints(self, keypoints, **params):
        return super().apply_to_keypoints(keypoints, **params)
