"""BLPose Test: Train Transforms

Author: Bo Lin (@linbo0518)
Date: 2020-12-30
"""

import sys

sys.path.append("../")
from blpose.train.transforms import *
from blpose.train.dataset import COCOKeypoints, COCOUtils


def test_transformer(transformer, **kwargs):
    transformed = transformer(**kwargs)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]
    transformed_masks = transformed["masks"]

    return transformed_image, transformed_keypoints, transformed_masks


def test_pipeline(coco_keypoints, idx, transformer, show_anno=False, verbose=False):
    print(f"Test {transformer}")
    image, keypoints_all, mask_all, mask_loss, anno = coco_keypoints[idx]

    if show_anno:
        coco_keypoints.show_annotations(image, anno)

    mask_all.append(mask_loss)
    transformed_image, transformed_keypoints, transformed_masks = test_transformer(
        transformer, image=image, keypoints=keypoints_all, masks=mask_all
    )

    transformed_mask_loss = transformed_masks.pop(-1)
    new_anno = coco_keypoints.generate_annotations(
        anno, transformed_keypoints, transformed_masks, transformed_mask_loss
    )
    if verbose:
        for idx, (ori_kpts, trans_kpts) in enumerate(
            zip(keypoints_all, transformed_keypoints)
        ):
            print(f"Keypoints: {idx:02}")
            print(f"  ori: {ori_kpts}")
            print(f"trans: {trans_kpts}")

    if show_anno:
        coco_keypoints.show_annotations(transformed_image, new_anno)
    print("Done")


coco_dir = "/Users/linbo0518/Movies/COCO2017"
coco_keypoints = COCOKeypoints(coco_dir)


augmenters = [
    PadIfNeeded(min_height=600, min_width=800, p=1.0),
    LongestMaxSize(max_size=800, p=1.0),
    Resize(height=200, width=500, p=1.0),
    RandomRotate90(p=1.0),
    RandomCrop(height=200, width=300, p=1.0),
    CenterCrop(height=200, width=300, p=1.0),
    Crop(x_min=100, y_min=100, x_max=400, y_max=400, p=1.0),
    Rotate(limit=(-100, -50), p=1.0),
    VerticalFlip(COCOUtils.trans_target_idx_map, p=1.0),
    HorizontalFlip(COCOUtils.trans_target_idx_map, p=1.0),
    Flip(COCOUtils.trans_target_idx_map, p=1.0),
    Transpose(COCOUtils.trans_target_idx_map, p=1.0),
]


for aug in augmenters:
    test_pipeline(coco_keypoints, 0, aug, show_anno=True, verbose=True)
