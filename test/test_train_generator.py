"""BLPose Test: Train Generator

Author: Bo Lin (@linbo0518)
Date: 2021-01-07
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from blpose.train.generator import TargetGenerator
from blpose.train.dataset import COCOKeypoints, COCOUtils

coco_dir = "/Users/linbo0518/Movies/COCO2017"
coco_keypoints = COCOKeypoints(coco_dir)
image, keypoints_all, mask_all, mask_loss, anno = coco_keypoints[0]

target_generator = TargetGenerator(
    COCOUtils.n_target_keypoints,
    COCOUtils.target_limbs_idx,
    image_shape=image.shape,
    stride=8,
)

plt.imshow(image[:, :, ::-1])
plt.axis("off")
plt.show()

heatmap, pafmap = target_generator(keypoints_all)

plt.imshow(np.sum(heatmap[:-1], axis=0), cmap="hot")
plt.axis("off")
plt.show()

plt.imshow(np.sum(pafmap, axis=0), cmap="hot")
plt.axis("off")
plt.show()
