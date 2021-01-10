"""BLPose Test: Train Dataset

Author: Bo Lin (@linbo0518)
Date: 2021-01-03
"""

import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from blpose.train.dataset import COCOKeypoints, COCOUtils
from blpose.utils.profiler import Profiler

p = Profiler()

coco_dir = "/Users/linbo0518/Movies/COCO2017"
coco_keypoints = COCOKeypoints(coco_dir)

with p.profiling("Test COCO Keypoints API"):
    image, keypoints_all, mask_all, mask_loss, anno = coco_keypoints[0]

plt.imshow(image[:, :, ::-1])
plt.axis("off")
plt.show()

plt.imshow(mask_loss, cmap="hot")
plt.axis("off")
plt.show()

offset = 0
for idx, keypoints in enumerate(keypoints_all):
    print(f"person: {idx}")
    print(f"coco to blpose: {keypoints}")
    keypoints = COCOUtils.from_blpose(keypoints)
    print(f"blpose to coco: {keypoints}")
    while anno[idx + offset]["num_keypoints"] == 0 or anno[idx + offset]["iscrowd"]:
        offset += 1
    print(f' original coco: {anno[idx + offset]["keypoints"]}')
