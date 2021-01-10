"""BLPose Train: Dataset

Author: Bo Lin (@linbo0518)
Date: 2021-01-01
"""
import os
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as _mask
from pycocotools.coco import COCO

from .dataset_utils import COCOUtils

__all__ = ["COCOKeypoints", "COCOUtils"]


class COCOKeypoints:
    def __init__(
        self,
        coco_dir,
        year: int = 2017,
        mode: str = "train",
        to_func=COCOUtils.to_blpose,
        from_func=COCOUtils.from_blpose,
    ):
        if not os.path.isdir(coco_dir):
            raise ValueError(f"{coco_dir} is not a valid dir for coco dataset")

        self.image_dir = os.path.join(coco_dir, f"{mode}{year}")
        annotation_file = os.path.join(
            coco_dir, f"annotations{os.sep}person_keypoints_{mode}{year}.json"
        )

        self.coco = COCO(annotation_file=annotation_file)
        self.image_ids = self.coco.getImgIds(
            catIds=self.coco.getCatIds(catNms="person")
        )
        self.to_func = to_func
        self.from_func = from_func

    def __repr__(self):
        self.coco.info()
        return ""

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        image = cv2.imread(os.path.join(self.image_dir, f"{image_id:012}.jpg"))
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        keypoints_all, mask_all, mask_loss = self._parse_annotations(image.shape, anno)
        return image, keypoints_all, mask_all, mask_loss, anno

    def generate_annotations(self, annotations, keypoints_all, mask_all, mask_loss):
        annotations = copy.deepcopy(annotations)
        new_annotations = []
        offset = 0
        # keypoints_all and mask_all
        for idx, (keypoints, mask) in enumerate(zip(keypoints_all, mask_all)):
            anno = annotations[idx + offset]
            while anno["num_keypoints"] == 0 or anno["iscrowd"]:
                offset += 1
                anno = annotations[idx + offset]

            anno["keypoints"] = self.from_func(keypoints)

            rle = _mask.encode(np.asfortranarray(mask))
            anno["segmentation"] = rle
            anno["area"] = _mask.area(rle)
            anno["bbox"] = _mask.toBbox(rle).tolist()
            new_annotations.append(anno)
        # mask_loss
        rle = _mask.encode(np.asfortranarray(1 - mask_loss))
        new_annotations.append(
            {
                "segmentation": rle,
                "num_keypoints": 0,
                "area": _mask.area(rle),
                "iscrowd": 1,
                "keypoints": [0 for _ in range(17 * 3)],
                "image_id": new_annotations[0]["image_id"],
                "bbox": _mask.toBbox(rle),
                "category_id": new_annotations[0]["category_id"],
                "id": new_annotations[0]["id"],
            }
        )
        return new_annotations

    def show_annotations(self, image, annotations):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.coco.showAnns(annotations)
        plt.axis("off")
        plt.show()

    def _parse_annotations(self, shape, annotations):
        h, w, _ = shape
        keypoints_all = []
        mask_all = []
        mask_loss = np.full((h, w), True)

        for anno in annotations:
            if anno["num_keypoints"] == 0 or anno["iscrowd"] == 1:
                mask_loss &= self.coco.annToMask(anno) == 0
            else:
                keypoints_all.append(self.to_func(anno["keypoints"]))
                mask_body = self.coco.annToMask(anno) != 0
                mask_all.append(mask_body.astype(np.uint8))
        mask_loss = mask_loss.astype(np.uint8)
        return keypoints_all, mask_all, mask_loss
