"""BLPose Train: Data Utils

Author: Bo Lin (@linbo0518)
Date: 2021-01-02
"""
"""BLPose Data Format:
type: List[List[int]]
shape: (n_keypoints, 3)
format:
    [[x1, y1, v1],
     [x2, y2, v2],
     ...
     [xk, yk, vk]]

    x: the x coordinate of the keypoint, dtype: int
    y: the y coordinate of the keypoint, dtype: int
    v: visibility of keypoint, dtype: int
        0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible
example: blpose_keypoints =
    [[407, 115, 1],
     [407, 105, 2],
     [0, 0, 0],
     [425, 95, 2],
     [0, 0, 0],
     [435, 124, 2],
     [457, 105, 2],
     [428, 187, 2],
     [447, 182, 2],
     [404, 210, 2],
     [419, 213, 2],
     [488, 222, 2],
     [515, 213, 2],
     [471, 293, 2],
     [487, 297, 2],
     [462, 372, 1],
     [486, 374, 2]]
"""


class COCOUtils:
    anno_idx_to_kpt = (
        "nose",
        "l eye",
        "r eye",
        "l ear",
        "r ear",
        "l shoulder",
        "r shoulder",
        "l elbow",
        "r elbow",
        "l wrist",
        "r wrist",
        "l hip",
        "r hip",
        "l knee",
        "r knee",
        "l ankle",
        "r ankle",
    )
    target_idx_to_kpt = (
        "nose",
        "neck",
        "r shoulder",
        "r elbow",
        "r wrist",
        "l shoulder",
        "l elbow",
        "l wrist",
        "r hip",
        "r knee",
        "r ankle",
        "l hip",
        "l knee",
        "l ankle",
        "r eye",
        "l eye",
        "r ear",
        "l ear",
    )
    heat_idx_to_kpt = target_idx_to_kpt + ("background",)
    # fmt: off
    anno_idx_to_target = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
    target_idx_to_anno = (0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)
    trans_anno_idx_map = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
    trans_target_idx_map = (0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16)
    target_limbs_idx = (
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16),
        (5, 17)
    )
    # fmt: on
    n_target_keypoints = len(target_idx_to_kpt)
    n_target_limbs = len(target_limbs_idx)

    @classmethod
    def to_blpose(cls, coco_keypoints):
        index_map = cls.anno_idx_to_target
        blpose_keypoints = [[0, 0, 0] for _ in range(18)]

        for idx in range(len(index_map)):
            for i in range(3):
                blpose_keypoints[index_map[idx]][i] = coco_keypoints[idx * 3 + i]
        if coco_keypoints[5 * 3 + 2] and coco_keypoints[6 * 3 + 2]:
            for i in range(3):
                blpose_keypoints[1][i] = round(
                    (coco_keypoints[5 * 3 + i] + coco_keypoints[6 * 3 + i]) / 2
                )
        return blpose_keypoints

    @classmethod
    def from_blpose(cls, blpose_keypoints):
        index_map = cls.target_idx_to_anno
        coco_keypoints = [0 for _ in range(17 * 3)]

        for idx in range(len(index_map)):
            if idx == -1:
                continue
            for i in range(3):
                coco_keypoints[index_map[idx] * 3 + i] = blpose_keypoints[idx][i]
        return coco_keypoints


class MPIIUtils:
    anno_idx_to_kpt = (
        "r ankle",
        "r knee",
        "r hip",
        "l hip",
        "l knee",
        "l ankle",
        "pelvis",
        "thorax",
        "upper neck",
        "head top",
        "r wrist",
        "r elbow",
        "r shoulder",
        "l shoulder",
        "l elbow",
        "l wrist",
    )
    target_idx_to_kpt = (
        "head top",
        "upper neck",
        "r shoulder",
        "r elbow",
        "r wrist",
        "l shoulder",
        "l elbow",
        "l wrist",
        "r hip",
        "r knee",
        "r ankle",
        "l hip",
        "l knee",
        "l ankle",
        "body center",
    )
    heat_idx_to_kpt = target_idx_to_kpt + ("background",)

    # fmt: off
    anno_idx_to_target = (10, 9, 8, 11, 12, 13, -1, -1, 1, 0, 4, 3, 2, 5, 6, 7)
    target_idx_to_anno = (9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, -1)
    trans_idx_map = (0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14)
    limbs_target_idx = (
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 9), (9, 10),
        (11, 12), (12, 13), (1, 14), (8, 14), (11, 14)
    )
    # fmt: on
