"""BLPose Test: Pose Machine

Author: Bo Lin (@linbo0518)
Date: 2020-12-15
"""

import sys
import cv2
import matplotlib.pyplot as plt

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.estimators import OpenPoseV1
from blpose import PoseMachine

p = Profiler()
pose_machine = PoseMachine(OpenPoseV1(19, 18), "../models/openpose_v1_coco.pt", "cpu")

with p.profiling("Test Pose Machine"):
    x = cv2.imread("assets/ski.jpg")
    candidate, bodies = pose_machine.image_predict(x)
    canvas = pose_machine.draw_pose(x, candidate, bodies)

print(candidate)
print(bodies)
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.show()
