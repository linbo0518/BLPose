from .openpose import *
from .hourglass import *


def get_estimator(name, *args, **kwargs):
    name = name.lower()
    if name == "openpose_v1":
        return OpenPoseV1(*args, **kwargs)
    elif name == "openpose_v2":
        return OpenPoseV2(*args, **kwargs)
    elif name in ("hourglass", "stacked_hourglass"):
        return OpenPoseV1(*args, **kwargs)
    else:
        raise NotImplementedError(f"'{name}' has not been implemented yet")
