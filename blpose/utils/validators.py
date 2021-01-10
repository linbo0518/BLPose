"""BLPose Utils: Validators

Author: Bo Lin (@linbo0518)
Date: 2020-12-20
"""

__all__ = [
    "check_aliquot",
    "check_eq",
    "check_gt",
    "check_len",
    "check_oneof",
    "check_type",
]


def check_aliquot(dividend, divisor):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} should be an integer multiple of {divisor}")


def check_eq(obj, value):
    if obj != value:
        raise ValueError(f"obj should be equal to {value} but {obj} was given")


def check_gt(obj, value):
    if obj <= value:
        raise ValueError(f"obj should be greater than {value} but {obj} was given")


def check_len(obj, length):
    if len(obj) != length:
        raise ValueError(f"length of obj should be {length} not {len(obj)}")


def check_oneof(obj, values):
    if obj not in values:
        raise ValueError(f"obj should be one of {values} but {obj} was given")


def check_type(obj, types):
    if not isinstance(obj, types):
        raise TypeError(f"Unsupported type: {type(obj)}")
