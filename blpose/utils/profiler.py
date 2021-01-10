"""BLPose Utils: Profiler

Author: Bo Lin (@linbo0518)
Date: 2020-10-15
"""

from time import time
from contextlib import contextmanager


def profiling(description=None, n_divider=30):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            divider = "#" * n_divider
            print(f"{divider} {description} {divider}")
            tic = time()
            res = func(*args, **kwargs)
            exec_time = round(time() - tic, ndigits=7)
            print(f"'{description}' executed for {exec_time} s")
            return res

        return inner_wrapper

    return wrapper


class Profiler:
    def __init__(self, unit="s", digits=7):
        opt_unit = ("s", "ms", "µs", "ns")
        unit = unit.lower()
        if unit not in opt_unit:
            raise ValueError(f"'unit' should be one of {opt_unit}")

        self._pow = 1e3 ** opt_unit.index(unit)
        self._unit = unit
        self._digits = digits

    @contextmanager
    def profiling(self, description=None, n_divider=30):
        # enter
        divider = "#" * n_divider
        print(f"{divider} {description} {divider}")
        tic = time()

        yield self

        # exit
        exec_time = round((time() - tic) * self._pow, ndigits=self._digits)
        print(f"'{description}' executed for {exec_time} {self._unit}")
