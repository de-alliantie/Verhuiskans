"""Collection of utility functions.

Example applications:

@log_step(step_name='calculations', method=print)
@timeit(units='ms', method=print)
def complex_add(a, b):
    time.sleep(1)
    return a+b

x = complex_add(1, 2)
"""
import datetime
import logging
import os
import time
from functools import wraps
from itertools import chain
from typing import Any, Callable, List, Optional

import numpy as np


def get_env_var(name: str, raise_if_empty: bool = True) -> Optional[str]:
    """Get the value of an env var and raise error if it isn't set.

    Args:
        name (str): name of the environment variable
        raise_if_empty (bool): if True, variable is mandatory

    Raises:
        ValueError: if it has no value

    Returns:
        Optional[str]: [description]
    """
    value = os.environ.get(name)
    if not value and raise_if_empty:
        raise ValueError(f"Environment variable {name} is not set but it is mandatory")
    return value


def flatten(list_of_lists: List[List]) -> List:
    """Flatten list of lists into single list.

    Args:
        list_of_lists (List[List]): [description]

    Returns:
        List: [description]
    """
    return list(chain.from_iterable(list_of_lists))


def shape_inspect(x: Any, level: int = 0) -> None:
    """Useful for inspecting complex objects.

    Args:
        x (Any): Structure of list
        level (int, optional): Needn't be set. Only for recursive calls
    """
    indent = "  " * level
    if isinstance(x, np.ndarray):
        print(f"{indent}np.array{x.shape}")
    elif isinstance(x, list):
        print(f"{indent}list({len(x)})")
        for child in x:
            shape_inspect(child, level + 1)
    elif isinstance(x, tuple):
        print(f"{indent}tuple({len(x)})")
        for child in x:
            shape_inspect(child, level + 1)
    else:
        pass


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Just to get a timestamp string that you can use for tagging filenames (for models e.g.)
    Args:
        fmt: (string) datetime format

    Returns:
        (str) timestamp string
    """
    return datetime.datetime.now().strftime(fmt)


def log_step(step_name: str, method: Callable = logging.info) -> Callable:
    """Decorator for logging start and finish statements before and after executing a function.

    Args:
        step_name: (string) Text describing the step
        method: (callable) [print, logging.info, logging.debug]

    Returns:
        decorated function that prints or logs start/finish messages
    """

    def log_step_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            method(f"[ STARTING STEP: {step_name} ]")
            res = func(*args, **kwargs)
            method(f"[ FINISHED STEP: {step_name} ]")
            return res

        return wrapper

    return log_step_decorator


def timeit(units: str = "s", method: Callable = logging.info) -> Callable:
    """Decorator for timing functions (that run for a significant while)

    Args:
        units: (string) [s, ms]
        method: (callable) [print, logging.info, logging.debug]

    Returns:
        decorated function that prints or logs execution time
    """

    def timeit_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            t1 = time.time()
            if units == "ms":
                diff = f"{(t1 - t0) * 1000:.0f} ms"
            else:
                diff = f"{(t1 - t0):.0f} seconds"
            msg = f"Call to {func.__name__} took {diff}"
            method(msg)
            return res

        return wrapper

    return timeit_decorator


class Timer:
    """
    Another way of timing: not by decorating functions, but by adding lines at specific moments

    Usage:
    - initialize time once: timer = Timer()
    - at any point: timer('some step', method=logging.info)
    - if you do: timer('some step', update=True, method=logging.info) the timer will be reset
       so your next call to timer() will have counted from this point on
    """

    def __init__(self):
        """Initializes the time function."""
        self.time = time.time()

    def __call__(self, msg: str = None, update: bool = True, method: Callable = print) -> None:
        """Logs the time elapsed since the last checkpoint.

        Parameters:
            msg (str): Optional message to identify the checkpoint.
            update (bool): If True, updates the internal timer to the current time.
            method (Callable): Function used for logging the elapsed time and checkpoint message.

        Returns:
            None
        """
        new_time = time.time()
        diff = new_time - self.time

        # update last timestamp
        if update:
            self.time = new_time

        if diff < 1.0:
            duration = f"{diff * 1000:.0f} ms"
        else:
            duration = f"{diff:.1f} seconds"

        method(f"Time elapsed: {duration} at checkpoint {msg}")
