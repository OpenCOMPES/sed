"""Module contains a call tracker and a dataclass for saving the method calls."""
from dataclasses import dataclass
from functools import update_wrapper
from typing import List


@dataclass
class MethodCall:
    """A dataclass that records a method call with it's name,
    args and any additional kwargs."""

    name: str
    args: tuple
    kwargs: dict


class CallTracker:
    """A decorator that records each call with its arguments to the
    decorated function. The call_tracker is a class attribute and can
    be accessed by the class name, and holds the list of all calls."""

    call_tracker: List[MethodCall] = []

    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        self.__class__.call_tracker.append(
            MethodCall(self.func.__qualname__, args, kwargs),
        )
        return self.func(self, *args, **kwargs)
