"""Module contains a call tracker and a dataclass for saving the method calls."""
from functools import wraps
from dataclasses import dataclass

@dataclass
class MethodCall:
    """A dataclass that records a method call with it's name, args and any additional kwargs."""
    name: str
    args: list
    kwargs: dict

def track_call(func):
    """A decorator that records each call to the decorated function."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._call_tracker.append(MethodCall(func.__name__, args, kwargs))

        return func(self, *args, **kwargs)

    return wrapper
