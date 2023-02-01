"""Module contains a call tracker and a dataclass for saving the method calls."""
from dataclasses import dataclass
from functools import wraps


@dataclass
class MethodCall:
    """A dataclass that records a method call with it's name, args and any
    additional kwargs."""

    name: str
    args: list
    kwargs: dict


def track_call(func):
    """A decorator that records each call to the decorated function."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.call_tracker.append(
            MethodCall(
                type(self).__name__ + "." + func.__name__,
                args,
                kwargs,
            ),
        )

        return func(self, *args, **kwargs)

    return wrapper
