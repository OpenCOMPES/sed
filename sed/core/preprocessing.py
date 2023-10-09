from __future__ import annotations

from typing import Sequence, Iterator
import inspect
from copy import deepcopy

import json
from dask import dataframe as ddf
import pandas as pd
from .metadata import MetaHandler


def as_pre_processing(func):
    """Decorator to create a PreProcessingStep from a function

    Args:
        func: the function to decorate

    Returns:
        a PreProcessingStep instance
    """
    signature = inspect.signature(func)  # get the signature of the function
    parameters = signature.parameters
    # create a dictionary with the default values of the parameters
    defaults = {
        param.name: param.default
        for param in parameters.values()
        if param.default is not inspect.Parameter.empty
    }
    def wrapper(*args, **kwargs):
        kwargs.update(defaults)
        return PreProcessingStep(func, *args, **kwargs)
    return wrapper


class PreProcessingStep:
    """PreProcessing step class which allows mapping a function to a dataframe

    Args:
        func: the function to map to the dataframe
        *args: the arguments of the function
        **kwargs: the keyword arguments of the function
    
    Attributes:
        func: the function to map to the dataframe
        args: the arguments of the function
        kwargs: the keyword arguments of the function
        df: the dataframe to apply the function to
        duplicate_policy: the policy to use when logging the function call to the metadata
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.df = kwargs.pop("df", None)
        self.kwargs = kwargs
        self.__doc__ = f"PreProcessing step class which allows mapping the " \
            f"function:\n\n{func.__name__} docstring:\n{func.__doc__}"
        self.duplicate_policy = kwargs.pop("duplicate_policy", "raise")
        self.name = kwargs.pop("name", func.__name__)

    def log_metadata(self, meta:MetaHandler) -> None:
        """ log the call of the function to the meta dictionary """
        meta.add(self.to_dict(), self.name, self.duplicate_policy)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the object"""
        return {"func": self.func, "args": self.args, "kwargs": self.kwargs}

    def to_json(self) -> str:
        """Return a json representation of the object"""
        return json.dumps(self.to_dict())

    def __call__(self, df=None, metadata:MetaHandler=None) -> ddf.DataFrame:
        """Apply the function to the dataframe by calling it directly"""
        df = self._get_df(df)
        if metadata is not None:
            self.log_metadata(metadata)
        return self.func(df=df, *self.args, **self.kwargs)

    def map(self, df=None, metadata:MetaHandler=None) -> ddf.DataFrame:
        """Apply the function to the dataframe using map_partitions"""
        df = self._get_df(df)
        if metadata is not None:
            self.log_metadata(metadata)
        # try:

            meta = self(df.head()).dtypes.to_dict()
            # Create the _meta information for Dask DataFrame
            # meta = 
        df = df.map_partitions(
            self.func, 
            *self.args, 
            **self.kwargs, 
            meta=meta)
        # except TypeError:
        #     out = df.map_partitions(self.func, *self.args, **self.kwargs)
        return df

    def _get_df(self, df=None) -> ddf.DataFrame:
        """Get the dataframe to apply the function to"""
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No dataframe provided")
        return df

    def __repr__(self) -> str:
        """Return the string representation of the object"""
        return f"{self.__class__.__name__}({self.func.__name__})"

    def __str__(self) -> str:
        """Return the string representation of the object"""
        s = f"{self.__class__.__name__}({self.func.__name__})"
        s += f"\n - args: {self.args}"
        s += f"\n - kwargs: {self.kwargs}"
        return s

    def _repr_html_(self) -> str:
        """Return the html representation of the object"""
        s = f"<h3>{self.__class__.__name__}({self.func.__name__})</h3>"
        s += f"<p>args: {self.args}</p>"
        s += f"<p>kwargs: {self.kwargs}</p>"
        return s

    def __eq__(self, __value: object) -> bool:
        """Check if two PreProcessingStep objects are equal"""
        return self.__dict__ == __value.__dict__

    def __hash__(self) -> int:
        """Return the hash of the object"""
        return hash(self.__dict__)

    def __copy__(self) -> PreProcessingStep:
        """Return a shallow copy of the object"""
        return PreProcessingStep(self.func, *self.args, **self.kwargs)

    def __deepcopy__(self) -> PreProcessingStep:
        """Return a deep copy of the object"""
        return PreProcessingStep(
            self.func, *deepcopy(self.args), **deepcopy(self.kwargs)
        )

    def __add__(self, other) -> PreProcessingPipeline:
        """Add two PreProcessingStep objects"""
        return PreProcessingPipeline([self, other])

    def __radd__(self, other) -> PreProcessingPipeline:
        """Add two PreProcessingStep objects"""
        return PreProcessingPipeline([other, self])


class PreProcessingPipeline:
    """ Allows sequentially mapping functions to a dataframe

    TODOs:  - add a method to test the pipeline by applying it to the head of the df
            - add a method to visualize the pipeline
            - add a method to visualize the pipeline as a graph
            - add a method to visualize the pipeline as a table

    Args:
        steps: the steps of the pipeline
    """

    def __init__(self, steps: Sequence[PreProcessingStep]) -> None:
        if isinstance(steps, Sequence):
            self.steps = steps
        else:
            self.steps = [steps]

    @classmethod
    def from_dict(cls, d: dict) -> PreProcessingPipeline:
        """Create a PreProcessingPipeline object from a dictionary"""
        steps = [PreProcessingStep(**step) for step in d["pre_processing"]]
        return cls(steps)

    @classmethod
    def from_json(cls, s: str) -> PreProcessingPipeline:
        """Create a PreProcessingPipeline object from a json string"""
        d = json.loads(s)
        return cls.from_dict(d)

    def test(self, df=None):
        """Test the pipeline by applying it to the head of the dataframe

        Args:
            df: the dataframe to apply the pipeline to

        Returns:
            a dictionary with the report of the test
        """
        df = df or self.df
        dfh = df.head()
        report = {}
        for step in self.steps:
            try:
                df = step(df)
                report[step.func.__name__] = "passed"
            except Exception as e:
                report[step.func.__name__] = e
        return report

    def to_dict(self) -> dict:
        """Return a dictionary representation of the object"""
        return {"pre_processing": [step.to_dict() for step in self.steps]}

    def to_json(self) -> str:
        """Return a json representation of the object"""
        return json.dumps(self.to_dict())

    def append(self, step: PreProcessingStep) -> None:
        """Append a step to the pipeline"""
        self.steps.append(step)

    def insert(self, index: int, step: PreProcessingStep) -> None:
        """Insert a step at a given index"""
        self.steps.insert(index, step)

    def remove(self, step: PreProcessingStep) -> None:
        """Remove a step from the pipeline"""
        self.steps.remove(step)

    def pop(self, index: int) -> PreProcessingStep:
        """Pop a step at a given index"""
        return self.steps.pop(index)

    def __call__(self, df: ddf.DataFrame, metadata=None) -> ddf.DataFrame:
        """Apply the function to the dataframe by calling it directly
        
        This is not the recommended way to apply the pipeline to a dataframe.
        Use the `map` method instead.
        
        Args:
            df: the dataframe to apply the pipeline to
            meta: the metadata to use for the dataframe
        
        Returns:
            the dataframe with the function applied
        """
        df = df or self.df
        for step in self.steps:
            df = step(df,metadata=metadata)
        return df

    def map(self, df: ddf.DataFrame=None, metadata: pd.DataFrame = None) -> ddf.DataFrame:
        """Apply the pipeline to the dataframe
        
        Args:
            df: the dataframe to apply the pipeline to
            metadata: the metadata handler where to log usage of the pipeline
        
        Returns:
            the dataframe with the pipeline applied
        """
        df = df if df is not None else self.df
        for step in self.iterate():
            df = step.map(df, metadata)
        return df

    def iterate(self) -> Iterator[PreProcessingStep]:
        """Return an iterator over the steps of the pipeline"""

        return (s for s in self.steps)
    
    def _get_df(self, df=None) -> ddf.DataFrame:
        """Get the dataframe to apply the function to"""
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No dataframe provided")
        return df

    def __repr__(self) -> str:
        """Return the string representation of the object"""
        return f"{self.__class__.__name__}({self.steps})"

    def __str__(self) -> str:
        """Return the string representation of the object"""
        s = f"{self.__class__.__name__}({self.steps})"
        return s

    def _repr_html_(self) -> str:
        """Return the html representation of the object"""
        s = f"<h3>{self.__class__.__name__}</h3>"
        s += "<ul>"
        for step in self.steps:
            s += f"<li>{step}</li>"
        s += "</ul>"
        return s

    def __add__(self, other) -> PreProcessingPipeline:
        """Add two PreProcessingPipeline objects"""
        if isinstance(other, PreProcessingStep):
            return PreProcessingPipeline([*self.steps, other])
        elif isinstance(other, PreProcessingPipeline):
            return PreProcessingPipeline([*self.steps, *other.steps])
        else:
            raise TypeError(
                f"Cannot add {self.__class__.__name__} and {other.__class__.__name__}"
            )

    def __radd__(self, other) -> PreProcessingPipeline:
        """Add two PreProcessingPipeline objects"""
        if isinstance(other, PreProcessingStep):
            return PreProcessingPipeline([other, *self.steps])
        elif isinstance(other, PreProcessingPipeline):
            return PreProcessingPipeline([*other.steps, *self.steps])
        else:
            raise TypeError(
                f"Cannot add {self.__class__.__name__} and {other.__class__.__name__}"
            )

    def __iter__(self) -> Iterator[PreProcessingStep]:
        """Return an iterator over the steps of the pipeline"""
        return self.iterate()
    
    def __next__(self) -> PreProcessingStep:
        """Return the next step of the pipeline"""
        return next(self.iterate())