"""This file contains code that performs several tests for the preprocessing functions
using pytest
"""
import pytest

import numpy as np
import pandas as pd


from sed.core.preprocessing import PreProcessingPipeline
from sed.core.preprocessing import PreProcessingStep
from sed.core.preprocessing import as_pre_processing

N_PTS = 100
cols = ["posx", "posy", "energy"]
df = pd.DataFrame(np.random.randn(N_PTS, len(cols)), columns=cols)

def square(df,col):
    df[f'col_squared'] = df[col] ** 2
    return df

def add(df,col,value):
    df[f'col_added'] = df[col] + value
    return df

def test_pp_decorator():
    """ test the PreProcessingStep decorator """
    def square(df,col):
        df[f'col_squared'] = df[col] ** 2
        return df
    square_dec = as_pre_processing(square(col='posx'))
    pp = square_dec(col='posx')
    assert isinstance(pp, PreProcessingStep)
    assert pp.func == square
    assert pp.kwargs == {'col': 'posx'}
    assert pp.args == ()
    assert pp.col == 'posx'

def test_pp_decorator_with_parameters():
    """ test the PreProcessingStep decorator """
    def add(df,col,value):
        df[f'col_added'] = df[col] + value
        return df
    add_dec = as_pre_processing(add(col='posx',value=1))
    pp = add_dec(col='posx',value=1)
    assert isinstance(pp, PreProcessingStep)
    assert pp.func == add
    assert pp.kwargs == {'col': 'posx', 'value': 1}
    assert pp.args == ()
    assert pp.col == 'posx'


def test_pp_decorator_error_missing_parameters():
    """ test that the decorator rises an error when missing parameters """
    def add(df,col,value):
        df[f'col_added'] = df[col] + value
        return df
    add_dec = as_pre_processing(add(col='posx'))
    pp = add_dec(col='posx',value=1)
    assert isinstance(pp, PreProcessingStep)
    with pytest.raises(TypeError):
        pp(df)


def test_pp_decorator_error_wrong_parameters():
    """ test that the decorator rises an error when wrong parameters """
    def add(df,col,value):
        df[f'col_added'] = df[col] + value
        return df
    add_dec = as_pre_processing(add(col='posx',value='asd'))
    pp = add_dec(col='posx',value=1)
    assert isinstance(pp, PreProcessingStep)
    with pytest.raises(TypeError):
        pp(df)


def test_preprocessing_pipeline():
    """Test the PreProcessingStep class"""
    def square(df):
        return df ** 2

    def add(df, value):
        return df + value

    square_step = PreProcessingStep(square)
    add_step = PreProcessingStep(add, args=(1,))
    pipeline = square_step + add_step
    assert isinstance(pipeline, PreProcessingPipeline)
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.steps[0], PreProcessingStep)
    assert isinstance(pipeline.steps[1], PreProcessingStep)
    assert pipeline.steps[0].func == square
    assert pipeline.steps[1].func == add
    assert pipeline.steps[1].args == (1,)
eeeeeeeeeeeee