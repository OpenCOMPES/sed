import numpy as np
import pandas as pd

from sed.core.dfops import apply_jitter

n_pts = 100
cols = ["posx", "posy", "energy"]
df = pd.DataFrame(np.random.randn(n_pts, len(cols)), columns=cols)


def test_apply_jitter():
    cols_jittered = [col + "_jittered" for col in cols]
    df_jittered = apply_jitter(df, cols=cols, cols_jittered=cols_jittered)
    assert isinstance(df_jittered, pd.DataFrame)
    for col in cols_jittered:
        assert col in df_jittered.columns
