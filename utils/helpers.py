# -*- coding: utf-8 -*-
"""This module is used to provide various helpers
such as an imputer to deal with missing values"""

# Importing libraries
import numpy as np
import pandas as pd

from impyute.imputation.ts import moving_window


def impute_ts(X):
    X = X.interpolate()

    return X


def merge_truncate(historical, social):
    #! TODO: implement a quick sanity check
    first_nonzero = next((i for i, x in enumerate(social.values[:, 2]) if x),
                         None)
    social = social.iloc[first_nonzero:, :]

    data = pd.merge(historical, social, on="time").reset_index()

    return data