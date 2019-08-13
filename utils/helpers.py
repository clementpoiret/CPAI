# -*- coding: utf-8 -*-
"""This module is used to provide various helpers
such as an imputer to deal with missing values"""

# Importing libraries
import numpy as np
import pandas as pd

from impyute.imputation.ts import moving_window
from sklearn.preprocessing import MinMaxScaler


def impute_ts(X):
    X = X.replace(0, np.nan)
    X = X.interpolate()

    return X


def merge_truncate(historical, social):
    #! TODO: implement a quick sanity check
    first_nonzero = next((i for i, x in enumerate(social.values[:, 2]) if x),
                         None)
    social = social.iloc[first_nonzero:, :]

    data = pd.merge(historical, social, on="time").reset_index(drop=True)

    return data


def scale(X):
    sc = MinMaxScaler(feature_range=(0, 1))
    X_scaled = sc.fit_transform(X)

    return X_scaled


def preprocessing_pipeline(X, n_past, n_future):
    columns_to_drop = []
    for col in X.columns:
        if (X[col] == 0).all():
            columns_to_drop.append(col)
    preprocessed = X.drop(columns=columns_to_drop)

    for col in preprocessed.columns:
        if (preprocessed[col] == 0).any():
            preprocessed[col] = impute_ts(X[col])

    preprocessed = preprocessed.astype(float)
    preprocessed = preprocessed.values

    preprocessed = scale(preprocessed)

    X_train = [
        preprocessed[i - n_past:i, :]
        for i in range(n_past,
                       len(preprocessed) - n_future + 1)
    ]
    y_train = [
        preprocessed[i:i + n_future + 1, 0]
        for i in range(n_past,
                       len(preprocessed) - n_future + 1)
    ]

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train