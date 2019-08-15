# -*- coding: utf-8 -*-
"""This module is used to provide various helpers
such as an imputer to deal with missing values"""

# Importing libraries
import numpy as np
import pandas as pd
import utils.cryptocurrency.cryptocompare as cc
import utils.technicalanalysis.indicators as ind

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


def get_data():
    # TODO: Change hardcoded parameters (ie. BTC/ETH)
    print("Getting data...")

    historical_btc = cc.get_historical_data(fsym="BTC")
    historical_eth = cc.get_historical_data(fsym="ETH")
    social = cc.get_social_data()

    print("Computing indicators...")
    print("Computing Ichimoku Kinko Hyo...")
    ichimoku = ind.ichimoku(historical_eth, shift=0)

    print("Computing MACD...")
    macd = ind.MACD(historical_eth)

    print("Computing bollinger bands...")
    bbands = ind.bollinger(historical_eth)

    print("Computing fourier transformations at 3, 6, 9 and 100 components...")
    fourier = ind.fourier(historical_eth)

    print("Computing stochastic rsi...")
    s_rsi = ind.stochastic_rsi(historical_eth)

    print("Computing ADX...")
    adx = ind.ADX(historical_eth)

    historical_btc.columns = [
        s + "_btc" if s != "time" else s for s in historical_btc.columns
    ]

    data = pd.merge(historical_eth, historical_btc, on="time").merge(
        ichimoku,
        on="time").merge(macd, on="time").merge(bbands, on="time").merge(
            fourier, on="time").merge(s_rsi, on="time").merge(adx, on="time")

    print("Merging data on seconds from epoch...")
    data = merge_truncate(data, social)

    return data


def split(X, testcol="close", limit=32):
    data_train = X.iloc[:len(X) - limit, :]
    y_test = X.iloc[len(X) - limit:, X.columns.get_loc("close")]

    return data_train, y_test


def compute_indicators(X):
    ind.ichimoku(X, shift=0)

    return indicators
