# -*- coding: utf-8 -*-
"""This module is used to provide various helpers
such as an imputer to deal with missing values"""

# Importing libraries
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from keras.models import Model
from keras.models import load_model

import utils.cryptocurrency.cryptocompare as cc
import utils.neuralnet.model as md
import utils.technicalanalysis.indicators as ind


def split(X, testcol="close", limit=32):
    data_train = X.iloc[:len(X) - limit, :]
    y_test = X.iloc[len(X) - limit:, X.columns.get_loc("close")]

    return data_train, y_test


def get_data(main="ETH", secondary="BTC"):
    # TODO: Change hardcoded parameters (ie. BTC/ETH)
    print("Getting data...")

    historical_main = cc.get_historical_data(fsym=main)
    historical_main = impute_ts(historical_main)

    historical_secondary = cc.get_historical_data(fsym=secondary)
    historical_secondary = impute_ts(historical_secondary)

    social = cc.get_social_data()

    print("Computing indicators...")
    print("Computing Ichimoku Kinko Hyo...")
    ichimoku = ind.ichimoku(historical_main, shift=0)

    print("Computing MACD...")
    macd = ind.MACD(historical_main)

    print("Computing bollinger bands...")
    bbands = ind.bollinger(historical_main)

    print("Computing fourier transformations at 3, 6, 9 and 100 components...")
    fourier = ind.fourier(historical_main)

    print("Computing stochastic rsi...")
    s_rsi = ind.stochastic_rsi(historical_main)

    print("Computing ADX...")
    adx = ind.ADX(historical_main)

    historical_secondary.columns = [
        s + "_{}".format(secondary.lower()) if s != "time" else s
        for s in historical_secondary.columns
    ]

    data = pd.merge(historical_main, historical_secondary, on="time").merge(
        ichimoku,
        on="time").merge(macd, on="time").merge(bbands, on="time").merge(
            fourier, on="time").merge(s_rsi, on="time").merge(adx, on="time")

    print("Merging data on seconds from epoch...")
    data = merge_truncate(data, social)

    return data


def features_selection(data, threshold=.01, plot=True):

    def get_feature_importance_data(data_income):
        data = data_income.copy()
        y = data['close']
        X = data.iloc[:, 1:]

        train_samples = int(X.shape[0] * 0.65)

        X_train = X.iloc[:train_samples]
        X_test = X.iloc[train_samples:]

        y_train = y.iloc[:train_samples]
        y_test = y.iloc[train_samples:]

        return (X_train, y_train), (X_test, y_test)

    (X_train_FI, y_train_FI), (X_test_FI,
                               y_test_FI) = get_feature_importance_data(data)

    regressor = xgb.XGBRegressor(gamma=0.0,
                                 n_estimators=150,
                                 base_score=0.7,
                                 colsample_bytree=1,
                                 learning_rate=0.05)

    xgbModel = regressor.fit(X_train_FI,y_train_FI, \
                            eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
                            verbose=False)

    features = [i for i in range(len(xgbModel.feature_importances_))]
    labels = X_test_FI.columns
    values = np.array(xgbModel.feature_importances_.tolist())

    selection = labels[values > threshold]

    if plot:
        fig = plt.figure(figsize=(8, 8))
        plt.xticks(rotation='vertical')
        plt.bar(features, values, tick_label=labels)
        plt.title('Figure 6: Feature importance of the technical indicators.')
        plt.save("features.png")
        plt.show()

    return [data.columns.get_loc(col) for col in selection]


def impute_ts(X):
    X = X.replace(0, np.nan)
    X = X.interpolate()

    return X


def merge_truncate(historical, social):
    #! TODO: implement a quick sanity check
    first_nonzero = next((i for i, x in enumerate(social.values[:, 2]) if x),
                         None)
    social = social.iloc[first_nonzero:, :]
    social = impute_ts(social)

    data = pd.merge(historical, social, on="time").reset_index(drop=True)

    return data


def scale(X,
          scaler=MinMaxScaler(feature_range=(0, 1)),
          save=True,
          filename="MinMaxScaler"):
    sc = scaler
    X_scaled = sc.fit_transform(X)

    if save:
        if not os.path.exists("scalers/"):
            os.mkdir("scalers/")

        joblib.dump(sc, "scalers/{}.pkl".format(filename))

    return X_scaled


def reg(y, return_values=False):
    ind = np.array([x for x in range(32)]).reshape(-1, 1)

    lin_reg = LinearRegression()
    lin_reg.fit(ind, y)

    if return_values:
        y_pred = lin_reg.predict(ind)
        return lin_reg.coef_, y_pred

    else:
        return lin_reg.coef_


def preprocessing_pipeline(X,
                           n_past,
                           n_future,
                           is_testing_set=False,
                           low_trigger=.7,
                           high_trigger=2):

    columns_to_drop = []
    for col in X.columns:
        if X[col].isnull().all():
            columns_to_drop.append(col)
    preprocessed = X.drop(columns=columns_to_drop)

    preprocessed = preprocessed.astype(float)
    preprocessed = preprocessed.values

    close = preprocessed[:, 0]
    preprocessed = preprocessed[:, 1:]

    if is_testing_set:
        #! to update

        #stdsc = joblib.load("scalers/StandardScaler.pkl")
        #pca = joblib.load("scalers/pca.pkl")
        mmsc_in = joblib.load("scalers/MinMaxScaler_encoder.pkl")
        mmsc_out = joblib.load("scalers/MinMaxScaler.pkl")
        mmsc_pred = joblib.load("scalers/MinMaxScaler_predict.pkl")
        m = load_model("models/autoencoder.h5")
        encoder = Model(m.input, m.get_layer('bottleneck').output)

        preprocessed = np.array(preprocessed)

        preprocessed = mmsc_in.transform(preprocessed)
        encoded = encoder.predict(preprocessed)
        encoded = mmsc_out.transform(encoded)

        #preprocessed = stdsc.transform(preprocessed)
        #preprocessed = pca.transform(preprocessed)
        #preprocessed = mmsc.transform(preprocessed)

        close = mmsc_pred.transform(close.reshape(-1, 1))
        preprocessed = np.concatenate([close, encoded], axis=1)

        X_test = np.array([preprocessed])

        return X_test

    else:
        preprocessed = scale(preprocessed,
                             scaler=MinMaxScaler(),
                             save=True,
                             filename="MinMaxScaler_encoder")

        #pca = PCA(.99)
        #preprocessed = pca.fit_transform(preprocessed)
        #joblib.dump(pca, "scalers/pca.pkl")
        preprocessed = np.array(preprocessed)
        m = md.build_autoencoder(preprocessed.shape[1], optimizer="adam")

        m.fit(preprocessed, preprocessed, batch_size=128, epochs=32, verbose=1)
        m.save("models/autoencoder.h5")

        encoder = Model(m.input, m.get_layer('bottleneck').output)
        encoder.save("models/encoder.h5")

        encoded = encoder.predict(preprocessed)
        encoded = scale(encoded)

        close = scale(close.reshape(-1, 1), filename="MinMaxScaler_predict")

        #preprocessed = scale(preprocessed)
        preprocessed = np.concatenate((close, encoded), axis=1)

        X_train = [
            preprocessed[i - n_past:i, :]
            for i in range(n_past,
                           len(preprocessed) - n_future)
        ]
        y_train = [
            preprocessed[i:i + n_future, 0]
            for i in range(n_past,
                           len(preprocessed) - n_future)
        ]

        print("Computing coefficients for y_train...")
        y_train = [reg(y) for y in y_train]

        σ = np.std(y_train)
        mean = np.mean(y_train)
        u = mean + low_trigger * σ
        l = mean - low_trigger * σ
        u2 = mean + high_trigger * σ
        l2 = mean - high_trigger * σ

        # 0 for ultralow, 1 for low, 2 for range, 3 for high, 4 for ultrahigh
        y_train = [
            0 if y <= l2 else 1 if y <= l and y > l2 else
            3 if y >= u and y < u2 else 4 if y >= u2 else 2 for y in y_train
        ]

        X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)

        # Range : (mean-std, mean+std)
        return X_train, y_train


def get_category(category):
    label = ""

    if category == 0:
        label = "Very Downward Trend"

    elif category == 1:
        label = "Downward Trend"

    elif category == 2:
        label = "Range"

    elif category == 3:
        label = "Upward Trend"

    elif category == 4:
        label = "Very Upward Trend"

    return label
