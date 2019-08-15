# -*- coding: utf-8 -*-
"""This module is used to build our regressor.
The model is currently at a very early stage and
is subject to a lot of changes"""

# Importing libraries
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.utils import plot_model


def build_regressor(n_past, n_features):
    regressor = Sequential()

    regressor.add(
        CuDNNLSTM(units=65,
                  return_sequences=True,
                  input_shape=(n_past, n_features)))
    regressor.add(Dropout(rate=.2))

    regressor.add(CuDNNLSTM(units=65, return_sequences=True))
    regressor.add(Dropout(rate=.2))

    regressor.add(CuDNNLSTM(units=65, return_sequences=True))
    regressor.add(Dropout(rate=.2))

    regressor.add(CuDNNLSTM(units=65, return_sequences=True))
    regressor.add(Dropout(rate=.2))

    regressor.add(Flatten())

    regressor.add(Dense(units=1, activation="linear"))

    regressor.compile(optimizer="rmsprop", loss="mean_squared_error")

    plot_model(regressor, to_file="model.png")

    return regressor