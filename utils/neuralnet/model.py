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
    regressor.add(Dropout(rate=.5))

    regressor.add(CuDNNLSTM(units=65, return_sequences=True))
    regressor.add(Dropout(rate=.5))

    regressor.add(CuDNNLSTM(units=65, return_sequences=True))
    regressor.add(Dropout(rate=.5))

    regressor.add(CuDNNLSTM(units=65))
    regressor.add(Dropout(rate=.5))

    regressor.add(Dense(units=32, activation="linear"))

    regressor.compile(optimizer="rmsprop", loss="mean_squared_error")

    plot_model(regressor, to_file="model.png")

    return regressor


def train_model(X_train,
                y_train,
                n_past,
                batch_size=64,
                validation_split=.2,
                epochs=128):
    regressor = build_regressor(n_past, X_train.shape[2])

    es = EarlyStopping(monitor='val_loss',
                       min_delta=1e-10,
                       patience=10,
                       verbose=1)

    rlr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5,
                            patience=5,
                            verbose=1)

    mcp = ModelCheckpoint(filepath='weights.h5',
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')

    history = regressor.fit(X_train,
                            y_train,
                            epochs=epochs,
                            callbacks=[es, rlr, mcp, tb],
                            verbose=1,
                            validation_split=validation_split,
                            batch_size=batch_size)

    return regressor, history