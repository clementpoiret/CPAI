# -*- coding: utf-8 -*-
"""This module is used to build our regressor.
The model is currently at a very early stage and
is subject to a lot of changes"""

# Importing libraries
import os

from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, CuDNNGRU, Dropout, Activation, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def build_regressor(n_past, n_features, optimizer="rmsprop"):
    regressor = Sequential()

    regressor.add(
        CuDNNGRU(units=256,
                 return_sequences=True,
                 input_shape=(n_past, n_features)))
    regressor.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    regressor.add(CuDNNGRU(units=128, return_sequences=True))
    regressor.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    regressor.add(CuDNNGRU(units=64, return_sequences=True))
    regressor.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    regressor.add(CuDNNGRU(units=32))
    regressor.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    regressor.add(Dense(units=32, activation="sigmoid"))

    regressor.compile(optimizer=optimizer, loss="mean_squared_error")

    plot_model(regressor, to_file="model.png")

    return regressor


def train_model(X_train,
                y_train,
                n_past,
                optimizer="rmsprop",
                batch_size=64,
                validation_split=.2,
                epochs=128):
    regressor = build_regressor(n_past, X_train.shape[2], optimizer=optimizer)

    if not os.path.exists("models/"):
        os.mkdir("models/")

    es = EarlyStopping(monitor='val_loss',
                       min_delta=1e-10,
                       patience=10,
                       verbose=1)

    rlr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5,
                            patience=5,
                            verbose=1)

    mcp = ModelCheckpoint(filepath='models/weights.h5',
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')

    regressor.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=[es, rlr, mcp, tb],
                  verbose=1,
                  validation_split=validation_split,
                  batch_size=batch_size)

    return regressor


def tune(X_train, y_train, parameters, cv=4, n_jobs=-1):
    estimator = KerasRegressor(build_fn=build_regressor)

    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=parameters,
                               scoring="accuracy",
                               cv=cv,
                               n_jobs=n_jobs)

    grid_search = grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_parameters, best_accuracy