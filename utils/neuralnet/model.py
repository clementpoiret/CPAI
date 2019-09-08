# -*- coding: utf-8 -*-
"""This module is used to build our regressor.
The model is currently at a very early stage and
is subject to a lot of changes"""

# Importing libraries
import os

import numpy as np
import tensorflow as tf

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Activation, CuDNNGRU, CuDNNLSTM, Dense, Dropout, PReLU
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.generic_utils import get_custom_objects
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def build_classifier(n_past, n_features, optimizer="rmsprop"):
    classifier = Sequential()

    #get_custom_objects().update({"gelu": Activation(gelu)})

    classifier.add(
        CuDNNGRU(units=128,
                 return_sequences=True,
                 input_shape=(n_past, n_features)))
    #regressor.add(Activation(gelu))
    classifier.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    classifier.add(CuDNNGRU(units=64, return_sequences=True))
    #regressor.add(Activation(gelu))
    classifier.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    classifier.add(CuDNNGRU(units=32, return_sequences=True))
    #regressor.add(Activation(gelu))
    classifier.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    classifier.add(CuDNNGRU(units=32))
    #regressor.add(Activation(gelu))
    classifier.add(PReLU(alpha_initializer="zeros"))
    #regressor.add(Dropout(rate=.55))

    #regressor.add(Dense(units=32, activation="sigmoid"))

    classifier.add(Dense(units=5, activation="softmax"))

    classifier.compile(optimizer=optimizer,
                       loss="sparse_categorical_crossentropy",
                       metrics=['sparse_categorical_accuracy'])

    plot_model(classifier, to_file="model.png")

    return classifier


from keras.utils.generic_utils import get_custom_objects


def build_autoencoder(n_features, code=16, optimizer="rmsprop"):

    get_custom_objects().update({'gelu': Activation(gelu)})

    autoencoder = Sequential()

    autoencoder.add(Dense(64, activation=gelu, input_shape=(n_features,)))

    autoencoder.add(Dense(32, activation=gelu))

    autoencoder.add(Dense(code, activation='linear', name="bottleneck"))

    autoencoder.add(Dense(32, activation=gelu))

    autoencoder.add(Dense(64, activation=gelu))

    autoencoder.add(Dense(n_features, activation='sigmoid'))

    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)

    return autoencoder


def train_model(X_train,
                y_train,
                n_past,
                optimizer="rmsprop",
                validation_split=.2,
                shuffle=False,
                batch_size=64,
                epochs=128):

    classifier = build_classifier(n_past, X_train.shape[2], optimizer=optimizer)

    if not os.path.exists("models/"):
        os.mkdir("models/")

    es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)

    rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)

    mcp = ModelCheckpoint(filepath='models/weights.h5',
                          monitor='loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')

    classifier.fit(X_train,
                   y_train,
                   epochs=epochs,
                   callbacks=[es, rlr, mcp, tb],
                   verbose=1,
                   shuffle=shuffle,
                   batch_size=batch_size)

    return classifier


def tune(X_train, y_train, parameters, cv=4, n_jobs=-1):
    estimator = KerasClassifier(build_fn=build_classifier)

    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=parameters,
                               scoring="accuracy",
                               cv=cv,
                               n_jobs=n_jobs)

    grid_search = grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_parameters, best_accuracy


def cv(X, y, n_past, batch_size=64, epochs=50, n_splits=5):

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X, y)

    accuracies = []
    overall = [0]

    for train_index, test_index in skf.split(X, y):
        classifier = None

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = train_model(X_train,
                                 y_train,
                                 n_past,
                                 optimizer="rmsprop",
                                 shuffle=True,
                                 batch_size=batch_size,
                                 epochs=epochs)

        y_pred = classifier.predict(X_test)
        y_pred = np.array([np.argmax(y) for y in y_pred]).reshape(-1, 1)

        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(cm)

        acc = np.trace(cm) / cm.sum()

        print("Accuracy: {}%".format(acc))

        overall = [acc if overall[0] == 0 else (overall[0] + acc) / 2]

        print("Overall accuracy: {}%".format(overall * 100))

    return np.array(accuracies)
