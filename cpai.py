# -*- coding: utf-8 -*-
"""CPAI (for CryptoCurrency Prediction AI), is developed to try to predict
future prices (or at least trends) of CryptoCurrencies.
Copyright (C) 2019  Clément POIRET

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For any questions, contact me at poiret.clement[at]outlook[dot]fr"""

# Import libraries
import numpy as np
import pandas as pd
import utils.helpers as hp
import utils.neuralnet.model as md

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Global variables
N_FUTURE = 32
N_PAST = 2048


def main():
    """Launcher."""
    data = hp.get_data()
    data.to_csv("tmp/data.csv", index=False)

    data = data.drop(columns=["time"])
    data_train, y_test = hp.split(data, "close", N_FUTURE)

    X_train, y_train = hp.preprocessing_pipeline(data_train, N_PAST, N_FUTURE)
    """
    X_train = np.load("tmp/X_train.npy")

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    y_train = np.load("tmp/y_train.npy")

    y_test = np.load("tmp/y_test.npy")
    """

    regressor = md.build_regressor(N_PAST, X_train.shape[2])

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
                            epochs=128,
                            callbacks=[es, rlr, mcp, tb],
                            verbose=1,
                            batch_size=64)


if __name__ == "__main__":
    main()
