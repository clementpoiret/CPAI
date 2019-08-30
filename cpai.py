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
import matplotlib.pyplot as plt
import joblib

# Global variables
N_FUTURE = 32
N_PAST = 2048


def get_datasets(validation_set=False):
    data = hp.get_data()
    data.to_csv("tmp/data.csv", index=False)

    data = data.drop(columns=["time"])

    if validation_set:
        data_train, y_test = hp.split(data, "close", N_FUTURE)
        X_train, y_train = hp.preprocessing_pipeline(data_train, N_PAST,
                                                     N_FUTURE)

        return data, X_train, y_train, y_test

    else:
        X_train, y_train = hp.preprocessing_pipeline(data, N_PAST, N_FUTURE)

    return data, X_train, y_train


def main():
    """Here we go again... Main function, getting data,
    training model, and computing predictions."""

    print("Getting X_train and y_train...")
    data, X_train, y_train = get_datasets()

    print("Building regressor...")
    regressor, history = md.train_model(X_train,
                                        y_train,
                                        N_PAST,
                                        optimizer="rmsprop",
                                        batch_size=64,
                                        epochs=30)
    regressor.save("models/regressor.h5")

    print("Getting last {} hours to predict next {} hours...".format(
        N_PAST, N_FUTURE))
    last = data.iloc[-N_PAST:, :]
    last = hp.preprocessing_pipeline(last,
                                     N_PAST,
                                     N_FUTURE,
                                     is_testing_set=True)

    prediction = regressor.predict(last)[0].reshape(-1, 1)

    sc = joblib.load("scalers/MinMaxScaler_predict.pkl")

    prediction = sc.inverse_transform(prediction)

    last_eth = data.iloc[-N_PAST:, :].close.values.reshape(-1, 1)
    prices = np.concatenate((last_eth, prediction))
    plt.plot(prices)
    plt.axvline(N_PAST, linestyle=":")
    plt.savefig("prediction.png")
    plt.show()
    #prediction = regressor.predict(X_test)[0].reshape(-1, 1)
    #prediction = sc.inverse_transform(prediction)

    #plt.plot(y_test)
    #plt.plot(prediction)
    #plt.show()


if __name__ == "__main__":
    main()
