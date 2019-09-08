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
along with this program.  If not, see <https://www.gnu.org/licenses/>.v

For any questions, contact me at poiret.clement[at]outlook[dot]fr"""

# Import libraries
import joblib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.models import load_model

import utils.helpers as hp
import utils.neuralnet.model as md

# Global variables
N_FUTURE = 32
N_PAST = 2048
LOW_TRIGGER = .19
HIGH_TRIGGER = 1


def get_datasets(validation_set=False):
    data = hp.get_data()
    data.to_csv("tmp/data.csv", index=False)

    time = data.time
    data = data.drop(columns=["time"])

    if validation_set:
        data_train, y_test = hp.split(data, "close", N_FUTURE)
        X_train, y_train = hp.preprocessing_pipeline(data_train, N_PAST,
                                                     N_FUTURE, False,
                                                     LOW_TRIGGER, HIGH_TRIGGER)

        return time, data, X_train, y_train, y_test

    else:
        X_train, y_train = hp.preprocessing_pipeline(data, N_PAST, N_FUTURE,
                                                     False, LOW_TRIGGER,
                                                     HIGH_TRIGGER)

    return time, data, X_train, y_train


def main():
    """Here we go again... Main function, getting data,
    training model, and computing predictions."""

    print("Getting X_train and y_train...")
    time, data, X_train, y_train, y_test = get_datasets(validation_set=1)

    #classifier = load_model("models/classifier.h5")
    accuracies = md.cv(X_train,
                       y_train,
                       n_past=N_PAST,
                       batch_size=64,
                       epochs=60,
                       n_splits=5)

    np.save("accuracies", accuracies)

    print("Building classifier...")
    classifier = md.train_model(X_train,
                                y_train,
                                N_PAST,
                                optimizer="rmsprop",
                                shuffle=True,
                                batch_size=128,
                                epochs=64)
    classifier.save("models/classifier.h5")

    print("Getting last {} hours to predict next {} hours...".format(
        N_PAST, N_FUTURE))

    #timepred = np.concatenate(
    #    (time[-N_PAST:].values,
    #     [time.iloc[-1] + (1 + n) * 3600 for n in range(N_FUTURE)]))

    last = data.iloc[-N_PAST:, :]
    last = hp.preprocessing_pipeline(last,
                                     N_PAST,
                                     N_FUTURE,
                                     is_testing_set=True)

    prediction = classifier.predict(last)[0].reshape(-1, 1)
    ind = np.array([x for x in range(5)]).reshape(-1, 1)
    prediction = np.concatenate((ind, prediction), axis=1)
    prediction = prediction[prediction[:, 1].argsort()]

    cat1 = hp.get_category(prediction[-1, 0])
    cat2 = hp.get_category(prediction[-2, 0])
    #sc = joblib.load("scalers/MinMaxScaler_predict.pkl")

    #prediction = sc.inverse_transform(prediction)
    coef, values = hp.reg(y_test.values, True)
    last_eth = data.iloc[-N_FUTURE -
                         N_FUTURE:-N_FUTURE, :].close.values.reshape(-1, 1)
    #prices_pred = np.concatenate((last_eth, prediction))
    prices_reg = np.concatenate((last_eth, values.reshape(-1, 1)))
    prices_real = np.concatenate((last_eth, y_test.values.reshape(-1, 1)))
    #plt.plot(prices_pred, label="Prediction", color="red")
    fig, ax = plt.subplots()
    plt.plot(prices_reg,
             label="Linear regression; {}".format(coef),
             color="red")
    plt.plot(prices_real, label="Reality", color="black")
    plt.text(.05,
             .1,
             'Prediction 1: {} @{:.2f}'.format(cat1, prediction[-1, 1]),
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax.transAxes)
    plt.text(.05,
             .05,
             'Prediction 2: {} @{:.2f}'.format(cat2, prediction[-2, 1]),
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax.transAxes)
    plt.axvline(N_FUTURE, linestyle=":", label="End of training set")
    plt.legend()
    plt.savefig("prediction.png")
    plt.show()

    #pd.DataFrame({
    #    "time": timepred,
    #    "prediction": prices_pred[:, 0]
    #}).to_csv("prediction.csv")

    #prediction = regressor.predict(X_test)[0].reshape(-1, 1)
    #prediction = sc.inverse_transform(prediction)

    #plt.plot(y_test)
    #plt.plot(prediction)
    #plt.show()


if __name__ == "__main__":
    main()
