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

    np.save("tmp/X_train.npy", X_train)
    np.save("tmp/y_train.npy", y_train)
    np.save("tmp/y_test.npy", y_test)

    regressor = md.build_regressor(N_PAST, X_train.shape[2])

    regressor.fit(X_train, y_train, batch_size=64, epochs=128)


if __name__ == "__main__":
    main()
