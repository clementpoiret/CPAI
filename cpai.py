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
import utils.cryptocurrency.cryptocompare as cc
import utils.helpers as hp

# Global variables
N_FUTURE = 32
N_PAST = 2048


# Functions
def get_data():
    print("Getting data...")

    historical_btc = cc.get_historical_data(fsym="BTC", save=False)
    historical_eth = cc.get_historical_data()
    social = cc.get_social_data()

    print("Merging data on seconds from epoch...")
    data = hp.merge_truncate(historical_eth, social)

    historical_btc.columns = [
        s + "_btc" if s != "time" else s for s in historical_btc.columns
    ]

    data = hp.merge_truncate(historical_btc, data).reset_index(drop=True)

    return data


def main():
    """Launcher."""
    data = get_data()

    X_train, y_train = hp.preprocessing_pipeline(data, N_PAST, N_FUTURE)

    pass


if __name__ == "__main__":
    main()
