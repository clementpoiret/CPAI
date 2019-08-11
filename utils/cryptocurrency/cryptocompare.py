# -*- coding: utf-8 -*-
"""This module is used to get informations from CryptoCompare"""

# Importing libraries
import pandas as pd
import requests

# Global variables
API_KEY = pd.read_csv("utils/cryptocurrency/credentials")["CryptoCompareAPI"][0]
BASE = "https://min-api.cryptocompare.com/data/"


def get_coinlist():
    base = "{}all/coinlist".format(BASE)

    f = requests.get(base).json()
    data = pd.DataFrame(f['Data'])

    return data.T


def get_exchangelist():
    raise NotImplementedError


def get_historical_data(fsym="ETH",
                        tsym="USD",
                        e="kraken",
                        limit=2000,
                        maxEntry=18000,
                        allData="true"):

    base = "{}histohour?".format(BASE)
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "e": e,
        "limit": limit,
        "allData": allData,
        "api_key": API_KEY
    }

    data = pd.DataFrame()
    for i in range(int(maxEntry / limit)):
        f = requests.get(base, params=params).json()
        data = pd.DataFrame(f['Data']).append(data)

        params["toTs"] = data.time.iloc[0] - 3600

    return data.reset_index()


def get_social_data(coin="ETH", limit=2000, maxEntry=18000):
    """TODO: SocialData & PricingData must have the same timestamps"""
    # BTC: 1182
    # ETH: 7605
    list = get_coinlist()
    coin_id = list.loc[coin, :].Id

    base = "{}social/coin/histo/hour?".format(BASE)
    params = {"coinId": coin_id, "limit": limit, "api_key": API_KEY}

    data = pd.DataFrame()
    for i in range(int(maxEntry / limit)):
        f = requests.get(base, params=params).json()
        data = pd.DataFrame(f['Data']).append(data)

        params["toTs"] = data.time.iloc[0] - 3600

    return data.reset_index()