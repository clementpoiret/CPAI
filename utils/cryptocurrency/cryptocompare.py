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
                        allData="true"):

    base = "{}histoday?".format(BASE)
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "e": e,
        "limit": limit,
        "allData": allData,
        "api_key": API_KEY
    }

    f = requests.get(base, params=params).json()
    data = pd.DataFrame(f['Data'])

    return data


def get_social_data(coin="ETH", limit=2000):
    """TODO: SocialData & PricingData must have the same timestamps"""
    # BTC: 1182
    # ETH: 7605
    list = get_coinlist()
    coin_id = list.loc[coin, :].Id

    base = "{}social/coin/histo/day?".format(BASE)
    params = {"coinId": coin_id, "limit": limit, "api_key": API_KEY}

    f = requests.get(base, params=params).json()
    data = pd.DataFrame(f['Data'])

    return data