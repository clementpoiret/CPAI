# -*- coding: utf-8 -*-
"""This module is used to get informations from CryptoCompare"""

# Importing libraries
import pandas as pd
import urllib
import requests

# Global variables
API_KEY = pd.read_csv("utils/cryptocurrency/credentials")["CryptoCompareAPI"][0]


def get_historical_data(fsym="ETH",
                        tsym="USD",
                        e="kraken",
                        limit=2000,
                        allData="true"):

    base = "https://min-api.cryptocompare.com/data/histoday?"
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "e": e,
        "limit": limit,
        "allData": allData,
        "api_key": API_KEY
    }

    url = "{}{}".format(base, urllib.parse.urlencode(params))

    f = requests.get(url)
    ipdata = f.json()
    data = pd.DataFrame(ipdata['Data'])

    return data