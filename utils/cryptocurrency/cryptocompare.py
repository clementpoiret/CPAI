# -*- coding: utf-8 -*-
"""This module is used to get informations from CryptoCompare"""

# Importing libraries
import os
import pandas as pd
import time
import datetime
import requests

# Global variables
API_KEY = pd.read_csv("credentials")["CryptoCompareAPI"][0]
BASE = "https://min-api.cryptocompare.com/data/"


def get_coinlist():
    base = "{}all/coinlist".format(BASE)

    f = requests.get(base).json()
    data = pd.DataFrame(f['Data'])

    return data.T


def get_exchangelist():
    base = "{}v3/all/exchanges".format(BASE)

    f = requests.get(base).json()
    data = pd.DataFrame(f['Data'])

    return data.T


def get_historical_data(fsym="ETH",
                        tsym="USD",
                        e="kraken",
                        limit=2000,
                        maxEntry=18000,
                        allData="true",
                        save=True):

    def inner_get(fsym=fsym,
                  tsym=tsym,
                  e=e,
                  limit=limit,
                  maxEntry=maxEntry,
                  allData=allData):

        base = "{}histohour?".format(BASE)
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "e": e,
            "limit": limit,
            "allData": allData,
            "api_key": API_KEY
        }

        print("Getting {}'s historical data from CryptoCompare's API...".format(
            fsym))

        data = pd.DataFrame()
        for i in range(int(maxEntry / limit)):
            f = requests.get(base, params=params).json()
            data = pd.DataFrame(f['Data']).append(data)

            params["toTs"] = data.time.iloc[0] - 3600

        return data.reset_index(drop=True)

    if os.path.exists("tmp/data_{}{}.csv".format(fsym, tsym)):
        last_historical = pd.read_csv("tmp/data_{}{}.csv".format(fsym, tsym))
        last_time = last_historical.time.iloc[-1]
        now = time.time()
        d = int((now - last_time) / 3600)

        if (d < 2000) & (d > 3):
            limit = d - 1
            maxEntry = d - 1

            print("Updating local historical database...")
            data = inner_get(fsym=fsym,
                             tsym=tsym,
                             e=e,
                             limit=limit,
                             maxEntry=maxEntry,
                             allData=allData)

            data = last_historical.append(data).reset_index(drop=True)

        elif d > 2000:
            print("Local database too old, rebuilding...")
            data = inner_get(fsym=fsym,
                             tsym=tsym,
                             e=e,
                             limit=limit,
                             maxEntry=maxEntry,
                             allData=allData)

        else:
            print("Local database is less than 3 hours old, no update needed.")
            data = last_historical
    else:
        data = inner_get(fsym=fsym,
                         tsym=tsym,
                         e=e,
                         limit=limit,
                         maxEntry=maxEntry,
                         allData=allData)

    if save:
        if not os.path.exists("tmp/"):
            os.mkdir("tmp/")

        data.to_csv("tmp/data_{}{}.csv".format(fsym, tsym), index=False)

    return data


def get_social_data(coin="ETH", limit=2000, maxEntry=18000, save=True):
    """TODO: SocialData & PricingData must have the same timestamps"""

    # BTC: 1182
    # ETH: 7605

    def inner_get(coin, limit, maxEntry):
        list = get_coinlist()
        coin_id = list.loc[coin, :].Id

        base = "{}social/coin/histo/hour?".format(BASE)
        params = {"coinId": coin_id, "limit": limit, "api_key": API_KEY}

        print(
            "Getting {}'s social data from CryptoCompare's API...".format(coin))

        data = pd.DataFrame()
        for i in range(int(maxEntry / limit)):
            f = requests.get(base, params=params).json()
            data = pd.DataFrame(f['Data']).append(data)

            params["toTs"] = data.time.iloc[0] - 3600

        return data.reset_index(drop=True)

    if os.path.exists("tmp/social_{}.csv".format(coin)):
        last_social = pd.read_csv("tmp/social_{}.csv".format(coin))
        last_time = last_social.time.iloc[-1]
        now = time.time()
        d = int((now - last_time) / 3600)

        if (d < 2000) & (d > 3):
            limit = d - 1
            maxEntry = d - 1

            print("Updating local social database...")
            data = inner_get(coin, limit, maxEntry)

            data = last_social.append(data).reset_index(drop=True)

        elif d > 2000:
            print("Local database too old, rebuilding...")
            data = inner_get(coin, limit, maxEntry)

        else:
            print("Local database is less than 3 hours old, no update needed.")
            data = last_social

    else:
        data = inner_get(coin, limit, maxEntry)

    if save:
        if not os.path.exists("tmp/"):
            os.mkdir("tmp/")

        data.to_csv("tmp/social_{}.csv".format(coin), index=False)

    return data
