# -*- coding: utf-8 -*-
"""This module is used to compute informations related to princing history"""

# Importing libraries
import numpy as np
import pandas as pd
import talib

# Global variables


# Functions
def ichimoku(X, shift=26):
    nine_period_high = X.high.rolling(window=9).max()
    nine_period_low = X.low.rolling(window=9).max()
    tenkan_sen = (nine_period_high + nine_period_low) / 2

    period26_high = X.high.rolling(window=26).max()
    period26_low = X.low.rolling(window=26).max()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(shift)

    period52_high = X.high.rolling(window=52).max()
    period52_low = X.low.rolling(window=52).max()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(shift)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


def MACD(X):
    raise NotImplementedError


def bollinger(X):
    #taking closing prices
    #returning (bolu, ma, bold)
    return talib.BBANDS(X)


def fourrier(X):
    raise NotImplementedError


def stochastic(X):
    raise NotImplementedError


def ADX(X):
    raise NotImplementedError
