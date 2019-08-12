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
    #taking close price
    macd, macdsignal, macdhist = talib.MACD(X.close,
                                            fastperiod=12,
                                            slowperiod=26,
                                            signalperiod=9)
    return macd, macdsignal, macdhist


def bollinger(X):
    #taking closing prices
    bolu, ma, bold = talib.BBANDS(X.close)
    return bolu, ma, bold


def fourier(X):
    close_fft = np.fft.fft(np.asarray(X.close.tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    #fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    #fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())

    fourier = pd.DataFrame()
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        fourier[num_] = np.fft.ifft(fft_list_m10)

    return fourier


def stochastic_rsi(X):
    fastk, fastd = talib.STOCHRSI(X.close,
                                  timeperiod=14,
                                  fastk_period=5,
                                  fastd_period=3,
                                  fastd_matype=0)
    return fastk, fastd


def ADX(X):
    real = talib.ADX(X.high, X.low, X.close, timeperiod=14)
    return real
