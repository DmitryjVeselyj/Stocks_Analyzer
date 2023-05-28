import numpy as np
import pandas as pd
from ta.trend import sma_indicator, ema_indicator
from ta.utils import dropna
from ta.momentum import rsi
from ta.volatility import BollingerBands
from ta.volume import on_balance_volume, money_flow_index, acc_dist_index, volume_price_trend, force_index, chaikin_money_flow


def create_dataset(df, time_offset, volume=[]):
    x = []
    y = []
    for i in range(time_offset, df.shape[0]):
        tmp_df = df[i-time_offset:i, 0]
        # rs = rsi(pd.Series(tmp_df), window=28)
        sma_12 = sma_indicator(pd.Series(tmp_df))
        sma_55 = sma_indicator(pd.Series(tmp_df), window=55)
        obv = force_index(pd.Series(tmp_df), pd.Series(volume[i - time_offset:i, 0]))
        # x.append(tmp_df)
        x.append(np.array([sma_12[len(sma_12) - 1] ,sma_55[len(sma_55) - 1], obv[len(obv) - 1]]))
        # x.append(np.array([ta.sma(tmp_df, length = 10), ta.ema(tmp_df, length = 10)]))
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

