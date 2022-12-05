from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest.utils import now
from tinkoff.invest.schemas import HistoricCandle, Quotation
from os import getenv
import datetime
import pandas as pd

TOKEN = getenv('INVEST_TOKEN', 'Токена нет')


def convert_to_money(value: Quotation):
    price = value.units + value.nano * 10**(-9)
    return price


def create_df(candles: list[HistoricCandle]):
    df = pd.DataFrame([{'time': candle.time,
                        'volume': candle.volume,
                        'open': convert_to_money(candle.open),
                        'close': convert_to_money(candle.close),
                        'high': convert_to_money(candle.high),
                        'low': convert_to_money(candle.low)} for candle in candles])
    df.index = pd.DatetimeIndex(df['time'])
    df = df.drop('time', axis=1)
    return df


def get_candles_df():
    df = None
    with Client(TOKEN, target=INVEST_GRPC_API) as client:
        df = create_df(client.get_all_candles(
            figi="BBG000B9XRY4",
            from_=now() - datetime.timedelta(days=365),
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        ))
    return df