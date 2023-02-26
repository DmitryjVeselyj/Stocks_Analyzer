from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.schemas import HistoricCandle
from tinkoff.invest.constants import INVEST_GRPC_API
import pandas as pd
import datetime
from src.utils import find_first, convert_to_money


class Loader:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_figi_by_ticker(self, ticker: str) -> str:
        with Client(self._token, target=INVEST_GRPC_API) as client:
            shares = client.instruments.shares().instruments
            tinkoff_share = find_first(
                shares, lambda share: share.ticker == ticker)
        return getattr(tinkoff_share, 'figi', None)

    def get_candles_df(self, ticker: str, time_from: datetime.datetime, interval: CandleInterval):
        with Client(self._token, target=INVEST_GRPC_API) as client:
            figi = self.get_figi_by_ticker(ticker)
            df = self._create_df(client.get_all_candles(
                figi=figi,
                from_=time_from,
                interval=interval,
            ))
        return df

    def _create_df(self, candles: list[HistoricCandle]) -> pd.DataFrame:
        df = pd.DataFrame([{'Time': candle.time.replace(tzinfo=None),
                            'Volume': candle.volume,
                            'Open': convert_to_money(candle.open),
                            'Close': convert_to_money(candle.close),
                            'High': convert_to_money(candle.high),
                            'Low': convert_to_money(candle.low)} for candle in candles])
        # df.index = pd.DatetimeIndex(df['Time'])
    
        # df = df.drop('Time', axis=1)
        
        return df
