import mplfinance as mpf
import datetime as dt
from ta import trend
from src.loader import Loader
from os import getenv
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import now

TOKEN = getenv('INVEST_TOKEN', 'Токена нет')


def run():
    df = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_DAY)

    ma8 = mpf.make_addplot(trend.sma_indicator(df['close'], window=8))
    ma55 = mpf.make_addplot(trend.sma_indicator(df['close'], window=55))
    ma144 = mpf.make_addplot(trend.sma_indicator(df['close'], window=144))
    print(df)
    plots = [ma8, ma55, ma144]
    mpf.plot(df[['volume', 'open', 'close', 'high', 'low']],
             type='candle', style='yahoo', addplot=plots, volume=True)
    mpf.show()


if __name__ == "__main__":
    run()
    