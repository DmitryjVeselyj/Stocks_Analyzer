import mplfinance as mpf
from ta import trend
from src.utils import get_candles_df

def run():
    df = get_candles_df()
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
