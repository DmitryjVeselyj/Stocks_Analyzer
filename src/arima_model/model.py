import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

def another_run():
    # df = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_HOUR)
    df = pd.read_csv('stocks_data/AAPL_HOUR.csv')
    train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(0,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    print(model_predictions)
    print(test_data)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))
    MAE_error = mean_absolute_error(test_data, model_predictions)
    print('Testing Mean Average Error is {}'.format(MAE_error))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('black')

    ax.plot(test_data, color='tab:blue', label='Original price')
    ax.plot(model_predictions, color='tab:green', label='Predicted price')
    plt.legend()
    plt.title('APPLE Stocks')
    plt.show()
