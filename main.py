import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt
from src.loader import Loader
from os import getenv
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import now
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from src.neural_network.model import create_model
from src.neural_network.dataset_creation import create_dataset
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
from keras.callbacks import History
from keras.utils import plot_model 
history = History()
from statsmodels.graphics.tsaplots import plot_acf
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller

TOKEN = getenv('INVEST_TOKEN', 'Токена нет')
SEED = 123
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.get_logger().setLevel(3)


def another_another_run(time_offset=90):
    df = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_HOUR)
    df_volume  = df['Volume'].values.reshape(-1, 1)

    dataset_train_volume = np.array(df_volume[:int(df.shape[0]*0.8)])
    dataset_test_volume = np.array(df_volume[int(df.shape[0]*0.8):])

    df = df['Close'].values
    
    

    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    
    # df = lfilter(b, a, df)
    df = df.reshape(-1, 1)
    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])
    x_train, y_train = create_dataset(dataset_train, time_offset, dataset_train_volume)
    x_test, y_test = create_dataset(dataset_test, time_offset, dataset_test_volume)
    # create an xgboost regression model
    model = xgb.XGBRegressor(n_estimators=300, max_depth=10)
    print(x_train, y_train)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('black')

    ax.plot(y_test, color='tab:blue', label='Original price')
    ax.plot(predictions, color='tab:green', label='Predicted price')

    plt.legend()
    plt.show()



    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pd.DataFrame({'pred': predictions.reshape(-1), 'pred_test': y_test.reshape(-1)}))


    MSE_error = mean_squared_error(y_test, predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))


def another_run():
    df = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_DAY)
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
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('black')

    ax.plot(test_data, color='tab:blue', label='Original price')
    ax.plot(model_predictions, color='tab:green', label='Predicted price')
    plt.legend()
    plt.title('APPLE Stocks')
    plt.show()


def run(time_offset = 14):
    df_begin = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_HOUR)
    # print(time_offset)
    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    
    # print(df)
    
    # df.to_csv("AAPL_DAY.csv")
    # df_begin = pd.read_csv('stocks_data/AAPL_DAY.csv')
   
   

    df = df_begin['Close'].values

    df = df.reshape(-1, 1)

    df_volume  = df_begin['Volume'].values.reshape(-1, 1)

    dataset_train_volume = np.array(df_volume[:int(df.shape[0]*0.8)])
    dataset_test_volume = np.array(df_volume[int(df.shape[0]*0.8):])

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
  
    dataset_train = scaler.fit_transform(dataset_train)
    # pd.plotting.autocorrelation_plot(dataset_train)
    # plt.show()
    dataset_test = scaler.transform(dataset_test)
    x_train, y_train = create_dataset(dataset_train, time_offset, dataset_train_volume)
    x_test, y_test = create_dataset(dataset_test, time_offset, dataset_test_volume)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    model = create_model(x_train[0].shape)
    print(x_train[0].shape)
    print(model.summary())
    model.fit(x_train, y_train, validation_split = 0.3, epochs=100, batch_size=32, callbacks=[history])
    # print(model.weights)
    plt.yscale("log") 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # # pd.DataFrame(history.history).plot(figsize=(16,8))

    # plt.show()

    # model.save('absolute_stock_prediction.h5') 

    # model = load_model("src/neural_network/stock_prediction.h5")
   
    # print(x_test)
    predictions = model.predict(x_test)
    # print(predictions)
    # predictions = model.predict(x_test[1])
    # print(predictions)
    # print(np.array([x_test[22]]))
    # predictions = model.predict(np.array([x_test[22]]))
    predictions = scaler.inverse_transform(predictions)
    
    y_train_scaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    x_test_scaled = scaler.inverse_transform(x_test.reshape(-1, 1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    MSE_error = mean_squared_error(y_test_scaled, predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))
    print(pd.DataFrame({'pred': predictions.reshape(-1), 'pred_test': y_test_scaled.reshape(-1)}))
    # diff = predictions[0] - y_test_scaled[0]
    # print(diff)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(pd.DataFrame({'pred': predictions.reshape(-1), 'pred_test': y_test_scaled.reshape(-1)}))
    # # print((predictions - diff)[:10])
    

    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('black')

    ax.plot(y_test_scaled, color='tab:blue', label='Original price')
    ax.plot(predictions, color='tab:green', label='Predicted price')
    plt.legend()
    plt.title('APPLE Stocks')
    plt.show()
    
    


if __name__ == "__main__":
    # steps = np.random.standard_normal(1000)
    # steps[0]=0
    # random_walk = np.cumsum(steps)
    # plt.plot(random_walk)
    # plt.show()
    # run(time_offset=64)
    # res = []
    # for time_offset in range(4, 80, 5):
    #     mse_error = run(time_offset)
    #     res.append([time_offset, mse_error])
    # print(res)    
    # another_run()
    another_another_run(time_offset=64)
    