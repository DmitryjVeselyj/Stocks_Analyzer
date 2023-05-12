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
from keras.callbacks import History 
history = History()
TOKEN = getenv('INVEST_TOKEN', 'Токена нет')


def another_run():
    df = Loader(TOKEN).get_candles_df("MSFT", now() - dt.timedelta(days=3000), CandleInterval.CANDLE_INTERVAL_DAY)
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


    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('black')

    ax.plot(model_predictions, color='tab:blue', label='Original price')
    ax.plot(test_data, color='tab:green', label='Predicted price')
    plt.legend()
    plt.show()


def run():
    time_offset = 14
    
    df_begin = Loader(TOKEN).get_candles_df("MSFT", now() - dt.timedelta(days=300), CandleInterval.CANDLE_INTERVAL_DAY)

    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    
    # print(df)
    
    # df.to_csv("AAPL_DAY.csv")
    # df_begin = pd.read_csv('stocks_data/AAPL_HOUR.csv')
   
    
    df = df_begin['Close'].values
    df = lfilter(b, a, df)
    df = df.reshape(-1, 1)
    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
  
    dataset_train = scaler.fit_transform(dataset_train)

    dataset_test = scaler.transform(dataset_test)
    x_train, y_train = create_dataset(dataset_train, time_offset)
    x_test, y_test = create_dataset(dataset_test, time_offset)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
   
    model = create_model(x_train.shape[1])
    model.fit(x_train, y_train, validation_split = 0.3, epochs=100, batch_size=32, callbacks=[history], metrics = ['accuracy'])
    plt.plot(history.history['loss'])
    plt.show()

    plt.plot(history.history['val_loss'])
    plt.show()
    # model.save('stock_prediction.h5') 

    # model = load_model("src/neural_network/stock_prediction.h5")
   
    predictions = model.predict(x_test)
    
    predictions = scaler.inverse_transform(predictions)
    train_predictions = scaler.inverse_transform(model.predict(x_train))
    y_train_scaled = scaler.inverse_transform(y_train.reshape(-1, 1)) 
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    MSE_error = mean_squared_error(y_test_scaled, predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))


    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('black')

    ax.plot(y_test_scaled, color='tab:blue', label='Original price')
    ax.plot(predictions, color='tab:green', label='Predicted price')
    plt.legend()
    plt.show()
    

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('black')

    ax.plot(y_train_scaled, color='tab:blue', label='Original price')
    ax.plot(train_predictions, color='tab:green', label='Predicted price')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    run()
    # another_run()
    