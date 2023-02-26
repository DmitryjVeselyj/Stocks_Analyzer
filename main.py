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
import pandas_ta as ta

TOKEN = getenv('INVEST_TOKEN', 'Токена нет')


        


def run():
    time_offset = 14
    # df = Loader(TOKEN).get_candles_df("AAPL", now() - dt.timedelta(days=700), CandleInterval.CANDLE_INTERVAL_DAY)
    # df.to_csv("AAPL_DAY.csv")
    df = pd.read_csv('stocks_data/TSLA_HOUR.csv')
    df = df['Close'].values
    df = df.reshape(-1, 1)
    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])
    
    model = ARIMA(dataset_train[:-1], order=(4,1,3))
    fitted = model.fit()
    prediction = fitted.forecast()
    print(prediction)
 
    # sti = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)['SUPERTd_10_3.0']
    
    # fig, ax = plt.subplots()
    # m = []
    # c = []
    # for i in range(1, len(sti)):
    #     if sti[i-1] == -1 and sti[i] == 1:
    #         c.append('green')
    #         m.append('^')
    #     elif sti[i-1] == 1 and sti[i] == -1:
    #         c.append('red')
    #         m.append('v')
    #     else:
    #         c.append('tab:blue')
    #         m.append('')
    # ax.plot(range(df.shape[0]), df['Close'])
    # print(len(m), df.shape[0])        
    # for i in range(1, df.shape[0]):
    #    ax.scatter(range(df.shape[0])[i], df['Close'][i], marker = m[i-1], color = c[i-1])
    # plt.show()
    
    # df = df['Close'].values
    # df = df.reshape(-1, 1)
    # dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    # dataset_test = np.array(df[int(df.shape[0]*0.8):])

    # scaler = MinMaxScaler(feature_range=(0,1))
    # dataset_train = scaler.fit_transform(dataset_train)
    # dataset_test = scaler.transform(dataset_test)
    # x_train, y_train = create_dataset(dataset_train, time_offset)
    # x_test, y_test = create_dataset(dataset_test, time_offset)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # # model = create_model(x_train.shape[1])
    # # model.fit(x_train, y_train, epochs=100, batch_size=32)
    # # model.save('stock_prediction.h5') 

    # model = load_model("src/neural_network/stock_prediction.h5")
   
    # predictions = model.predict(x_test)
    # predictions = scaler.inverse_transform(predictions)
    # y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # fig, ax = plt.subplots(figsize=(16,8))
    # ax.set_facecolor('black')
    
    # ax.plot(y_test_scaled, color='tab:blue', label='Original price')
    # ax.plot(predictions, color='tab:green', label='Predicted price')
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    run()
    