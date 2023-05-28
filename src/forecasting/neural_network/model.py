import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from tinkoff.invest.utils import now
from sklearn.preprocessing import MinMaxScaler
from .dataset_creation import create_dataset
from keras.callbacks import History
from keras.utils import plot_model
from ..base_model.model import BaseModel 
history = History()


class NeuralNetworkModel(BaseModel):
    @staticmethod
    def create_model(inp_shape):
        model = Sequential(name="LSTM_1_model")
        model.add(LSTM(units=48, return_sequences=True, input_shape=inp_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=48, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(units=48))
        # # model.add(Dropout(0.2))
        # model.add(LSTM(units=48))
        # model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model


    def run(self, time_offset=64):
        begin_time = now() - dt.timedelta(days=8)
        # df_begin = Loader(TOKEN).get_candles_df("AAPL", begin_time - dt.timedelta(days=365), CandleInterval.CANDLE_INTERVAL_DAY)
        # print(time_offset)
        n = 15  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        
        # print(df)
        
        # df_begin.to_csv("stocks_data/AAPL_DAY.csv")
        df_begin = pd.read_csv('stocks_data/AAPL_HOUR.csv')
    
    

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
        model = self.create_model(x_train[0].shape)
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
        MAE_error = mean_absolute_error(y_test_scaled, predictions)
        print('Testing Mean Average Error is {}'.format(MAE_error))
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


