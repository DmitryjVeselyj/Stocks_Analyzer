import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt
from ..neural_network.dataset_creation import create_dataset
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..base_model.model import BaseModel

class XGBoostModel(BaseModel):
    def run(self, time_offset=64):
        df = pd.read_csv('stocks_data/AAPL_HOUR.csv')
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
        MAE_error = mean_absolute_error(y_test, predictions)
        print('Testing Mean Average Error is {}'.format(MAE_error))

