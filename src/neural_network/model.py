import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_model(inp_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(inp_shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
