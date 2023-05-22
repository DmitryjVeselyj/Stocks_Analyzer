import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN


def create_model(inp_shape):
    model = Sequential(name="LSTM_1_model")
    model.add(LSTM(units=48,return_sequences=True, input_shape=inp_shape))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=66,return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(units=48, return_sequences=True))
    # # model.add(Dropout(0.2))
    model.add(LSTM(units=48))
    # model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model
