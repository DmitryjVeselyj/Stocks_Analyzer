import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt
from os import getenv
import tensorflow as tf
import random
from src.forecasting import ARIMAModel

from src.forecasting import XGBoostModel
from src.forecasting import NeuralNetworkModel

TOKEN = getenv('INVEST_TOKEN', 'Токена нет')
SEED = 123
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

    

if __name__ == "__main__":
    # NeuralNetworkModel().run(time_offset=64)
    # ARIMAModel().run(time_offset=64)
    XGBoostModel().run(time_offset=64)
    