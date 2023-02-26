import gym
import gym_anytrading
from stable_baselines3 import A2C
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions 



def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]
    return prices, signal_features


class MyStocksEnv(StocksEnv):
    _process_data = my_process_data