import gym
from DDPG import DDPG
from params import *
import numpy as np
import tensorflow as tf


def test_main():
    """
    DDPG训练过程
    """
    # 1. 创建环境
    env = gym.make('Pendulum-v0')
    # 2. 实验重现
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
    # 2. DDPG训练方法
    #定义状态空间，动作空间，动作幅度范围
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    ddpg = DDPG(env, s_dim, a_dim, a_bound)
    ddpg.train()

if __name__ == '__main__':
    print('TensorFlow:', tf.__version__)
    test_main()
