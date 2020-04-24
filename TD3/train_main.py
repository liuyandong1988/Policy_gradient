'''
Environment
---
Openai Gym Pendulum-v0, continuous action space
https://gym.openai.com/envs/Pendulum-v0/

Prerequisites
---
tensorflow 2.1.0
tensorflow-probability 0.9.0
tensorlayer 2.2.1
gym 0.17.1
'''

import  TD3
import gym
import numpy as np
import tensorflow as tf
from params import *



def main():
    """
    Training AC model
    :return:
    """
    # 1. 创建环境
    env = gym.make('Pendulum-v0')
    # 2. 实验重现
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
    # 2. DDPG训练方法
    td3 = TD3.TD3(env)
    td3.train()




if __name__ == '__main__':
    main()