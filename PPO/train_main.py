import gym
import tensorflow as tf
import numpy as np
from PPO import PPO


def train_main():
    # 1.创建交互环境
    env = gym.make('Pendulum-v0' )
    # env.reset()
    # env.render()
    # input('123')
    # 2.实验重现
    RANDOMSEED = 1  # random seed
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
    # 3. OPP算法实现
    ppo = PPO(env)
    ppo.train(1000)


if __name__ == '__main__':
    train_main()