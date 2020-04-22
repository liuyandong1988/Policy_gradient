import time
import numpy as np
import gym
from tensorflow.keras import models
import tensorflow_probability as  tfp
import tensorflow as tf

actor = models.load_model('./model/ppo_actor.h5')
env = gym.make("Pendulum-v0")

# 1.初始环境
s = env.reset()
score = 0
while True:
    # 2. GUI 图示
    env.render()
    time.sleep(0.01)
    # 3. observation--> action
    s = s[np.newaxis, :].astype(np.float32)
    mu, sigma = actor.predict(s)  # 通过actor计算出分布的mu和sigma
    pi = tfp.distributions.Normal(mu, sigma)  # 用mu和sigma构建正态分布
    action = tf.squeeze(pi.sample(1), axis=0)[0]  # 根据概率分布随机出动作
    # 4. 与环境交互
    s, r, done, _ = env.step(action)
    score += r
    if done:
        time.sleep(1)
        print('Using proximal policy optimization score: ', score)  # 打印分数
        break
input('Game over')
