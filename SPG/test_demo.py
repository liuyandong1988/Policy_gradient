#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/19 22:43

import time
import numpy as np
import gym
from tensorflow.keras import models

saved_model = models.load_model('./model/CartPole-v0-pg.h5')
env = gym.make("CartPole-v0")

for i in range(5):
    s = env.reset()
    score = 0
    while True:
        time.sleep(0.05)
        env.render()
        prob = saved_model.predict(np.array([s]))[0]
        a = np.random.choice(len(prob), p=prob)
        s, r, done, _ = env.step(a)
        score += r
        if done:
            time.sleep(1)
            print('using policy gradient, score: ', score)  # 打印分数
            break
env.close()