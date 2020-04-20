#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    : play_demo.py
@Time    : 2020/3/13 23:49
@Author  : Yandong
@Function : 
"""

import a2c
import csv

# 1.生成学习框架
AC_player = a2c.AC()
# 2.加载模型参数
AC_player.load()
# 3.展示
# AC_player.play('acs')

# read the history and plot the reward and loss
first_time = True
episode, reward, actor_loss, critic_loss = list(), list(), list(), list()
sFileName='./history/ac_sparse.csv'
with open(sFileName,newline='',encoding='UTF-8') as csvfile:
    rows=csv.reader(csvfile)
    for row in rows:
        if first_time:
            first_time = False
            continue
        episode.append(int(row[0]))
        reward.append(float(row[1]))
        actor_loss.append(float(row[2]))
        critic_loss.append(float(row[3]))
history = dict()
history['episode'] = episode
history['Episode_reward'] = reward
history['actor_loss'] = actor_loss
history['critic_loss'] = critic_loss
AC_player.plot(history)