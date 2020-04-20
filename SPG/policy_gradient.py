#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/19 21:16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def build_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation = 'relu')(inputs)
    x = Dense(32,  activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


class PolicyGradient(object):

    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.pg_model = build_model(self.state_dim, self.action_dim)
        self.pg_model.compile(loss='mean_squared_error', optimizer=Adam())


    def choose_action(self, state):
        """choose the action based on possibility"""
        state = np.array(state)[np.newaxis, :]
        prob = self.pg_model.predict(state)[0]
        return np.random.choice(len(prob), p=prob)

    def discount_rewards(self, rewards, gamma=0.95):
        """discount rewards centralized and normalized"""
        prior = 0
        out = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            prior = prior * gamma + rewards[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def train(self, records):
        """each record: s0, a0, r1"""
        s_batch = np.array([record[0] for record in records])  # batch states
        # action one-hot prob_batch
        a_batch = np.array([[1 if record[1] == i else 0 for i in range(self.action_dim)] for record in records])
        # 假设predict的概率是 [0.2, 0.7, 0.1]，选择的动作是 [0, 1, 2]
        # 则动作[0, 1]的概率等于 [0, 0.7, 0] = [0.2, 0.7, 0.1] * [0, 1, 0]
        prob_batch = self.pg_model.predict(s_batch) * a_batch
        r_batch = self.discount_rewards([record[2] for record in records])
        # 折损回报值作为模型的权重参数进行训练
        self.pg_model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)

    def learning(self, episodes=1000):
        score_list = []  # records
        for i in range(episodes):
            # 1. 初始化全局状态
            s0 = self.env.reset()
            score = 0
            replay_records = []
            while True:
                # 2.预测动作
                a0 = self.choose_action(s0)
                # 3.与环境交互，环境动力学
                s1, r1, done, _ = self.env.step(a0)
                # 4.记录信息，回溯计算折损回报值
                replay_records.append((s0, a0, r1))
                score += r1
                s0 = s1
                if done:
                    # 5.完成一个episode，Monte Carlo方法训练网络
                    self.train(replay_records)
                    score_list.append(score)
                    print('episode:', i, 'score:', score, 'max:', max(score_list))
                    break
            # 最后20次的平均分大于 195 时，停止并保存模型
            if np.mean(score_list[-20:]) > 195:
                print('Save the model......')
                self.pg_model.save('./model/CartPole-v0-pg.h5')
                break
        self.env.close()

        plt.plot(score_list)
        x = np.array(range(len(score_list)))
        smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
        plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
        plt.title('Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.show()