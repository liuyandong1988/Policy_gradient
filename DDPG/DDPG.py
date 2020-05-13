#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    : pendulum_main.py
@Time    : 2020/3/7 22:04
@Author  : Yandong
@Function :
"""
# -*- coding: utf-8 -*-
import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from params import *
import time
import matplotlib.pyplot as plt


class DDPG():
    """
    Deep Deterministic Policy Gradient Algorithms.
    """
    def __init__(self, env, s_dim, a_dim, a_bound):

        self.env = env
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0  # 记录存储经验数
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)  # 注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)


        # 建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        # 更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # 建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()
        # 建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()
        self.R = tl.layers.Input([None, 1], tf.float32, 'r')
        # 建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement
        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


    def train(self):
        """
        training model.
        """
        reward_buffer = []  # 用于记录每个EP的reward，统计变化
        c_loss_record, a_loss_record = list(), list()
        t0 = time.time()  # 统计时间
        for i in range(MAX_EPISODES):
            t1 = time.time()
            # 1. 初始化环境状态
            s = self.env.reset()
            ep_reward = 0  # 记录当前EP的reward
            ec_loss, ea_loss = list(), list()
            for j in range(MAX_EP_STEPS):
                # if self.pointer < MEMORY_CAPACITY:
                #     print('Run data for buffer ... %d'%i)
                # 2.根据观测选择动作
                a = self.choose_action(s)  # 这里很简单，直接用actor估算出a动作
                # --- 策略探索 --- #
                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。
                # 这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪
                new_var = VAR * (1-i/MAX_EPISODES)
                if new_var < 0.05:
                    new_var = 0.05
                a = np.clip(np.random.normal(a, new_var), -2, 2)
                # 3.与环境进行互动
                s_, r, done, info = self.env.step(a)
                # 4.保存s，a，r，s_
                self.store_transition(s, a, r, s_)
                # 5.开始学习
                if self.pointer > MEMORY_CAPACITY:
                    c_loss, a_loss = self.learn()
                    ec_loss.append(c_loss)
                    ea_loss.append(a_loss)
                # 6. 状态转移
                s = s_
                ep_reward += r  # 记录当前EP的总reward
                if j == MAX_EP_STEPS - 1 and self.pointer> MEMORY_CAPACITY:
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward, time.time() - t1))
                    reward_buffer.append(ep_reward)
                    c_loss_record.append(np.mean(np.array(ec_loss)))
                    a_loss_record.append(np.mean(np.array(ea_loss)))

        #     if reward_buffer:
        #         plt.ion()
        #         plt.cla()
        #         plt.title('DDPG')
        #         plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
        #         plt.xlabel('episode steps')
        #         plt.ylabel('normalized state-action value')
        #         plt.ylim(-2000, 0)
        #         plt.show()
        #         plt.pause(0.1)
        # plt.ioff()
        # plt.show()
        print('\nRunning time: ', time.time() - t0)
        np.save('history/reward.npy', np.array(reward_buffer))
        np.save('history/closs.npy', np.array(c_loss_record))
        np.save('history/aloss.npy', np.array(a_loss_record))
        self.save_ckpt()

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # 从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]  # 从bt获得数据r
        bs_ = bt[:, -self.s_dim:]  # 从bt获得数据s'

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_   # y label
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        # target model soft update
        self.ema_update()

        return td_error, a_loss


    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        obs = np.array([s], dtype=np.float32)
        action = self.actor(obs)[0]
        return action

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))
        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_target.hdf5', self.critic_target)


