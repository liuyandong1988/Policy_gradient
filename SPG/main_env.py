#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/19 21:14

import gym
from policy_gradient import PolicyGradient


def test_cartpole():
    env = gym.make('CartPole-v0')
    agent_pg = PolicyGradient(env)
    agent_pg.learning(episodes=5000)


if __name__ == '__main__':
    test_cartpole()