# -*- coding: utf-8 -*-
import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DRL:

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')

        self.model = None
        self.actor = None

    def play(self, m='pg'):
        """
        play game with model.
        """
        print('play...')
        observation = self.env.reset()
        reward_sum = 0
        random_episodes = 0
        while random_episodes < 1:
            self.env.render()
            x = observation.reshape(-1, 4)
            if m == 'pg':
                prob = self.model.predict(x)[0][0]
                action = 1 if prob > 0.5 else 0
            elif m == 'acs':
                prob = self.actor.predict(x)[0][0]
                action = 1 if prob > 0.5 else 0
            else:
                action = np.argmax(self.model.predict(x)[0])
            observation, reward, done, _ = self.env.step(action)
            reward_sum += reward
            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                random_episodes += 1
                reward_sum = 0
                observation = self.env.reset()
        self.env.close()

    def plot(self, history):
        x = history['episode']
        r = history['Episode_reward']
        a_l = history['actor_loss']
        c_l = history['critic_loss']
        # plot reward
        plt.figure(1)
        plt.plot(x, r)
        plt.title('Episode_reward')
        plt.xlabel('episode')
        plt.ylabel('reward')
        # 均值线
        sx = np.array(range(len(r)))
        smooth_func = np.poly1d(np.polyfit(sx, r, 3))
        plt.plot(sx, smooth_func(sx), label='Mean', linestyle='--')


        fig2 = plt.figure(2)
        ax = fig2.add_subplot(121)
        ax.plot(x, a_l)
        ax.set_title('Actor loss')
        ax.set_xlabel('episode')
        ax = fig2.add_subplot(122)
        ax.plot(x, c_l)
        ax.set_title('Critic loss')
        ax.set_xlabel('episode')
        plt.show()

    def save_history(self, history, name):
        name = os.path.join('history', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
