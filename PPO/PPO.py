from params import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import tensorflow_probability as tfp


def critic_model(state_dim):
    """
    build model
    """
    inputs = Input(shape=(state_dim), name='state')
    x = Dense(100, activation='relu')(inputs)
    action = Dense(1)(x)
    model = Model(inputs=inputs, outputs=action)
    model.summary()
    return model


def actor_model(state_dim, action_dim, name):
    '''
    Build policy network
    '''
    # 连续动作型问题，输出mu和sigma。
    inputs = Input(shape=(state_dim,), name=name + '_state')
    x = Dense(100, activation='relu', name=name + '_l1')(inputs)
    a = Dense(action_dim, activation='tanh', name=name + '_a')(x)
    mu = Lambda(lambda x: x * 2, name=name + '_lambda')(a)
    sigma = Dense(action_dim, activation='softplus', name=name + '_sigma')(x)
    model = Model(inputs=inputs, outputs=[mu, sigma])
    return model


class PPO(object):

    def __init__(self, env):

        self.env = env
        # 1. 构建critic网络：输入state，输出V值
        self.critic = critic_model(S_DIM)
        self.critic.compile(loss='mean_squared_error', optimizer=Adam(C_LR))
        # 2. 构建actor网络：
        # actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入时state，输出是描述动作分布的mu和sigma
        self.actor = actor_model(S_DIM, A_DIM, 'pi')
        self.actor_old = actor_model(S_DIM, A_DIM, 'oldpi')
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    def train(self, episode_max):
        """
        训练actor网络
        """
        all_ep_r = []
        for ep in range(episode_max):
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            t0 = time.time()
            for t in range(EP_LEN):  # in one episode
                # env.render()
                # 1. 策略动作
                a = self.choose_action(s)
                # 2. 环境交互
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # 对奖励进行归一化。
                # 状态转移
                s = s_
                ep_r += r
                # 3.N步更新的方法，每BATCH步了就可以进行一次更新
                if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                    # 3.1 计算n步中最后一个state的v_s_
                    v_s_ = self.get_v(s_)
                    # 3.2 和PG一样，向后回溯计算。
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    # 所以这里的br并不是每个状态的reward，而是通过回溯计算的V值
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 3.3 N-step 历史经验更新网络 bs:batch*3 ba:batch*1 br:batch*1
                    self.update(bs, ba, br)
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * GAMMA + ep_r *(1-GAMMA))
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep, episode_max, ep_r, time.time() - t0))

            # 4. 画图
            np.save('./data/episode_reward.npy', all_ep_r)
            plt.ion()
            plt.cla()
            plt.title('PPO')
            plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            plt.ylim(-2000, 0)
            plt.xlabel('Episode')
            plt.ylabel('Moving averaged episode reward')
            plt.show()
            plt.pause(0.1)
        self.save_ckpt()
        plt.ioff()
        plt.show()

    def a_train(self, tfs, tfa, tfadv):
        '''
        更新策略网络(actor policy network)
        '''
        # 输入时s，a，td-error。这个和AC是类似的。
        tfs = np.array(tfs, np.float32)         #state
        tfa = np.array(tfa, np.float32)         #action
        tfadv = np.array(tfadv, np.float32)     #td-error


        with tf.GradientTape() as tape:

            # 我们需要从两个不同网络，构建两个正态分布pi，oldpi。
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下，同样输出a的概率的比值
            # 除以(oldpi.prob(tfa) + EPS)，其实就是做了import-sampling。怎么解释这里好呢
            # 本来我们是可以直接用pi.prob(tfa)去跟新的，但为了能够更新多次，我们需要除以(oldpi.prob(tfa) + EPS)。
            # 在AC或者PG，我们是以1,0作为更新目标，缩小动作概率到1or0的差距
            # 而PPO可以想作是，以oldpi.prob(tfa)出发，不断远离（增大or缩小）的过程。
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            # 这个的意义和带参数更新是一样的。
            surr = ratio * tfadv
            # 我们还不能让两个分布差异太大。
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            # PPO2：
            # 很直接，就是直接进行截断。
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(ratio * tfadv,  #surr
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        # actor loss 带权重更新 tfadv：N-step TD-error
        a_grad = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor.trainable_weights))
        if METHOD['name'] == 'kl_pen':
            return kl_mean


    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)


    def c_train(self, tfdc_r, s):
        '''
        更新Critic网络
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32) #tfdc_r可以理解为PG中就是G，通过回溯计算。只不过这PPO用TD而已。

        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v                  # 就是我们说的td-error
            closs = tf.reduce_mean(tf.square(advantage))

        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))


    def cal_adv(self, tfs, tfdc_r):
        '''
        计算advantage，也就是td-error
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)           # advantage = r - gamma * V(s_)
        return advantage.numpy()

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)
        # 1.更新产生数据策略的参数
        self.update_old_pi()
        # 2.计算TD-error r+gamma*V(s) - critic(s) --> y_label-y_predict
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful
        # update actor
        #### PPO1比较复杂:
        # 动态调整参数 adaptive KL penalty
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution

        #### PPO2比较简单，直接就进行a_train更新:
        # clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        # 更新 critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)


    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor.predict(s)                   # 通过actor计算出分布的mu和sigma
        pi = tfp.distributions.Normal(mu, sigma)    # 用mu和sigma构建正态分布
        a = tf.squeeze(pi.sample(1), axis=0)[0]     # 根据概率分布随机出动作
        return np.clip(a, -2, 2)                    # 最后sample动作，并进行裁剪。

    def get_v(self, s):
        '''
        计算value值。
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]  # 要和输入的形状对应。
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        self.critic.save("model/ppo_critic.h5")
        self.actor.save("model/ppo_actor.h5")
        self.actor_old.save("model/ppo_actor_old.h5")
    #
    # def load_ckpt(self):
    #     """
    #     load trained weights
    #     :return: None
    #     """
    #     tl.files.load_hdf5_to_weights_in_order('model/ppo_actor.hdf5', self.actor)
    #     tl.files.load_hdf5_to_weights_in_order('model/ppo_actor_old.hdf5', self.actor_old)
    #     tl.files.load_hdf5_to_weights_in_order('model/ppo_critic.hdf5', self.critic)