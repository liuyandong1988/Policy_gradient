import os

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from DRL import DRL


class AC(DRL):
    """Actor Critic Algorithms with sparse action.
    """

    def __init__(self):
        super(AC, self).__init__()
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.gamma = 0.99

    def load(self):
        if os.path.exists('model/actor_acs.h5') and os.path.exists('model/critic_acs.h5'):
            self.actor.load_weights('model/actor_acs.h5')
            self.critic.load_weights('model/critic_acs.h5')

    def _build_actor(self):
        """
        actor model.
        """
        inputs = Input(shape=(4,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.summary()
        return model

    def _build_critic(self):
        """
        critic model.
        """
        inputs = Input(shape=(4,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.summary()
        return model

    def _actor_loss(self, y_true, y_pred):
        """actor loss function.
        Arguments:
            y_true: (action, reward)
            y_pred: action_prob
        Returns:
            loss: reward loss
        """
        action_pred = y_pred
        action_true, td_error = y_true[:, 0], y_true[:, 1]
        action_true = K.reshape(action_true, (-1, 1))
        loss = K.binary_crossentropy(action_true, action_pred)
        loss = loss * K.flatten(td_error)
        return loss

    def discount_reward(self, next_states, reward, done):
        """Discount reward for Critic
        Arguments:
            next_states: next_states
            rewards: reward of last action.
            done: if game done.
        """
        v = self.critic.predict(next_states)[0][0]
        target = reward
        if not done:
            target = reward + self.gamma * v
        return target

    def train(self, episode):
        """training model.
        Arguments:
            episode: ganme episode
        Returns:
            history: training history
        """
        self.actor.compile(loss=self._actor_loss, optimizer=Adam(lr=0.001))
        self.critic.compile(loss='mse', optimizer=Adam(lr=0.01))

        history = {'episode': [], 'Episode_reward': [],
                   'actor_loss': [], 'critic_loss': []}

        for i in range(episode):
            observation = self.env.reset()
            rewards = []
            alosses = []
            closses = []

            while True:
                x = observation.reshape(-1, 4)
                # 1. 依概率action choice action with prob.
                prob = self.actor.predict(x)[0][0]
                action = np.random.choice(np.array(range(2)), p=[1 - prob, prob])
                # 2. 与环境交互
                next_observation, reward, done, _ = self.env.step(action)
                next_observation = next_observation.reshape(-1, 4)
                rewards.append(reward)
                # 3. Critic TD-error： R+V(s_next)-V(s)
                target = self.discount_reward(next_observation, reward, done)
                y = np.array([target])
                # TD_error = (r + gamma * next_v) - current_v
                td_error = target - self.critic.predict(x)[0][0]
                critic_loss = self.critic.train_on_batch(x, y)
                # 4. Actor Policy Gradient
                y = np.array([[action, td_error]])
                actor_loss = self.actor.train_on_batch(x, y)
                observation = next_observation[0]
                alosses.append(actor_loss)
                closses.append(critic_loss)
                # 5. 完成一个episode
                if done:
                    episode_reward = sum(rewards)
                    aloss = np.mean(alosses)
                    closs = np.mean(closses)

                    history['episode'].append(i)
                    history['Episode_reward'].append(episode_reward)
                    history['actor_loss'].append(aloss)
                    history['critic_loss'].append(closs)

                    print('Episode: {} | Episode reward: {} | actor_loss: {:.3f} | critic_loss: {:.3f}'.format(i,
                                                                                                               episode_reward,
                                                                                                               aloss,
                                                                                                               closs))

                    break

        self.actor.save_weights('model/actor_acs.h5')
        self.critic.save_weights('model/critic_acs.h5')

        return history


if __name__ == '__main__':
    model = AC()
    history = model.train(200)
    model.save_history(history, 'ac_sparse.csv')
    model.load()
    model.play('acs')