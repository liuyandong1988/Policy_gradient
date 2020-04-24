import numpy as np
import matplotlib.pyplot as plt

actor_loss = np.load('al.npy')
critic_loss_1 = np.load('cl1.npy')
critic_loss_2 = np.load('cl2.npy')
reward = np.load('reward.npy')

plt.figure('reward')
plt.title('Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.plot(reward)
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(131)
ax.plot(actor_loss)
ax.set_title('Actor loss')
ax.set_xlabel('train step')
ax = fig2.add_subplot(132)
ax.plot(critic_loss_1)
ax.set_title('Critic network 1 loss')
ax.set_xlabel('train step')
ax = fig2.add_subplot(133)
ax.plot(critic_loss_2)
ax.set_title('Critic network 2 loss')
ax.set_xlabel('train step')
plt.show()
