import numpy as np
import matplotlib.pyplot as plt

actor_loss = np.load('aloss.npy')
critic_loss = np.load('closs.npy')
reward = np.load('reward.npy')


# reward
plt.figure('reward')
plt.title('Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.plot(reward)
plt.show()

# loss
fig2 = plt.figure(2)
ax = fig2.add_subplot(121)
ax.plot(actor_loss)
ax.set_title('Actor loss')
ax.set_xlabel('episode')
ax = fig2.add_subplot(122)
ax.plot(critic_loss)
ax.set_title('Critic loss')
ax.set_xlabel('episode')
plt.show()
