import numpy as np
import matplotlib.pyplot as plt

reward = np.load('episode_reward.npy')

plt.title('Episode reward')
plt.xlabel('episode')
plt.ylabel('reward')
plt.plot(reward)
plt.show()