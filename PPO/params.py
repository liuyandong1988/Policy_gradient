S_DIM, A_DIM = 3, 1     # state dimension, action dimension
A_LR = 0.0001                   # learning rate for actor
C_LR = 0.0002                   # learning rate for critic
EP_LEN = 200                    # total number of steps for each episode
BATCH = 32                      # update batchsize
GAMMA = 0.95                     # reward discount

# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better  PPO2
][1]

A_UPDATE_STEPS = 10     # actor update steps
C_UPDATE_STEPS = 10     # critic update steps

EPS = 1e-8              # epsilon