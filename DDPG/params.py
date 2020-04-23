RANDOMSEED = 1              # random seed
LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic


GAMMA = 0.95                 # reward discount
TAU = 0.01                  # soft replacement

MAX_EPISODES = 500          # total number of episodes for training
MAX_EP_STEPS = 200          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes

VAR = 3                     # control exploration
MEMORY_CAPACITY = 10000     # size of replay buffer
BATCH_SIZE = 128