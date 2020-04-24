RANDOMSEED = 1              # random seed
MAX_FRAMES = 40000  # total number of steps for training

MAX_STEPS = 200  # maximum number of steps for one episode
EXPLORE_STEPS = 500  # 500 for random action sampling in the beginning of training
BATCH_SIZE = 64  # udpate batchsize
UPDATE_ITR = 3  # repeated updates for single step

BUFFER_SIZE = 5e5  # size of replay buffer

C_LR = 3e-4  # q_net learning rate
A_LR = 3e-4  # policy_net learning rate
ACTOR_UPDATE_INTERVAL = 3  # delayed steps for updating the policy network and target networks