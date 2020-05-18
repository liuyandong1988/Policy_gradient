import gym
from DDPG import DDPG
import numpy as np

def play(env, actor):
    """
    play game with model.
    """
    for _ in range(3):
        print('play...')
        observation = env.reset()
        reward_sum = 0
        while True:
            env.render()
            action = actor(np.array([observation], dtype=np.float32))[0]
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                break

    input('Game over !')
    env.close()

if __name__ == '__main__':
    # 1.环境
    env = gym.make('Pendulum-v0')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    # 2.模型
    ddpg = DDPG(env, s_dim, a_dim, a_bound)
    # 3.加载参数
    ddpg.load_ckpt()
    play(env, ddpg.actor)
