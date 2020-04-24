# 强化学习：策略梯度
## 1. [SPG](https://github.com/liuyandong1988/Policy_gradient/tree/master/SPG) 
使用随机策略梯度方法实现 CartPole Game。使用Monto Carlo的求解方法，重点在于使用折损的回报值，作为损失函数的权重，起到了评价策略梯度好坏的作用。缺点，训练周期长，容易陷入局部最优解。

## 2. [Actor-Critic](https://github.com/liuyandong1988/Policy_gradient/tree/master/Actor_Critic)
使用AC算法实现CartPole Game。Critic 作为评价动作好坏的网络，使用TD-error作为损失函数更新网络。Critic的网络输出作为Actor网络的损失函数权重系数。每一步都更新网络，提高了样本利用率，加速了训练速度。但每走一步就训练一次，过犹不及，有没有更好的办法呢？

## 3. [PPO](https://github.com/liuyandong1988/Policy_gradient/tree/master/PPO) 
PPO算法实现了Pendulum Game，克服了SPG算法一个完整的episode，模型的更新速度慢；A-C算法，每一步一更新（on-policy）采样效率低等缺点。使用了N-step回溯的思想，critic网络评价一段时期内action的好坏，并使用TD-error（N）作为损失函数更新critic网络。Actor网络使用了importance sampling的思想，把actor和old-actor样本采样差异作为actor的损失函数，并将critic的网络输出作为损失函数的权重。

## 4. [DDPG](https://github.com/liuyandong1988/Policy_gradient/tree/master/DDPG)
DDPG实现了Pendulum Game。DDPG采用了A-C架构，但和actor-critic在各自功能上完全不一样。DDPG的actor输出一个具体的动作，使用梯度上升方法训练网络，而不是带权重的损失函数；DDPG的critic输出Q-value，一个具体行为的评价，使用TD-error方法训练网络。

## 5. [Twin Delay DDPG (TD3)](https://github.com/liuyandong1988/Policy_gradient/tree/master/TD3)
TD3 玩Pendulum-v0游戏。TD3是改进版的DDPG：（1）增大了两个critic网络，克服了DDPG对行为Q值评估过高的缺点； （2） 延时更新actor网络，使网络寻优action过程变得更稳定； （3）critic更新Q值对行为加入噪声，target policy smoothing regularization，在一个区域中的行为，使得critic网络更稳健。 