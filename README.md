# 强化学习：策略梯度
## 1. [SPG](https://github.com/liuyandong1988/Policy_gradient/tree/master/SPG) 
使用随机策略梯度方法实现 CartPole Game。使用Monto Carlo的求解方法，重点在于使用折损的回报值，作为损失函数的权重，起到了评价策略梯度好坏的作用。缺点，训练周期长，容易陷入局部最优解。

## 2. [Actor-Critic](https://github.com/liuyandong1988/Policy_gradient/tree/master/Actor_Critic)
使用AC算法实现CartPole Game。Critic 作为评价动作好坏的网络，使用TD-error作为损失函数更新网络。Critic的网络输出作为Actor网络的损失函数权重系数。每一步都更新网络，提高了样本利用率，加速了训练速度。但每走一步就训练一次，过犹不及，有没有更好的办法呢？