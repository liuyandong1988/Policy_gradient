# 强化学习：策略梯度
## 1.[SPG][https://github.com/liuyandong1988/Policy_gradient/tree/master/SPG] 
使用随机策略梯度方法实现 CartPole Game。使用Monto Carlo的求解方法，重点在于使用折损的回报值，作为损失函数的权重，起到了评价策略梯度好坏的作用。缺点，训练周期长，容易陷入局部最优解。