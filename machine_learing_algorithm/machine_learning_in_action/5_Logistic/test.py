import log_reg
import numpy as np
# bath gradient ascend
data_array, label_matrix = log_reg.load_dataset()
theta = log_reg.bath_GA(data_array, label_matrix)
log_reg.decision_boundary(theta)

# stochastic GA
theta_s = log_reg.stochastic_GA(np.array(data_array), label_matrix)
log_reg.decision_boundary(theta_s)

# 改进后的随机梯度算法
theta_us = log_reg.sto_GA_update(np.array(data_array), label_matrix)
log_reg.decision_boundary(theta_us)

# 马病分类
log_reg.multiple_test()