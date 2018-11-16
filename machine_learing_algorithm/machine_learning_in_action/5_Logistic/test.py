import log_reg
data_array, label_matrix = log_reg.load_dataset()
theta = log_reg.grad_ascent(data_array, label_matrix)


log_reg.decision_boundary(theta)
