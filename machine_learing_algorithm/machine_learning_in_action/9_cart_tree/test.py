import reg_tree
import numpy as np
import matplotlib.pyplot as plt
#
# test_matrix = np.mat(np.eye(4))
#
# mat0, mat1 = reg_tree.bin_split_dataset(test_matrix, 1, 0.5)
#
# dataset = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
#                                 "machine_learning_in_action/9_cart_tree/ex00.txt")
# dataset_matrix = np.mat(dataset)
# reg_tree.create_tree(dataset_matrix)
# dataset_x = []
# dataset_y = []
# for x, y in dataset:
#     dataset_x.append(x)
#     dataset_y.append(y)
#
# fig = plt.figure(figsize=(6, 8))
# plt.scatter(dataset_x, dataset_y)
# plt.show()
#
#
#
# dataset2 = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm"
#                                  "/machine_learning_in_action/9_cart_tree/ex0.txt")
# dataset2_matrix = np.mat(dataset2)
# my_tree = reg_tree.create_tree(dataset2_matrix, ops=(0,1))
#
# # dataset2_x = []
# # dataset2_y = []
# # for a,x, y in dataset2:
# #     dataset2_x.append(x)
# #     dataset2_y.append(y)
# # plt.scatter(dataset2_x, dataset_y)
#
# dataset3 = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
#                                  "machine_learning_in_action/9_cart_tree/ex2.txt")
#
# dataset3_matrix = np.mat(dataset3)
# tree = reg_tree.prune(my_tree, dataset3_matrix)
#
#
# model_data = reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/machine_learing_algorithm/"
#                                    "machine_learning_in_action/9_cart_tree/exp2.txt")
# model_data = np.mat(model_data)
# reg_tree.create_tree(model_data, leaf_type=reg_tree.model_leaf, err_type=reg_tree.model_error, ops=(1, 10))

# 对比回归树 与 模型树,使用判定系数R方对比
train_matrix = np.mat(reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                                            "machine_learing_algorithm/machine_learning_in_action/"
                                            "9_cart_tree/bikeSpeedVsIq_train.txt"))
test_matrix = np.mat(reg_tree.load_dataset("/home/zhangzhiliang/Documents/my_git/DATA-SCIENTIST-/"
                                           "machine_learing_algorithm/machine_learning_in_action/"
                                           "9_cart_tree/bikeSpeedVsIq_test.txt"))
reg = reg_tree.create_tree(train_matrix, ops=(1, 20))
y_reg = reg_tree.create_forecast(reg,test_matrix[:, 0],
                                 model_value=reg_tree.reg_tree_value)
reg_Rs = np.corrcoef(y_reg, test_matrix[:, 1], rowvar=False)[0, 1]


mod = reg_tree.create_tree(train_matrix, leaf_type=reg_tree.model_leaf,
                           err_type=reg_tree.model_error, ops=(1, 20))

y_mod = reg_tree.create_forecast(mod, test_matrix[:,0],
                                 model_value=reg_tree.model_tree_value)
mod_Rs = np.corrcoef(y_mod, test_matrix[:, 1], rowvar=False)[0, 1]


ws, X, Y = reg_tree.linear_model(train_matrix)
y_linear = y_reg.copy()
for i in range(np.shape(test_matrix)[0]):
    y_linear[i] = test_matrix[i, 0] * ws[1, 0] + ws[0, 0]
linear_Rs = np.corrcoef(y_linear, test_matrix[:, 1], rowvar=False)[0, 1]
print("reg_Rs is {}\nmod_Rs is {}\nlinear_Rs is {}".format(reg_Rs, mod_Rs, linear_Rs))

# 