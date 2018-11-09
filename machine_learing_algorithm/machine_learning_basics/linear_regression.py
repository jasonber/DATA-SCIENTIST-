# https://github.com/zotroneneis/machine_learning_basics/blob/master/linear_regression.ipynb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(123)

# dataset
X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.rand(500, 1)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.title("Dataset")
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 线性回归类
class LinearRegression(object):
    def __int__(self):
        pass

    def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100):
        """
        使用梯度下降算法
        """
        # 初始化参数
        n_samples, n_feature = X.shape
        self.weights = np.zeros(shape=(n_feature, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # 计算权重和偏差
            y_predict = np.dot(X, self.weights) + self.bias

            # 在训练集上计算损失
            cost = (1 / n_samples) * np.sum((y_predict - y) ** 2)
            costs.append(cost)

            if i % 100 == 0:
                print("Cost at iteration {}: {}".format(i, cost))

            # 计算偏微分 梯度
            dJ_dw = (2 / n_samples) * np.dot(X.T, (y_predict - y))
            dJ_db = (2 / n_samples) * np.sum((y_predict - y))
            # 更新参数
            self.weights = self.weights - learning_rate * dJ_dw
            self.bias = self.bias - learning_rate * dJ_db
        return self.weights, self.bias, costs

    def train_normal_equation(self, X, y):
        """
        使用正规方程
        """
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.bias = 0
        return self.weights, self.bias

    def predict(self, X):
        """
        返回预测值
        """
        return np.dot(X, self.weights) + self.bias


# 使用梯度下降训练模型
regressor = LinearRegression()
w_train, b_train, costs = regressor.train_gradient_descent(X_train, y_train, learning_rate=0.0005, n_iters=10000)

fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(10000), costs)
plt.title('Development of during training')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

# 测试模型
n_samples, _ = X_train.shape
n_samples_test, _ = X_test.shape
y_p_train = regressor.predict(X_train)
y_p_test = regressor.predict(X_test)
error_train = (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test = (1 / n_samples) * np.sum((y_p_test - y_test) ** 2)

print('Error on training set: {}'.format(np.round(error_train, 4)))
print('Error on test set: {}'.format(np.round(error_test, 4)))

# 使用正规方程训练
X_b_train = np.c_[np.ones((n_samples)), X_train]
X_b_test = np.c_[np.ones((n_samples_test)), X_test]

reg_normal = LinearRegression()
w_train_ne = reg_normal.train_normal_equation(X_b_train, y_train)

y_p_train_ne = reg_normal.predict(X_b_train)
y_p_test_ne = reg_normal.predict(X_b_test)

error_train_ne = (1 / n_samples) * np.sum((y_p_train_ne - y_train) ** 2)
error_test_ne = (1 / n_samples) * np.sum((y_p_test_ne - y_test) ** 2)

print("Error on training set: {}".format(np.round(error_train_ne, 4)))
print("Error on test set: {}".format(np.round(error_test_ne, 4)))

# 结果可视化测试集结果
fig = plt.figure(figsize= (8, 6))
plt.title("Data in blue, predictions for test in orange")
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.ylabel("First feature")
plt.xlabel("Second feature")
plt.show()
