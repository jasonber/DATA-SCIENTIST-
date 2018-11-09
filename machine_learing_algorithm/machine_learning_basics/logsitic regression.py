# https://github.com/zotroneneis/machine_learning_basics/blob/master/logistic_regression.ipynb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(123)

# dataset
X, y_true = make_blobs(n_samples=1000, centers=2)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.title("Dataset")
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()

# 将y变成列向量
y_true = y_true[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

class LogisticRegression:
    def __int__(self):
        pass

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def train(self, X, y_true, n_iters, learning_rate):
        """
        初始化参数
        :param X:
        :param y_true:
        :param n_iters:
        :param learning_rate:
        :return:
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # 计算权重和bias
            # 可以看做是用sigmoid函数作为激活函数
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # 计算cost
            cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1 - y_true) * (np.log(1 - y_predict)))

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y_true))
            db = (1 / n_samples) * np.sum(y_predict - y_true)

            # 更新参数
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            costs.append(cost)
            if i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
        return self.weights, self.bias, costs

    def predict(self, X):
        """
        对X类别的预测， 1 或 0
        :param self:
        :param X:
        :return:
        """
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

        return np.array(y_predict_labels)[:, np.newaxis]

# 训练模型
classifier = LogisticRegression()
w_train, b_train, costs = classifier.train(X_train, y_train, n_iters=600, learning_rate=0.009)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(600), costs)
plt.title("Development of cost over training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

# 测试模型
y_p_train = classifier.predict(X_train)
y_p_test = classifier.predict(X_test)

accuracy_train = 100 - np.mean(np.abs(y_p_train - y_train)) * 100
accuracy_test = 100 - np.mean(np.abs(y_p_test - y_test)) * 100

print("train accuracy:{}%".format(accuracy_train))
print("test accuracy:{}%".format(accuracy_test))