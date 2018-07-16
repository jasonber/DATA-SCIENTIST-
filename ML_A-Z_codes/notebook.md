# part 1 :data preprocessing

**1、概念**

```python
pandas  
numpy  
matplotlib.pyplot  
sklearn.preprocessing.LableEncoder, OneHotEncoder, Imputer, StandardScaler
sklearn.
```

class 类 :  某一类事物 调取的就是类

object 对象 : 某一类事物中的具体实例，带有具体的参数 

method 方法 : 具体实例中的具体动作  通过类的提供的方法来构建一个对象（实例）

dummy variable: 适用于分类变量，方法是onehotencoder ，避免数字带来的大小、等级影响

Menko variable :？

Feature scaling:特征缩放 防止欧式距离带来的误差

##2、操作

1、

```python
import class   

object1 = class(parameter)    

method = object.fit(data1)   

object2 =  method.transform(data2)
```

 2、feature scaling: train and test 均适用train的 度量

## 3、疑问

dummy variable 是否需要scaling ?

是否需要scaling取决你的模型需要多好的解释。虽然scaling有相同的尺度，但是我们会失去dummy data的原始含义。因为dummy data之前只是用来表示分类的，scaling之后就无法表示原来的分类。

dependent variable 是否需要scaling？

按照数据的特点选择是需要scaling。表示分类的不用。其他的需要



# part 2 :regression

##section 4  simple linear regression SLR

**1、概念**

ordinary least square :最小二乘法

**2、操作**

画图

```python
plt.图的形式(x轴，y轴，颜色)
plt.plot(x,y, 颜色)在图上增加其他的内容
plt.title()
plt.xlabel()
plt.ylabel()
plt.show()显示上面的图片 也意味着这次作图结束
```

**3、疑问**

为什么dependent variable 不需要 scaling ？

## section 5 multiple linear regression

statmodels.formula.api

**3、疑问**

为什么X总是不能查看？

因为X中包含了多个类型的变量，它属于object

automatic backward elimination, len(x[0])

2、操作

a、fit的含义

```python
LabelEncoder
OneHotEncoder(categorical_features = 变量列index）
model.fit(data) # fit model(object) to the data
```

b、np.append的特殊用法

```
np.append(arr在哪个数据中插入, values插入什么样的值, axis) 在哪个数据的末尾插入什么样的值，append含义就是在末尾悬挂
```

c、先进行labelencoder 再进行onehotencoder

d、axis = 0 以line加入数据  axis =1 以column加入数据

e、获得模型的统计指标

```
衡量器（object） = sm.衡量方式（）.fit
衡量器.
```

## section 6 polynomial regression

1、不需要 feature scaling

***f、总结***

```
读取数据

预处理
    missing value
         imputer.fit_transform
    labelencoder onehotencoder
    train_test_split
    standardscale
            使用train的量纲
    feature select : BE FS Stepwise
            statmodel 
     model
            model.fit_transform
            model.fit
            moeld.predict
    plt
          plt.图表（x,y ,color）
          plt.plot(x,y ,color)
          plt.tilte()
          plt.xlabel()
          plt.xlabel()
          plt.show()

```

## section 7 SVR

1、框架

先列框架 再coding

```python
# SVR
# Import the libraries
import pandas as pd
import numpy as np

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

# spliting

# feature scaling

# fit SVR to the dataset

# create regressor

# predicting 

# visual the SVR results

import matplotlib.pyplot as plt
plt.scatter(X, y, c = 'green')
plt.show()
```

2、must  feature scaling 

## section 8 Decision Tree Regression

model 需要 x，y同维度

```python
X_grid = np.arange(min(x), max(x), step)
X_grid = X_grid.reshape((len(X_grid),1))
```

## section 9 Random Forest Regression 

.values:Return Series as ndarray or ndarray-like depending on the dtype

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.values.html?highlight=values#pandas.Series.values

不是树越多越平缓，而是在某个树的数目上，RF会收敛

这几中模型是啥？

linear regression simple linear  multiple linear

non-linear  regression  polynomial SVR DT RF

non-linear regression non-continuous regression

non-linear or non-continuous regression

## section 10 Evaluating Regression Models Performance

1、$$R^2$$是什么

$$SS_{rest} = \sum{(y_i - \hat{y_i})}^2$$  $$SS_{tot} = \sum{(y_i - y_{avg})}^2$$  $$R^2 = 1 - \frac{SS_{rest}}{SS_{tot}}=\frac{SS_{tot}-SS_{rest}}{SS_{tot}}$$

$$R^2\leq1$$ 越接近1说明预测值与真实值的差距越小，模型拟合的越好。可以为负，模型极其差。所以$$R^2$$ 越大越好

2、调整的$$R^2$$ 是什么

更为全面的指标

多元回归（多个x）。

 $$R^2$$不会下降是固定的，不能很好的指示模型的拟合度

调整$$R^2$$更为敏感，对样本数和X个数均敏感，因为增加了惩罚项

3、***如何区分线性非线性***

# Part 3 Classification

## section 11 classification

##section 12 Logistics Regression

1、visualising the training set results

```python
from matplotlib.colors import ListedColormap # this library colorize all the data points
X_set, y_set = X_test, y_test # get the features and the value of label. Do this for plotting.
# meshgrid get the frame of my coordinate
# X1 and X2 are the axes. Minus 1 and plus 1 make sure the coordinate don`t lost points
# step means the scale of axes. This makes sure we can get every pixel. 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
# contour function = contourf get the points between the two prediction regions with two features
# If the probility of point support 0 is red, else is green
# make the X of predictor to be matrix. This matrix must same as X_train.
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),  X2.ravel()]).T).reshape(X1.shape), alpha = 0.25, cmap = ListedColormap(('red', 'green')))
# limits the age and salary
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# use loop plot all points on the graph 
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend() # illustrate the colors of points
plt.show()
```

classification作图步骤：

a、设计坐标的框架 （两个轴的截距、格子的大小（step））。meshgrid mesh：网状  gird：格子 网状格子。  

b、设计图形的轮廓（点，透明度，各类点颜色（ListedColormap））。contourf  contour：轮廓   f：function

c、图形的大小

d、画上点

e、图例 legend

2、classification 模板

```python
# import libraries
# import dataset
# split dataset into train and test
from sklearn.cross_validation import train_test_split
# feature scaling
from sklearn.preprocessing import StandardScaler
# fit the model of train set
# predict the test set
# making confusion matrix
from sklearn.metrics import confusion_matrix
# visualising the result
from matplotlib.pyplot import plt
from matplotlib.colors import ListedColormap
np.meshgrid
plt.contorf
plt.legend()
```

3、ravel 和 flatten

首先声明两者所要实现的功能是一致的（将多维数组降位一维）。这点从两个单词的意也可以看出来，ravel(散开，解开)，flatten（变平）。两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。

4、什么时候需要特征缩放

## section 13 K - nearest -neighbors KNN

KNN 可选择L1 L2

线性与否 与图形有关

已经分好类别，随机抽取样本，计算样本中各类别比例，以此为出发点，找到最好的分类抽样。当有新样本出现时，按照训练集抽样的比例，判定新数据的类别。

##section 14 SVM support vector machine

linear 效果是不好。。。。。

有很多kernel

## section 15 Kernel SVM

Kernel 意味着landmark的选取方式和距离计算方式不同。分类方法K是否大于0

Kernel Trick指的就是 Kernel的通过landmark和距离分类的方法

Kernel ： RBF 高斯分布，sigmoid sigmoid函数， polynomial函数 多项式分布

 http://mlkernels.readthedocs.io/en/latest/

## section 16 Naive Bayes

取样方法很重要

## section 17 Decision Tree Classifier

```python
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.25, cmap = ListedColormap(('red', 'green')))

T1, T2 = X1.ravel(), X2.ravel() # 将matrix变成一维数组（横向量）
A1 = np.array([T1, T2]) # 将两个一维数组，拼成二维数组（2行）
A2 = A1.T               # 转置成2列
R = classifier.predict(A2)
R2 = R.reshape(X1.shape) # 调整为X1的矩阵样子，这样才能plot
R3 = R.reshape(X2.shape)
```

y只能是1维数组，否则plot会出错

保证数据的样子是一致的

## section 18 Random Forest 

Real-time Human Pose Recognition in parts Single Depth Images

np.arange

np.array

== 表示左右两个值是否相等，相等返回True，不等返回False

出现多余的分类，可判定为overfitting

## section 19 evaluating classificaiton performance

1、阳性、阴性：概率的投影。大于0.5为1， 小于0.5为0

false positive： 1 error，预测是阳性的但真实值是阴性的。 采纳错了

false negative： 2 error， 预测是阴性的但真实值是阳性。 拒绝错了

2、正确率 accuracy rate = correct / total

错误率 error rate = wrong / total

3、正确率悖论

数据不平衡：阳性和阴性的比例相差太多

4、CAP cumulative accuracy profile  

如果curve 接近平均概率（随机样本）的直线就是不好，离平均概率越远越好。如果在直线下方就是非常差

crystal ball

roc = Receiver Operating Characteristic 竟然不讲。。。

1) ROC(Receiver Operating Characteristic Curve):接受者操作特征曲线。ROC曲线及AUC系数主要用来检验模型对客户进行正确排序的能力。ROC曲线描述了在一定累计好客户比例下的累计坏客户的比例，模型的分别能力越强，ROC曲线越往左上角靠近。AUC系数表示ROC曲线下方的面积。AUC系数越高，模型的风险区分能力越强。  
2)CAP(Cumulative Accuracy Profile):累积准确曲线。CAP曲线及准确性比率/AR描绘了每个可能的点上累计违约排除百分比。为了画出CAP曲线，需要首先自高风险至低风险排列模型的分数，然后对于横坐标客户总数中特定的比例，CAP曲线的纵坐标描述风险评级分数小于或等于横坐标x中的违约个数百分比。一个有效的模型应当在样本客户处于同一排除率的情况下，排除更高百分比的坏客户。  

3)KS（Kolmogorov-Smirnov）检验:K－S检验主要是验证模型对违约对象的区分能力，通常是在模型预测全体样本的信用评分后，将全体样本按违约与非违约分为两部分，然后用KS统计量来检验这两组样本信用评分的分布是否有显著差异。

5、cap curve analysis

AR = model curve/ crystal curve

AR<60% rubbish     

60%<AR<70% poor    

70%<AR<80% good    

80%<AR<90% very good    

90%<AR<100% Too good  容易overfitting 

总结

1、利弊

2、模型选择

linear Logistic Regression、SVM。

non-linear K-NN ， Naive Bayes， Decision Tree， Random Forest。

使用概率预测 LR（线性） NB（非线性）  如何判断线性与否？   

多类别预测 SVM   

有清晰的解释（特征） DT

好的分类模型 RF

# part 4 clustering

对未知的知识，先用无监督模型进行聚类，打上标记，然后再用监督学习建立模型，去找到知识的规律

正好是两种认知模式，自上而下（k均值），自下而上（层聚类）

聚类与降维的异同？

## section 12 K-means

一、算法过程（由上到下）

1、选择簇的K个数  

2、随机选择K个中心，任意值不局限在数据中的点

3、选择个数据点到中心的距离  #这3项构建出了K个簇

4、计算新的中心

5、找距中心最近的点，直到不再计算新的中心。没有点在各个中心的中点线上时收敛。  

二、随机初始化陷阱

不好的初始点

K-mean ++算法 避免该陷阱

三、簇的个数 elbow method

降低 各个簇的MSE的和的和，WCSS组内平方和最小是的簇数就是最佳簇数。

碎石图 可以帮忙确定，类似降维

elbow method

四、模板

```python
# -*- coding: utf-8 -*-

# K-mean
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset 无分类 无因变量所以不用split
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# useing the elbow method find the optimal number of clusters 尝试不同的
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# split datasets 为什么不划分训练集和测试集
# feature scaling 在同一个数量级上
# fit model
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# predict
# confusion matrix
# visualising 只适用于二维数据 那三维的怎么办？
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') 
# what`s the meaning of y == 0. S is the size of dot
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()

# name the cluster 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
```

##section 22 Hierarchical Clustering

一、算法过程（由下到上）

1、将每个数据点看做一个簇

2、将距离最近的两个点合并为一个簇，剩下N-1个簇

3、将距离最近的两个簇（数据点）合并成一个簇，剩下N-2个簇

4、重复第3步，直到形成一个簇

簇之间的距离计算会影响算法 欧式距离

二、测量两个簇的距离：

1、最近的点的距离

2、最远的点的距离

3、两个簇中所有点的距离的平均距离

3、两个簇的中心的距离

三、dendrograms 密度图

记录HC聚类的过程

横轴为数据点、竖轴为数据点的距离距离

四、确定层数

簇距离的最后阈限

dissimilarity thresholds 不相似阈限

在最接近阈限的下方有几个竖线 就有几个簇。

阈限直线与最长距离的交点个数就是簇的个数

五、模板

```python
# -*- coding: utf-8 -*-

# hieracrchical clusetring
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# prprocessing
# split dataset
# feature scaling
# find the optimal number of clusters dendorgrams
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean distance')

# fit model
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

# predict
y_hc = hc.fit_predict(X)

# visualising
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 4')
plt.title('Cluster of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# name the cluster
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensibel')
plt.title('Cluster of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```



 # Part 5  Association Rule Learning

关联法则学习，买了这个东西那么又买那个。推荐系统

## section 4 Apriori learning

同时买两样东西

步骤: 朴素贝叶斯

1、设置最小支持和最小 confidence

support(M) = 用户的交易中包含某件商品M/所有交易（概率）

confidence(M1-M2) = 用户购买了M1又购买了M2/所有购买M1的交易（概率）

2、在所有交易中选取高于最小支持的子集合交易 

3、在所有子交易集合中选取高于最小confidence的confidence

4、通过降低lift 将规则分类

lift(M1-M2) = confidence(M1-M2)/support(M2) （值）

对lift降序得到分类结果。规则越强lift越高，关联性越强，所以就能推荐

```python
# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
results_list = []
for k in range(0, len(results)):
    results_list.append('Rule:\t' + str(results[i][0]) + '\nSupport:\t' + str(results[i][1]) +
                        '\nCONF:\t' + str(results[i][2][0][2]) + '\nLIFT:\t' + str(results[i][2][0][3]))
    
```

