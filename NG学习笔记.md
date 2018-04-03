# 第一课程 初识机器学

## 1\\机器学习定义

A computer program is said to learn from experience E with respect to some task
T, and some performance measure P, if its performance on T, as measured by P,
improves with experience E.

计算机程序从经验E中学习解决某一任务T进行某一性能度量P,通过P测定在T上的表现因经验而提高.

计算机通过学习经验E,而学会了做某种任务T(预测,监督)或进行性能P的评估(分类,无监督)
.

通过P测定在T上的表现因E而提高.

可学习E,可完成任务T,可衡量P,

## 2\\监督学习

监督学习:给出了确定的算法,经验中有正确的答案,通过经验的学习,机器学会了怎么判断正确与错误.

数据的结构已知(自变量,因变量都已知),可以预测未来.人只需要教给电脑就行

由人控制的学习

线性回归 logistic 回归

## 3\\无监督学习(探索因素分析)

无监督学习:需要计算机自己去学习,没有明确的算法,让计算机去分析数据的结构.

数据的结构和作用未知(自变量,因变量未知),因此也不知道这些数据能做什么,也没有明确的算法.

由计算自己学习

聚类分析,鸡尾酒算法

# 第二节课 单变量线性回归

1. 什么是模型以及术语介绍

> 监督模型 ：在数据中已经给出了样本的正确答案，如线性回归

回归就是回归到现实的指标

> 术语：

> m= 训练集样本的个数

> x\`s=输入变量/特征变量

> y\`s=输出变量/目标变量

> (x, y)代表一个样本

> ($$x^{\left( i \right)},y^{\left( i \right)}$$)表示第几个样本

> **监督模型的工作流程：监督算法的思想**

假设函数：$$h_{\varnothing} = \varnothing_{0} + \varnothing_{1}x + \ldots +\varnothing_{n}x$$

参数 Parameters：$$\varnothing_{i}$$

1. 代价函数 cost function

> 代价函数：$$J\left( \varnothing_{0},\varnothing_{1}\ldots\varnothing_{n}\right) = \frac{1}{2m}\sum_{i = 1}^{m}{(h_{\varnothing}(x^{i}) -y^{i})}^{2}$$,
> 所谓代价指的是假设函数对真实值的预测的距离（偏差），所以要使它最小化minimize
> J

> 要想使假设函数有用，就要最小化代价函数，即对代价函数的优化
> 目标函数的定义和意义？

> 在三维图中，J的值就是点的高度，也就是高度代表这一组参数的代价也就是偏差

![](C:/Users/Administrator/Desktop/media/6ae1dfc05c5145c8755c5752cc0f692d.png)

![](C:/Users/Administrator/Desktop/media/6181ff17dcde9e652373ee9a0f09a062.png)

1. 梯度下降：如何最小化代价函数

> 类比如何下山

**思想：**把参数们都放入假设函数中，进行不同的组合，找出代价函数小的那组参数或局部最小

参数们一般初始化为0

![](C:/Users/Administrator/Desktop/media/fa8d526bf4622da2bf995d37e693a62d.png)

如何实现梯度下降（算法）：

一直重复，直到收敛convergence：$$\varnothing_{j}\ : = \varnothing_{j} -\alpha\frac{\partial}{\partial\varnothing_{j}}J(\varnothing_{0},\varnothing_{1})$$，

标识的含义：

“： =”代表将运算结果放回到参数中，再进入下一次迭代，也就是所谓的重复

α：学习率，控制梯度下降时迈出的步子的大小，也就是下山的速度

后半部分是一个倒数

**注意的问题：**

要同时更新所有参数，同时算完同一组参数，参能进入下一次迭代

**凸函数是什么？batch不太懂**

# 第三课 矩阵 向量 及其运算

1. 矩阵和向量

> 矩阵matrix

> A=$$\begin{matrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \\ \end{matrix}$$

   矩阵A是一个3（行）\*2（列）的矩阵，$$A_{\text{ij}}$$=第i行，第j列的值 如A12

> = 4，A13是错误的

> 向量vector

> 是矩阵的一种，即1\*n的矩阵，即n维向量

1. 加法和标量乘法

> 对应的值相加，需要n 、m相同，否则会出现错误

> 矩阵中的每个值与标量相乘，n、m与矩阵的相同，满足乘法交换律

1. 矩阵和向量相乘

> 矩阵m\*n的每行与向量n相乘的和，即为结果，然后组成m维的向量

> 在模型中的应用

> 可以把自变量x看成是$$\begin{bmatrix} 1 & x_{1} \\ 1 & x_{n} \\\end{bmatrix}$$n\*2的矩阵，参数看作是向量$$\begin{bmatrix}\varnothing_{1}\\ \varnothing_{n} \\\end{bmatrix}$$，相乘即得到因变量y的向量$$\begin{bmatrix} y_{1} \\ y_{n} \\\end{bmatrix}$$为n维

1. 矩阵乘法

> 矩阵A m\*n与矩阵B
> n\*o相乘，将B看成是o个向量，与A相乘，然后将相乘的结果放入的m\*o的矩阵中

> A的“列”必须与B的“行”相同

> 在模型中的应用：可以与多组参数组成的矩阵相乘，算出多组y的矩阵

1. 矩阵的乘法律

> **切记不符合乘法交互律**

> 符合加法交互律

> 符合乘法集合律A\*B\*C=A\*（B\*C）=（A\*B）\*C

> 单位矩阵 identity matrix ：即A\*I=I\*A=A，I就是单位矩阵，作用同实数1

$$
\begin{matrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{matrix}
$$

> 记住A的列与I的行相同，A m\*n 的单位矩阵是 I n\*n

1. 逆和转置

> 矩阵的逆 inverse 类似实数的倒数，即$$A*A^{- 1} = I$$，符合乘法交换律

> 转置Transpose 将n\*m的矩阵 变换成 n\*m的矩阵

$$
B = A^{T},B_{\text{ij}} = A_{\text{ji}}
$$

> 奇异矩阵:没有逆矩阵的矩阵

> 凸函数：梯度下降生成的图形

# 第五课 多变量线性回归

1. 多功能

> 有多个属性特征（x的形式），

> n = 特征的个数

> $$x^{i}$$= 训练样本的第i个特征（向量），代表一个索引（行的号才能称为索引）

> $$x_{j}^{i}$$= 训练样本的第i个特征的第j个值

多变量线性回归的简写 **内积是什么？**

$$h_{\varnothing}\left( x \right) = \varnothing_{0} + \ldots +\varnothing_{n}x_{n} = \begin{bmatrix} \varnothing_{0} \\ \varnothing_{1} \\\ldots \\ \varnothing_{n} \\ \end{bmatrix}^{T}\ \begin{bmatrix} x_{0} \\ x_{1}\\ \ldots \\ x_{n} \\ \end{bmatrix} = \varnothing^{T}X$$

也叫多元回归方程，多元指的是多属性 多变量（x）。

1. 多元回归方程的梯度下降

> 代价函数: $$J\left( \varnothing_{0},\ \varnothing_{1},\cdots\varnothing_{n} \right) = J\left( \varnothing \right) =\frac{1}{2m}\sum_{i = 1}^{m}{(h_{\varnothing}\left( x^{i} \right) -y^{i})}^{2}$$

> 梯度下降：(n\>=1) for｛

> $$\varnothing_{j}\ : = \ \varnothing_{j} -\alpha\frac{\partial}{\partial\varnothing_{j}}J(\varnothing)$$

> $$\varnothing_{j}\ : = \ \varnothing_{j} - \alpha\frac{\partial}{\partial\varnothing_{j}}J(\varnothing)x_{j}^{i}$$

> }同时调整所有参数

##3、多元梯度下降演练

特征缩放 feature scaling: 使得特征的**量纲**一致（范围）

**不需要特精确，只是为了让梯度下降更快**

例子：

$x_1 = size(0-2000 feet^2)$

$x_2 = number\ of\ bedroom(1-5)$

缩放后：

$x_1 = \frac{size(feet^2)}{2000}$

$x_2 = \frac{number\ of\ bedrooms}{5}$

不一致的量纲会使得凸函数变得陡峭，也就是梯度下降的距离会比较长。缩放后，凸函数就会变成圆形，这样梯度下降的距离就会变小。

思想：

使的每一个特征的取值范围在$-1<=x_i<=1$,其实没有必要限定在这个区间，相差不大即可（不能太大，不能太小）在-3到3，$-\frac{1}{3}$到$\frac{1}{3}$

归一化mean normalization ：归一的是**均值**，使得特征的均值归一为0，特征值减去均值除以全距

 例子：

$x_1 = \frac{size-1000}{2000}$

$x_2 = \frac{bedrooms-2}{5}$

正态化：均一化的一种，$x_1 = \frac{x_1-\mu_1}{s_1}$

## 4、如何选择学习率

调参的一种

$\theta_j := \theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)x_j^i$

$\alpha$是学习率 learning rate

![52172888389](C:\Users\ADMINI~1\AppData\Local\Temp\1521728883894.png)300之后就收敛了。判断是否正常下降的标准，1、图中曲线是否正常收敛（常用，靠谱），2、阀值的确定。

学习率过大 ：优点下降速度快，缺点无法收敛。

学习率过小：优点保证能够收敛，缺点下降速度慢

学习率的选择$\alpha$: 0.001, 0.01, 0.1, 1,或者0.001, 0.003, 0.01, 0.03, 0.1， 0.3， 1 

## 5、特征和多项回归

多项回归指的是多项式，即特征的运算方式不同

例子：$y = \theta_0 + \theta_1 x_1+\theta_2 x_2^2+\cdots +\theta_n x_n^p$

多项式的出现使得假设函数的形状发生变化，目的是更好的拟合数据。如何选择形状与业务有关，

即与如何看待特征有关

记得特征缩放，使得特征值有可比性

##6、正规方程

$J(\theta) = a\theta^2 + b\theta +c$ 对其求导$\frac{\partial} {\partial\theta}$ ,并令这个导数为0即 $\frac{\partial} {\partial\theta} = 0$ 这样求出的$\theta$就是使得数最小的值

对代价函数适用，对每个$\theta$求其为0的偏导就能获得使代价函数最小的$\theta$值

例子：有m个样本$((x^1, y^1),\cdots, (x^m, y^m))$; n个特征

$x_i = \begin{bmatrix} x_0^i\\ x_1^i\\ x_2^i\\ \vdots\\ x_n^i\\ \end{bmatrix}\in R^{n+1}$，n+1是因为$x_0^i$是我们自己添加的， 

设计矩阵design matrix:$X = \begin{bmatrix}{(x^1)}^T \\ \vdots\\  {(x^m)}^T \end{bmatrix}$ , 对应到多元回归方程中，令向量$x_0^i$等于1，对应的参数为$\theta_0$。

以一元回归方程为例那么特征向量$x^i=\begin{bmatrix}1\\ x_1^i\\ \end{bmatrix}$的设计矩阵为$x_i^T=X=\begin{bmatrix} 1&x_1^1\\ \vdots&\vdots\\ 1&x_m^1\\ \end{bmatrix}$然后$X\theta=y=\begin{bmatrix} y_1\\ y_2\\ \vdots\\ y_m \end{bmatrix}$

所以，$\theta = \frac{y}{X} =X^{-1}y={(X^TX)}^{-1}X^Ty$

T代表转置，-1代表逆矩阵

何时使用正则方程

优点：与梯度下降相比，1、不需要特征缩放，2、不用迭代，一步就能得到结果，3、不需要选择学习率$\alpha$

缺点：算法复杂度$O(n^3)$，当特征数过大时（n>=10000）会变慢。

结论：n过大时使用梯度下降，小的时候使用正则，以10000个特征为界限

## 7、正规方程不可逆的情况

octave中使用伪逆矩阵来解决，pinv

1、减少共线性特征一个，

共线性是指特征的相关性过高

2、特征过多，m小n，样本数少于特征数

删除特征或正则化特征

# 第七课 logistic 回归

凸函数 非凸函数 对数函数的图形 极大似然法 之前梯度下降中的算式可能有问题

**补充理解：**

梯度下降:不理解偏导的运算需要补充

代价函数：$J(\theta)=\frac{1}{m}\sum_{i=1}^m{[\frac{1}{2}(h_\theta(x^i)-y^i)^2]}$ 

梯度下降：$\theta_j := \theta_i - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$, 

其中$\frac{\partial}{\partial\theta_j}J(\theta)$的意思是 对$J(\theta)=\frac{1}{m}\sum_{i=1}^m{(\frac{1}{2}(h_\theta(x^i)-y^i)^2)}$ 求偏导，

即：

$\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}((h_\theta(x^i)-y^i)x_j^i)$

所以梯度下降的表达式展开为：

$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}((h_\theta(x^i)-y^i)x_j^i)$  for j in features=[0,...,j]

## 1、分类

最简单的分类$y\in \{0,1\}$  ，而分类只有1和0两类，其中

”0“：negative class 负类 ，没有我们要找的东西

”1“：positive class 正类，有我们要找的东西

但正负的含义由我们决定，它只是个分类符号。

为什么不能使用线性回归来分类？

classification：y=0 或 y=1

线性回归的假设：$h_\theta(x) >1 或<0$, 不能等于1或0，即使y都是 0 或1 时，$h_\theta(x)$也不能等于1或0, 所以线性回归是不合适的

logistic 回归却能使假设函数的值等于1 或等于0

##2、logistic 回归的假设

logistic回归与线性回归的假设函数



线性回归模型：$h_\theta(x)$ 是任意连续型数据， $h_\theta(x) = 1*\theta_0 + \theta_1x_1+\cdots+\theta_nx_n$ 

模型解释 ：当输入x时，对y值的预测

![](/home/zhangzhiliang/Pictures/snapshot-1-1.png)



logistic回归模型 ：$0\leq h_\theta(x) \leq1$ ， 

$h_\theta(x) = g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$   ,其中

 $g(z)=\frac{1}{1+e^{-z}}$ 被称为sigmoid函数，也叫logistic函数，$\theta^Tx$可以理解为是线性回归的$h_\theta$

模型解释：当输入为x时，输出为y=0，或y=1的概率的预测。$h_\theta(x)=P(y=1|x; \theta)$ ,在x和$\theta$的情况下， y=1的概率为$h_\theta(x)$ 。

![](/home/zhangzhiliang/Pictures/snapshot-1.png)

IF $x=\begin{bmatrix} x_0 \\ x_1\\ \end{bmatrix} = \begin{bmatrix} 1 \\ {tumor size}\\ \end{bmatrix}$，$h_\theta(x)=0.7$，说明病人有70%的概率是恶性肿瘤

## 3、决策界限

决定y是1还是0的那个界限，这是函数（模型）的特性而不是数据集的属性。

当我们规定$h_\theta(x)\geq 0.5 \implies y=1$ ，即$\frac{1}{1+e^{-{\theta^Tx}}} \geq 0.5 \implies y =1$  ，所以$1+e^{-{\theta^Tx}} \leq 2 \implies y=1$，所以 $\theta^Tx \geq 0 \implies y=1$ ，所以$\theta_1x_1 + \theta_2 x_2 +\cdots+\theta_nx_n \geq \theta_0 \implies y=1$那么$\theta_1x_1+\cdots+\theta_nx_n = \theta_0$这条***直线就是决策边界***decision boundary

决策边界不一定都是直线，它是由$\theta$决定的

## 4、代价函数

代价函数也称优化目标、目标函数

logistic回归的模型定义

训练集：$\lbrace(x^1, y^1),(x^2, y^2),\cdots,(x^m,y^m)\rbrace$

m个样本：$x\in\begin{bmatrix} x_0\\ x_1\\ \cdots\\ x_n\\ \end{bmatrix}$ $x_0 =1, y\in\lbrace1,0\rbrace$ 

假设函数：$h_\theta(x) = \frac{1}{1+e^{-{\theta^Tx}}}$ 

代价函数：

线性回归:$J(\theta)=\frac{1}{m}\sum_{i=1}^m\frac{1}{2}((h_\theta(x^i)-y^i))^2=\frac{1}{m}\sum_{i=1}^m (Cost(h_\theta(x^i)-y^i))$

$Cost(h_\theta(x^i), y^i) = \frac{1}{2}(h_\theta(x^i)-y^i)^2$  ，在线性回归中$J(\theta)$是凸函数有全局最小值

logistic 回归再使用上面这个代价函数，会因为$h_\theta$的不同而使得代价函数$J(\theta)$变为非凸函数non-convex，会产生局部最小值,所以logistic 回归的代价函数：

$J(\theta) = \frac{1}{m}\sum_{i=1}{m}Cost(h_\theta(x^i), y^i)$

$Cost(h_\theta(x),y)=\begin{Bmatrix} -\log(h_\theta(x)) \quad if \quad y=1\\ -log(1-h_\theta(x))  \quad if \quad y=0\\ \end{Bmatrix}$ **y始终等于1或0** 

简化：$Cost(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)log(1-h_\theta(x))$ 所以logistic回归的代价函数为：

$J(\theta)= \frac{1}{m}Cost(h_\theta(x^i), y^i) = -\frac{1}{m} [\sum_{i=1}^m {y^i\log(h_\theta(x^i))+(1-y^i)\log(1-h_\theta(x^i))}]$

这就是**极大似然估计 max likelihood estimation**：极大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是已知某个随机样本满足某种[概率分布](https://baike.baidu.com/item/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83)，但是其中具体的参数不清楚，[参数估计](https://baike.baidu.com/item/%E5%8F%82%E6%95%B0%E4%BC%B0%E8%AE%A1)就是通过若干次试验，观察其结果，利用结果推出参数的大概值。极大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率最大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值。

## 5、梯度下降

$minJ(\theta)=-\frac{1}{m}[\sum_{i=1}^{m} {y^i\log(h_\theta(x^i))+(1-y^i)\log(1-h_\theta(x^i))]}$

for j in [0,......,n] : {    $\theta_j=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)$  } 切记n个参数要同时更新

$\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{[h_\theta(x^i)-y^i)x_j^i]}$是对$J(\theta)$的偏导

与线性回归的公式一样，但是它们不是同一个算法，因为假设函数发生了变化

线性回归 $h_\theta(x) = \theta^TX$

logistic回归$h_\theta(x) = \frac{1}{1+e^{-\theta^TX}}$ 

除了以上的梯度下降还可以使用正则法（向量化）来找到参数

## 6、高级优化

conjugate gradient、BFGS、 L-BFGS 

优点：1不需要设置学习率2速度快

缺点：太复杂

不用担心直接调包就好～～

## 7、多元logistic回归

分解为多个二元分类问题

目标为1，即为正。非目标为0,即为负。有几个分类就有几个分类器

![](/home/zhangzhiliang/Pictures/one_vs_all.png)

![](/home/zhangzhiliang/Pictures/one_vs_all-1.png)

# 第8章 正则化

正则化 与正规方程

正则是为了限制模型复杂度，防止过拟合

正规方程是一种求最小化代价函数的方法



方差 偏差的概念

https://www.zhihu.com/question/27068705

偏差：描述的是模型预测的准不准，效度

方差：描述的是模型的稳定性，信度

## 1、过拟合

定义：如果我们有太多的特征，学习过的假设函数可能在训练集上的拟合很好，也就是$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}[(h_\theta(x)^{(i)}-y^{(i)})^2]\approx 0$ 

，但是假设函数不能很好的泛化“generalize“到非训练集上。

**过拟合的原因**：过多的变量，很少的训练数据，复杂的模型就会导致过拟合

**解决办法**：

1、减少特征的数量：人工选择特征，模型选择算法计算出保留哪些特征。缺陷：损失一些特征

2、正则化（Regularization）: 通过减少参数$\theta_j$保留所有特征对y有任何一点影响的特征

## 2、正则化的代价函数

惩罚项penalize：使得复杂模型中参数变得非常小，惩罚$\theta_j$带来的影响

正则化：最小所有化参数，减少过拟合的可能性

优化目标：$min J(\theta) = \frac{1}{m}\sum_{i=1}^{m}{\frac{1}{2}[(h_\theta(x)^{i}-y^{(i)})^2]}$ 

正则化的代价函数：$J(\theta) = \frac{1}{2m}[\sum_{i=1}^{m}{(h_\theta(x)^{(i)}-y^{(i)})^2}]+\lambda\sum_{i=1}^{n}\theta_j^2$ ，切记不需要从$\theta_0$开始

$\lambda$正则化参数的两个目标：1、保证***$Cost(h_\theta(x)^2,y^2)$***的值最小更好的拟合。2、保证参数最小。通过这两个目标来减少模型的复杂度，避免过拟合

为什么正则项$\lambda\sum_{i=1}^{n}{\theta_j^2}$会起到这两个作用，因为要$min J(\theta)$所以正则项就要接近0,而$\lambda$是我们规定好的正则化参数（常数），所以$\theta$们的和就要接近0。因此达到以上两个目标。

惩罚项$\lambda\sum_{i=1}^{n}{\theta_j^2}$太大的话，也就是惩罚太大的化，假设函数的参数就会接近0,假设函数$h_\theta(x)=\theta_0$就会变成一条直线，就会欠拟合。所以我们要想选择学习率$\alpha$一样，去选择正则参数$\lambda$。

## 3、线性回归的正则化

在梯度下降中加入正则项：

for j in [1....n] 不从0开始了，因为正则化从$\theta_1$开始：

{

​    $\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}{[(h_\theta(x)^{(i)}-y^{(i)}) x_0^{(i)}]}=\theta_0-\alpha \frac{1}{m} \frac{\partial}{\partial\theta_0}J(\theta)$

   $\theta_j := \theta_1 -\alpha[(\frac{1}{m}\sum_{i=1}^{m}{[(h_\theta(x)^{(i)}-y{(i)})x_j^{(i)} )+ \lambda\frac{1}{m}\theta_j}] $

   $\theta_j := \theta_j(1-\alpha\lambda\frac{1}{m})-\alpha\frac{1}{m}\sum_{i=1}{m}[{(h_\theta^{(i)}-y^{(i)}x_j^{(i)}}]=\theta_j(1-\alpha\lambda\frac{1}{m})-\alpha\frac{1}{m}\frac{\partial}{\partial\theta_j}J(\theta)$

}

 与非正则化的梯度下降相比，因为$1-\alpha\lambda\frac{1}{m} < 1$，所以$\theta_j$会先变小一些，在减去$J(\theta)$的偏导，就能更好的下最小值下降和收敛

正规方程中的正则化

正规方程：$\theta = (X^TX)^{-1}X^Ty$ 

正则化的正规方程：

假设： $m(examples)\leq n(features)$ , $\theta = (\theta^TX)^{-1}X^Ty$

如果$\lambda > 0$

​         $\theta = (X^TX+\lambda\begin{bmatrix} 0 &\cdots &0 & \cdots & 0\\ 0 & 1 & 0 & \cdots\\ 0 & 0 & 1 & 0 & \cdots \end{bmatrix})X^Ty$

矩阵为$(n+1)*(n+1)$

## 4 、logistic回归

代价函数：

$$J(\theta) = -[\frac{1}{m}\sum_{i=1}^{m}{y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log (1-h_\theta(x^{(i)}))}]$$ 

正则化的代价函数：

$J(\theta) = -[\frac{1}{m}\sum_{i=1}^{m}{y^{(i)} \log h_\theta(x^{(i)})+(1-y^{(i)})\log (1-h_theta(x^{(i)}))}] + \lambda\frac{1}{2m}\sum_{i=1}^{m}{\theta_j^2}$

梯度下降：

for j in [1....n] {

$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}{(h_\theta(x{(i)}-y^{(i)})x_0^{(i)})}$

$\theta_j := \theta_j - \alpha[(\frac{1}{m}\sum_{i=1}^{m}{(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\lambda\frac{1}{m}\theta_j}]$

}