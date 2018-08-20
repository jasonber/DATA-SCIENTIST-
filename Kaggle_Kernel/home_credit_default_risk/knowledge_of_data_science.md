# 数据探索

相关性：探索特征与目标的相关性，进行特征的简单选取

# [特征工程](https://www.cnblogs.com/wxquare/p/5484636.html)

https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html

特征构建：增加现有数据的特征

特征选择：选择最有影响的特征或降维得到特征

## 特征构建--多项式特征

做现有特征的平方以及交互相乘项

交互相乘是为了是多个特征变成一个特征，使得多个特征的影响合二为一

Adding interaction terms to a regression model can greatly expand understanding of the relationships among the variables in the model and allows more hypotheses to be tested.  

增加回归模型的交互项能极好模型中各变量之间的相关性。

# 特征构建--Domain Knowledge Features

领域知识特征：指的是数据所属领域的专业知识。需要有专业知识背景

# minmax归一化

线性归一化（minmax_scale） 
通俗地解释 ： 
归一化结果=该点样本值与最小样本的差/样本该轴跨度⋅放缩范围+放缩最小值

无法用于online learning

# 模型评估

## confusion matrix

找到FP TP FN TN 

列为实际值 行为预测值

https://www.zhihu.com/question/30643044

f1 

PRC

## ROC

fpr

tpr

# [不均衡数据处理](http://www.dataivy.cn/blog/3-4-%E8%A7%A3%E5%86%B3%E6%A0%B7%E6%9C%AC%E7%B1%BB%E5%88%AB%E5%88%86%E5%B8%83%E4%B8%8D%E5%9D%87%E8%A1%A1%E7%9A%84%E9%97%AE%E9%A2%98/)

class_weight 惩罚权重的方法

# [调参Hperpot](https://blog.csdn.net/gg_18826075157/article/details/78068086)







