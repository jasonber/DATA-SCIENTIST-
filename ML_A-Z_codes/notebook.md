# section 2  part 1 :data preprocessing

##1、概念

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