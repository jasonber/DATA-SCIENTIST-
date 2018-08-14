# 电影推荐系统

## 1.探索数据

1.1关键词  

1.2 缺失值处理  

1.3 电影年份  

1.4 电影类别

##2.清洗数据

2.1 清洗关键词

   2.1.1 通过root分组  

   2.1.2 同义词的组  

2.2 相关性分析  

2.3 缺失值  

​    2.3.1 年份缺失值  

​    2.3.2 提取电影名关键词  

​    2.3.3 回归填充缺失值  

## 3. 推荐系统

3.1 推荐系统的基本函数  

​    3.1.1 相似度  

​    3.1.2 流行度  

3.2 推荐系统  

3.3 有意义的推荐  

3.4 测试推荐系统  

## 4. 结论

   



# 库

## 1. [json](http://www.runoob.com/python/python-json.html)

json.dumps 用于将 Python 对象编码成 JSON 字符串。

json.loads 用于解码 JSON 数据。该函数返回 Python 字段的数据类型。

JSON(JavaScript Object Notation) 是一种轻量级的数据交换格局。它基于ECMAScript的一个子集。 JSON选用完全独立于言语的文本格局，但是也使用了类似于C言语宗族的习气（包含C、C++、C#、Java、JavaScript、Perl、Python等）。这些特性使json调试成为抱负的数据交换言语。 易于人阅览和编写，同时也易于机器解析和生成(一般用于提高网络传输速率)。 

 JSON  VS   XML：
    1.JSON和XML的数据可读性根本相同
    2.JSON和XML相同具有丰厚的解析手法
    3.JSON相对于XML来讲，数据的体积小
    4.JSON与JavaScript的交互愈加方便
    5.JSON对数据的描述性比XML较差
    6.JSON的速度要远远快于XML

**在这里只用到了loads，是因为数据来自于TMDB，属于网页信息，要转换成python的数据类型。**

## 2. nltk

NLTK 模块是一个巨大的工具包，目的是在整个自然语言处理（NLP）方法上帮助您。 NLTK 将为您提供一切，从将段落拆分为句子，拆分词语，识别这些词语的词性，高亮主题，甚至帮助您的机器了解文本关于什么。在这个系列中，我们将要解决意见挖掘或情感分析的领域。

[wordnet](https://blog.csdn.net/King_John/article/details/80252594)

## 3. [fuzzywuzzy](https://www.cnblogs.com/laoduan/p/python1.html)

fuzzywyzzy 是python下一个模糊匹配的模块。首先要安装fuzzywuzzy

##4. [wordcloud](https://blog.csdn.net/u010309756/article/details/67637930)





## 2. 代码

try

try的工作原理是，当开始一个try语句后，python就在当前程序的上下文中作标记，这样当异常出现时就可以回到这里，try子句先执行，接下来会发生什么依赖于执行时是否出现异常。

如果当try后的语句执行时发生异常，python就跳回到try并执行第一个匹配该异常的except子句，异常处理完毕，控制流就通过整个try语句（除非在处理异常时又引发新的异常）。
如果在try后的语句里发生了异常，却没有匹配的except子句，异常将被递交到上层的try，或者到程序的最上层（这样将结束程序，并打印缺省的出错信息）。
如果在try子句执行时没有发生异常，python将执行else语句后的语句（如果有else的话），然后控制流通过整个try语句。

```python
try:
<语句>        #运行别的代码
except <名字>：
<语句>        #如果在try部份引发了'name'异常
except <名字>，<数据>:
<语句>        #如果引发了'name'异常，获得附加的数据
else:
<语句>        #如果没有异常发生
```

