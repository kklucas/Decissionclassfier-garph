
# coding: utf-8

# In[3]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures #创建交互特征 
#https://blog.csdn.net/LULEI1217/article/details/49582821
from sklearn.pipeline import make_pipeline

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x* np.sin(x)

# generate points used to plot
x_plot = np.linspace(0, 10, 100)#在０－１０之间均匀的取100个数
print ('x_plot',x_plot,x_plot.shape)
# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)#产生一个伪随机数
rng.shuffle(x)#现场修改序列，改变自身内容。（类似洗牌，打乱顺序）
x = np.sort(x[:20])#产生一个子集,只去前20个数.用于作为训练点
print ('x',x,type(x),x.shape)
y = f(x) #y=x * np.sin(x)
print ('y',y,type(y),y.shape)
# create matrix versions of these arrays
X = x[:, np.newaxis]
print ('X',X,type(X),X.shape)
X_plot = x_plot[:, np.newaxis]#将一维的数组转化为矩阵形式
print ('X_plot',X_plot,type(X_plot),X_plot.shape )


plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")#画出散点图


for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())#使用岭回归来进行多项式的特征输出
    model.fit(X, y)#训练模型
    y_plot = model.predict(X_plot)#预测标签值
    plt.plot(x_plot, y_plot, label="degree %d" % degree)


plt.legend(loc='lower left')#画出画线标签


plt.show()


# In[11]:

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(max_depth=4,min_samples_leaf=4)
clf = clf.fit(iris.data, iris.target)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) # doctest: +SKIP
graph = graphviz.Source(dot_data) # doctest: +SKIP
graph.render("iris") # doctest: +SKIP
dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                            feature_names=iris.feature_names,  # doctest: +SKIP
                            class_names=iris.target_names,  # doctest: +SKIP
                            filled=True, rounded=True,  # doctest: +SKIP
                            special_characters=True)  # doctest: +SKIP
graph = graphviz.Source(dot_data)  # doctest: +SKIP
graph # doctest: +SKIP


# In[ ]:



