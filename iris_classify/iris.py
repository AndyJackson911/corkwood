import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')

#print(os.getcwd())
iris=pd.read_csv('iris_classify/iris.csv')
# print(iris)
# print(type(iris))
# print(iris.shape)
# print(iris.columns)
# print(iris.describe())



n = len(iris[iris['Species']=='versicolor'])
print("No of Versicolor in Dataset: ",n)

n1 = len(iris[iris['Species']=='virginica'])
print("No of Versicolor in Dataset: ",n)

n2 = len(iris[iris['Species']=='setosa'])
print("No of Versicolor in Dataset: ",n2)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['versicolor','setosa','virginica']
s = [50,50,50]
ax.pie(s, labels= l, autopct='%1.2f%%')
plt.show()

plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()


