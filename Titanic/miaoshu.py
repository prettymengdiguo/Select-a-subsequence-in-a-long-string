# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:38:25 2018

@author: zhaokaituo
"""
[python] view plain copy
<code class="language-python">import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np  
  
titanic=pd.read_csv('train.csv')  
#print(titanic.head())  
#设置某一列为索引  
#print(titanic.set_index('PassengerId').head())  
  
# =============================================================================  
# #绘制一个展示男女乘客比例的扇形图  
# #sum the instances of males and females  
# males=(titanic['Sex']=='male').sum()  
# females=(titanic['Sex']=='female').sum()  
# #put them into a list called proportions  
# proportions=[males,females]  
# #Create a pie chart  
# plt.pie(  
# #        using proportions  
#         proportions,  
# #        with the labels being officer names  
#         labels=['Males','Females'],  
# #        with no shadows  
#         shadow=False,  
# #        with colors  
#         colors=['blue','red'],  
#         explode=(0.15,0),  
#         startangle=90,  
#         autopct='%1.1f%%'  
#         )  
# plt.axis('equal')  
# plt.title("Sex Proportion")  
# plt.tight_layout()  
# plt.show()  
# =============================================================================  
  
  
# =============================================================================  
# #绘制一个展示船票Fare,与乘客年龄和性别的散点图  
# #creates the plot using  
# lm=sns.lmplot(x='Age',y='Fare',data=titanic,hue='Survived',fit_reg=False)  
# #set title  
# lm.set(title='Fare x Age')  
# #get the axes object and tweak it  
# axes=lm.axes  
# axes[0,0].set_ylim(-5,)  
# axes[0,0].set_xlim(-5,85)  
# =============================================================================  
  
# =============================================================================  
# #绘制一个展示船票价格的直方图  
# #sort the values from the top to least value and slice the first 5 items  
# df=titanic.Fare.sort_values(ascending=False)  
# #create bins interval using numpy  
# binsVal=np.arange(0,600,10)  
# #create the plot  
# plt.hist(df,bins=binsVal)  
# plt.xlabel('Fare')  
# plt.ylabel('Frequency')  
# plt.title('Fare Payed Histrogram')  
# plt.show()  
# =============================================================================  
  
#哪个性别的年龄的平均值更大  
#print(titanic.groupby('Sex').Age.mean())  
#打印出不同性别的年龄的描述性统计信息  
#print(titanic.groupby('Sex').Age.describe())  
#print(titanic.groupby(['Sex','Survived']).Fare.describe())  
#先对Survived再Fare进行排序  
#a=titanic.sort_values(['Survived','Fare'],ascending=False)  
#print(a)  
#选取名字以字母A开头的数据  
#b=titanic[titanic.Name.str.startswith('A')]  
#print(b)  
#找到其中三个人的存活情况  
#c=titanic.loc[titanic.Name.isin(['Youseff, Mr. Gerious','Saad, Mr. Amin','Yousif, Mr. Wazli'])\  
#              ,['Name','Survived']]  
#print(c)  
# =============================================================================  
# ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))  
# ts = ts.cumsum()  
# ts.plot()  
# plt.show()  
#   
# df = pd.DataFrame(np.random.randn(1000, 4),index=ts.index,columns=['A', 'B', 'C', 'D'])  
# df=df.cumsum()  
# plt.figure()  
# df.plot()  
# plt.legend(loc='best')  
# plt.show()  
# =============================================================================  
#对应每一个location，一共有多少数据值缺失  
#print(titanic.isnull().sum())  
#对应每一个location，一共有多少数据值完整  
#print(titanic.shape[0]-titanic.isnull().sum())  
#查看每个列的数据类型  
#print(titanic.info())  
#print(titanic.dtypes)</code>

