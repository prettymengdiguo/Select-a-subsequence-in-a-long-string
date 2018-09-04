# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:25:11 2018

@author: zhaokaituo
"""

import unicodecsv
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('train.csv') #导入数据
#查看数据集的基本信息
titanic_df.info()
#该数据集有12个字段，PassengerId：乘客编号，Survived：乘客是否存活，
#Pclass：乘客所在的船舱等级；Name：乘客姓名，Sex：乘客性别，Age：乘客年龄，
#SibSp：乘客的兄弟姐妹和配偶数量，Parch：乘客的父母与子女数量，
#Ticket：票的编号，Fare：票价，Cabin：座位号，Embarked：乘客登船码头。

#查看数据缺失值
titanic_df.isnull().sum()
#查看数据集的描述统计情况
titanic_df[['Survived','Age','SibSp','Parch','Fare']].describe()

#分类变量，计数，唯一值，众数，频次
titanic_df.describe(include=[np.object])#利用include=[np.object]查看分类型数据的描述性统计
#查看前5行数据
titanic_df.head(5)
#处理缺失值-用最接近的数据替换，数值型数据用该列的数据的均值或者中位数替换，分类型数据用该列出现频数最大的数（众数
#或者后续操作跳过空值
#用中位数替换age缺失值
age_median=titanic_df.Age.median()
print(age_median)
titanic_df.Age.fillna(age_median,inplace=True)#使用fillna填充缺失值，inplace=True表示在原数据Titanic直接修改
titanic_df.Age.describe()
#用众数填充Embarked
titanic_df.Embarked.describe(include=[np.object])
#另一种方式，计数统计
titanic_df.Embarked.value_counts()
titanic_df.Embarked.fillna("S",inplace=True)
#titanic_df.fillna({"Embarked":"S"},inplace=Ture)
titanic_df.Embarked.isnull().sum()#查看缺失值填充效果
#生存率基本情况
total_survived=titanic_df['Survived'].sum()
total_no_survived=891-total_survived
total_survived_rate=total_survived/891
#coding=utf-8
print("总生存人数")
print(total_survived)
print("总生存率")
print(total_survived_rate)
plt.figure(figsize=(12,5))#创建画布
plt.subplot(121)#添加第一个图
sns.countplot(x='Survived',data=titanic_df)
plt.title('Survival count')
plt.subplot(122)
plt.pie([total_no_survived,total_survived],labels=['No Survived','Suevived'],autopct='%1.0f%%')
plt.title('Survival rate')
plt.show()

#单变量探索
#性别与生存率
#用groupby函数，各性别生存率=各性别生存人数/各性别总人数
sex_survived = (titanic_df.groupby(['Sex']).sum())['Survived']#求各性别生存人数
sex_total = (titanic_df.groupby(['Sex']).count())['Survived']#求各性别总人数
sex_survived_rate=sex_survived/sex_total
print("各性别总人数")
print(sex_total)
print("各性别生存率")
print(sex_survived_rate)

plt.figure(figsize=(12,4))#创建画布
plt.subplot(121)#添加第一个图
sex_total.plot(kind='pie',autopct='%.0f%%')
plt.title('sex_total')
plt.subplot(122)
sex_survived_rate.plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('sex_survived_rate')
plt.show()
#年龄与生存率
Age_survived_rate = ((titanic_df.groupby(['Age'])).sum()/titanic_df.groupby(['Age']).count())['Survived']
Age_survived_rate.plot()
plt.title('Age_survived_rate')
plt.show()#得到的图比较乱。因此对年龄进行分段处理
#Age列的描述性统计值
titanic_df.Age.describe()
#用groupby函数
bins=np.arange(0,81,10)#设置分组的段：0-80，每10岁一段
age_cut=pd.cut(titanic_df['Age'],bins)#将数据Age进行分段
age_cut_grouped = titanic_df.groupby(age_cut)#将数据进行分组
age_cut_survived = age_cut_grouped.sum()['Survived']#计算分段分组后的年龄生存总数
age_cut_total = age_cut_grouped.count()['Survived']#计算分段分组后的年龄生存总人数
age_survived_rate=age_cut_survived /age_cut_total #计算各个年龄段的幸存率
#coding=utf-8
print('各年龄段总人数')
print(age_cut_total)
print('各年龄段生存率')
print(age_survived_rate)
plt.figure(figsize=(12,5))#创建画布
plt.subplot(121)#添加第一个图
age_cut_total.plot(kind='pie',autopct='%.0f%%')
plt.title('age_cut_total')
plt.subplot(122)
age_survived_rate.plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('age_survived_rate')
plt.show()

#船舱等级与生存率
#用groupby函数，船舱等级生存率=船舱等级生存人数/船舱等级总人数
pclass_survived = (titanic_df.groupby(['Pclass']).sum())['Survived']
pclass_total = (titanic_df.groupby(['Pclass']).count())['Survived']
pclass_survived_rate = pclass_survived / pclass_total
#coding=utf-8
print('各船舱等级人数')
print(pclass_total)
print('各船舱等级生存率')
print(pclass_survived_rate)
plt.figure(figsize=(12,5))#创建画布
plt.subplot(121)#添加第一个图
pclass_total.plot(kind='pie',autopct='%.0f%%')
plt.title('pclass_total')
plt.subplot(122)
pclass_survived_rate.plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('pclass_survived_rate')
plt.show()

#性别、船舱等级与生存率的关系
pclass_sex_survived_rate = (titanic_df.groupby(['Pclass','Sex']).sum()/titanic_df.groupby(['Pclass','Sex']).count())['Survived']
print('船舱等级和性别与生存率')
print(pclass_sex_survived_rate)
pclass_sex_survived_rate.unstack().plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('pclass_sex_survived_rate')
plt.ylabel('survived_rate')
plt.show()
#年龄、性别与生存率
bins = np.arange(0,81,10)
age_cut = pd.cut(titanic_df['Age'],bins)
age_sex_survived_rate = (titanic_df.groupby([age_cut,'Sex']).sum()/titanic_df.groupby([age_cut,'Sex']).count())['Survived']
#coding=utf-8
print('年龄、性别与生存率')
print(age_sex_survived_rate)
age_sex_survived_rate.unstack().plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('age_sex_survived_rate')
plt.ylabel('survived_rate')
plt.show()

#年龄、舱位等级与生存率
bins = np.arange(0,81,10)
age_cut = pd.cut(titanic_df['Age'],bins)
age_pclass_survived_rate = (titanic_df.groupby([age_cut,'Pclass']).sum()/titanic_df.groupby([age_cut,'Pclass']).count())['Survived']
#coding=utf-8
print('年龄、船舱等级与生存率')
print(age_pclass_survived_rate)
age_pclass_survived_rate.unstack().plot(kind='bar')#bar代表柱状图，pie代表饼状图
plt.title('age_pclass_survived_rate')
plt.ylabel('survived_rate')
plt.show()

print('船舱等级、年龄和生存率')
titanic_df.groupby([age_cut,'Pclass'])['Survived'].count()
