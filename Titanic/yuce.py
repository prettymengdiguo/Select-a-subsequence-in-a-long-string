# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:39:53 2018

@author: zhaokaituo
"""

import pandas as pd
titanic=pd.read_csv("train.csv")
#print(titanic.describe())


titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
#print(titanic.describe())

#print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1


#print(titanic["Embarked"].value_counts())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2
#线性回归
# =============================================================================
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import KFold
# predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
# alg=LinearRegression()
# kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
# predictions=[]
# for train,test in kf:
#     train_predictors=(titanic[predictors].iloc[train,:])
#     train_target=titanic["Survived"].iloc[train]
#     alg.fit(train_predictors,train_target)
#     test_predictions=alg.predict(titanic[predictors].iloc[test,:])
#     predictions.append(test_predictions)
# 
# 
# import numpy as np
# predictions=np.concatenate(predictions,axis=0)
# predictions[predictions>.5]=1
# predictions[predictions<=.5]=0
# accuracy=sum(predictions==titanic["Survived"])/len(predictions)
# print(accuracy)
# =============================================================================
#逻辑回归
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
# predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
# alg=LogisticRegression(random_state=1)
# scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
# print(scores.mean())
# =============================================================================
#随机森林
# =============================================================================
# from sklearn import cross_validation
# from sklearn.ensemble import RandomForestClassifier
# predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
# alg=RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=12,min_samples_leaf=1)
# kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
# scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
# print(scores.mean())
# =============================================================================
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))


#提取名字信息
import re
def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""


titles=titanic["Name"].apply(get_title)
#print(pd.value_counts(titles))


title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Mlle":7,"Major":8,"Col":9,"Ms":10,"Mme":11,"Lady":12,"Sir":13,"Capt":14,"Don":15,"Jonkheer":16,"Countess":17}
for k,v in title_mapping.items():
    titles[titles==k]=v
#print(pd.value_counts(titles))
titanic["Title"]=titles
#特征选择
# =============================================================================
# import numpy as np
# from sklearn.feature_selection import SelectKBest,f_classif
# import matplotlib.pyplot as plt
# predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]
# selector=SelectKBest(f_classif,k=5)
# selector.fit(titanic[predictors],titanic["Survived"])
# scores=-np.log10(selector.pvalues_)
# 
# plt.bar(range(len(predictors)),scores)
# plt.xticks(range(len(predictors)),predictors,rotation='vertical')
# plt.show()
# =============================================================================


# =============================================================================
# from sklearn import cross_validation
# from sklearn.ensemble import RandomForestClassifier
# predictors=["Pclass","Sex","Fare","Title","NameLength"]
# alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=12,min_samples_leaf=1)
# kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
# scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
# print(scores.mean())
# =============================================================================


#集成学习
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
algorithms=[
        [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),["Pclass","Sex","Fare","Title","NameLength"]],
        [LogisticRegression(random_state=1),["Pclass","Sex","Fare","Title","NameLength"]]]


kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions=[]
for train,test in kf:
    train_target=titanic["Survived"].iloc[train]
    full_test_predictions=[]
    for alg,predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:],train_target)
        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions=(full_test_predictions[0]+full_test_predictions[1])/2
    test_predictions[test_predictions<=.5]=0
    test_predictions[test_predictions>.5]=1
    predictions.append(test_predictions)

predictions=np.concatenate(predictions,axis=0)
accuracy=sum(predictions==titanic["Survived"])/len(predictions)
print(accuracy)
