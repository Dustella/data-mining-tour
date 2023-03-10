from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import re
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('./1/bankloan.csv', header=0)
# 用决策树方法分类


class Cleaner:

    def __init__(self, data) -> None:
        self.data = data
        pass

    def process(self):
        # 找出每一列的众数，并且用众数补齐缺失的值
        for i in self.data.columns:
            self.data[i].fillna(self.data[i].mode()[0], inplace=True)
        # 对于数据中的每一列，如果它是object类型的，就用数字代替
        for i in self.data.columns:
            if self.data[i].dtype == 'object':
                self.data[i] = pd.Categorical(self.data[i]).codes

    def train(self):
        # 使用决策树分类
        # predictors是A1-A15
        predictors = [f"A{i}" for i in range(1, 16)]
        clf_tree = tree.DecisionTreeClassifier()
        score = cross_val_score(
            clf_tree, self.data[predictors], data["A16"], cv=3)
        print(score.mean())


if __name__ == "__main__":
    cleaner = Cleaner(data)
    cleaner.process()
    cleaner.train()
