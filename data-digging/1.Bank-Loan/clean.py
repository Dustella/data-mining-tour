
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('./data_mining/1.Bank-Loan/bankloan.csv', header=0)
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
        # 最大最小归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        data_minmax = min_max_scaler.fit_transform(self.data)
        print(data_minmax)


if __name__ == "__main__":
    cleaner = Cleaner(data)
    cleaner.process()
    cleaner.train()
