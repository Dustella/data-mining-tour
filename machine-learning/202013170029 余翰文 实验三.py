
import operator
from math import log


class DecisionTree:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        myTree = self.create_tree(dataset, labels)
        self.tree = myTree

    def get_tree(self):
        return self.tree

    def get_entropy(self, dataset):
        """
        计算信息熵
        :param dataset: 数据集
        :return: 信息熵
        """
        cnt = {}
        for feature in dataset:
            currentlabel = feature[-1]
            if currentlabel not in cnt.keys():
                cnt[currentlabel] = 0
            cnt[currentlabel] += 1
        entropy = 0
        for key in cnt:
            # 依照信息熵的公式求 Ent(D)
            tmp = float(cnt[key]) / len(dataset)
            entropy -= tmp * log(tmp, 2)
        return entropy

    # 依照最大信息增益的 feature，选择最优特征

    def generate_best_feature(self, dataset):
        """
        依照最大信息增益的 feature，选择最优特征
        :param dataset: 数据集
        :return: 最优特征
        """
        # 特征数量
        base_entropy = self.get_entropy(dataset)
        best_infogain = 0
        best_feature = -1
        for i in range(len(dataset[0]) - 1):
            # 获取dataset的第i列所有特征
            features = [example[i] for example in dataset]
            unique = set(features)
            new_entropy = 0
            for val in unique:
                #  依据特征划分数据集，根据公式计算对应特征的信息增益
                subDataSet = self.split(dataset, i, val)
                prob = len(subDataSet) / float(len(dataset))
                new_entropy += prob * self.get_entropy(subDataSet)
            infogain = base_entropy - new_entropy
            if infogain > best_infogain:
                #  选择最大的信息增益
                best_infogain = infogain
                best_feature = i
        return best_feature

    # 依照特征划分数据集

    def split(self, dataset, axis, value):
        """
        :param dataset: 待划分的数据集
        :param axis: 划分数据集的特征
        :param value: 需要返回的特征的值
        :return: 划分后的数据集
        """
        res = []
        for feature in dataset:
            if feature[axis] == value:
                # 将符合特征的数据抽取出来
                se_feature = feature[:axis]
                se_feature.extend(feature[axis + 1:])
                res.append(se_feature)
        return res

    # 依照出现次数最多的类别作为叶子节点

    def get_majority(classList):
        """
        :param classList: 类别列表
        :return: 出现次数最多的类别
        """
        cnt = {}
        for vote in classList:
            if vote not in cnt.keys():
                cnt[vote] = 0
            cnt[vote] += 1
        sorted_cnt = sorted(
            cnt.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_cnt[0][0]

    def create_tree(self, dataset, labels):
        """
        :param dataset: 数据集
        :param labels: 标签集
        :return: 决策树
        """
        ls = [example[-1] for example in dataset]
        if ls.count(ls[0]) == len(ls):
            return ls[0]
        if len(dataset[0]) == 1:
            return self.get_majority(ls)
        # 选择最优特征
        best_feature = self.generate_best_feature(dataset)
        best_feature_label = labels[best_feature]
        # 根据最优特征的标签生成树
        myTree = {best_feature_label: {}}
        del(labels[best_feature])
        featValues = [example[best_feature] for example in dataset]
        # 获取列表中所有的属性值
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # 依照最优特征的不同取值，划分数据集
            child_ele = labels[:]
            myTree[best_feature_label][value] = self.create_tree(
                self.split(dataset, best_feature, value), child_ele)
        return myTree


if __name__ == '__main__':
    import json
    # 1. 随机生成数据集
    dataset = [['Sunny', 'Hot', 'High', 'Weak', 'no'],
               ['Sunny', 'Hot', 'High', 'Strong', 'no'],
               ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
               ['Rain', 'Mild', 'High', 'Weak', 'yes'],
               ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
               ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
               ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
               ['Sunny', 'Mild', 'High', 'Weak', 'no'],
               ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
               ['Rain', 'Mild', 'Normal', 'Weak', 'yes'],
               ['Sunny', 'Mild', 'Normal', 'Strong', 'yes'],
               ['Overcast', 'Mild', 'High', 'Strong', 'yes'],
               ['Overcast', 'Hot', 'Normal', 'Weak', 'yes'],
               ['Rain', 'Mild', 'High', 'Strong', 'no']]
    labels = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
    # 2. 创建决策树
    dt = DecisionTree(dataset, labels)
    # 3. 获取决策树
    tree = dt.get_tree()
    print(json.dumps(tree, indent=2, ensure_ascii=False))
