import numpy as np


class Node:

    """
    决策树节点的类定义。
    """

    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):

        self.feature_idx = feature_idx  # 特征编号
        self.threshold = threshold  # 特征阈值
        self.value = value  # 叶节点的类别
        self.left = left  # 左子树节点
        self.right = right  # 右子树节点


class DecisionTree:

    """
    决策树的类定义。
    """

    def __init__(self, max_depth=5):

        self.max_depth = max_depth  # 决策树的最大深度
        self.root = None  # 根节点

    def fit(self, X, y):
        """
        使用训练数据拟合决策树。
        """
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """
        对新数据进行预测。
        """
        y_pred = []
        for x in X:
            node = self.root
        while node.left:
            if x[node.feature_idx] < node.threshold:
                node = node.left
        else:
            node = node.right
            y_pred.append(node.value)
        return y_pred

    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树。
        """
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            # 达到最大深度或叶节点只有一种类别，返回叶节点
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value)

    # 寻找最佳分割点
        best_feature_idx, best_threshold = self._find_best_split(X, y)

        # 递归构建左右子树
        left_idxs = X[:, best_feature_idx] < best_threshold
        right_idxs = X[:, best_feature_idx] >= best_threshold
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)

        # 返回当前节点
        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        """
        寻找最佳分割点。
        """
        best_gain = 0
        split_feature_idx, split_threshold = None, None
        n_samples, n_features = X.shape

        # 遍历所有特征和阈值
        for feature_idx in range(n_features):
            for threshold in np.unique(X[:, feature_idx]):
                left_idxs = X[:, feature_idx] < threshold
                right_idxs = X[:, feature_idx] >= threshold

                # 计算信息增益
                gain = self._information_gain(y, y[left_idxs], y[right_idxs])
                if gain > best_gain:
                    best_gain = gain
                    split_feature_idx = feature_idx
                    split_threshold = threshold

        return split_feature_idx, split_threshold

    def _information_gain(self, parent, left_child, right_child):
        """
        计算信息增益。
        """
        # 计算父节点的熵
        parent_entropy = self._entropy(parent)

        # 计算左右子节点的权重和熵
        n_left, n_right = len(left_child), len(right_child)
        left_entropy = self._entropy(left_child)
        right_entropy = self._entropy(right_child)

        # 计算信息增益
        info_gain = parent_entropy - (n_left / (n_left + n_right)) * \
            left_entropy - (n_right / (n_left + n_right)) * right_entropy

        return info_gain

    def _entropy(self, y):
        """
        计算熵。
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _most_common_class(self, y):
        """
        计算样本数最多的类别。
        """
        _, counts = np.unique(y, return_counts=True)
        return _[np.argmax(counts)]
