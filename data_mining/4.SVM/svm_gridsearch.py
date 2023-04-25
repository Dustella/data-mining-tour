
# system lib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC, LinearSVC, NuSVC
# from sklearn import cross_validation

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn import tree
import pandas as pd
# 用于参数搜索
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc  # 绘制ROC曲线
import pylab as pl

from time import time
import datetime
import numpy as np
# python 2.7
from sklearn.preprocessing import MinMaxScaler
# import cPickle as pickle
# python 3.x
import pickle
from sklearn.model_selection import cross_validate
# local lib

# pickle持久化保存模型的方法示例，需要先调用open方法创建文件对象
# pickle.dump(vectors, output, 1)
# pickle.load(filename)


def load_data():
    """根据数据格式，读取数据中的X和分类标签y
    """
    # 读取数据，NaN用众数填充
    data = pd.read_csv(
        "./data_mining/4.SVM/preprocess_train.csv", index_col='sample_id', sep=',', header=0, encoding='utf-8', na_values='NaN')
    # 使用众数填充缺失值
    data = data.fillna(data.mode().iloc[0])
    # 对数据预处理
    # 找出并且打印所有方差为0的列
    zero_variance_columns = data.columns[data.var() == 0]
    print(zero_variance_columns)
    # 进行 max min 归一化, 使用minmaxscaler
    X = data.drop(['label'], axis=1)
    Y = data['label']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # 将数据转换为DataFrame
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    # 将数据分为训练集和测试集
    # 从sklearn.model_selection中导入train_test_split用于数据分割。
    from sklearn.model_selection import train_test_split
    # 随机采样25%的数据构建测试样本，其余作为训练样本。
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=33)

    return x_train, y_train, x_test, y_test


def evaluate_classifier(real_label_list, predict_label_list):
    """
       return Precision, Recall and ConfusionMatrix
       Input : predict_label_list,real_label_list
    """
    msg = ''
    Confusion_matrix = confusion_matrix(real_label_list, predict_label_list)
    msg += '\n Confusion Matrix\n ' + str(Confusion_matrix)
    precision = precision_score(
        real_label_list, predict_label_list, average=None)
    recall = recall_score(real_label_list, predict_label_list, average=None)
    msg += '\n Precision of tag 0 and 1 =%s' % str(precision)
    msg += '\n Recall of tag 0 and 1 =%s' % str(recall)

    return msg


def test_svm():
    """用SVM分类 """
    # use SVM directly

    train_xdata,  train_ylabel, test_xdata, test_ylabel = load_data()

    print('\nuse SVM directly')

    # classifier1 = SVC(kernel='linear')
    # classifier1 = SVC(kernel='linear',probability=True, C=200, cache_size=500)
    classifier1 = SVC(kernel='rbf', probability=True,
                      C=1000, cache_size=500, gamma=2, class_weight='balanced')

    classifier1.fit(train_xdata, train_ylabel)

    predict_labels = classifier1.predict(test_xdata)
    accuracy = accuracy_score(test_ylabel, predict_labels)
    print("\n The Classifier's Accuracy is : %f" % accuracy)
    #
    eval_msg = evaluate_classifier(test_ylabel, predict_labels)
    print(eval_msg)
    #
    # GridSearchCV搜索最优参数示例
    print("GridSearchCV搜索最优参数......")
    t0 = time()
    param_grid = {
        "C": [1, 10, 1e2, 1e3],
        "gamma": [0.1, 0.2, 0.5, 0.8],
    }
    classifier1 = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True), param_grid)
    classifier1 = classifier1.fit(train_xdata, train_ylabel)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(classifier1.best_estimator_)

    # 对于SVM来说，概率是通过交叉验证得到的，与其预测的结果未必一致，对小数据集来说，此概率没什么意义
    probas_ = classifier1.predict_proba(test_xdata)

    # 对于二分类问题，可为分类器绘制ROC曲线，计算AUC
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_ylabel, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('SVM ROC')
    pl.legend(loc="lower right")
    pl.show()


def main():

    test_svm()
    # load_data()


if __name__ == '__main__':
    main()
