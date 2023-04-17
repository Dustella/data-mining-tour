# define dataset
import random


# implement k-means to cluster


def distance(sample, centroid):
    return sum([(sample[i] - centroid[i]) ** 2 for i in range(len(sample))]) ** 0.5


def average(cluster):
    return [sum(dimension) / len(dimension) for dimension in zip(*cluster)]


def k_means(dataset, k):
    centroids = random.sample(dataset, k)
    last_centroids, clusters = None, None
    # 当质心不再变化时，停止迭代
    while last_centroids != centroids:
        # create clusters
        clusters = [[] for i in range(k)]
        # record the last centroids
        last_centroids = centroids
        # assign each sample to the nearest centroid
        for sample in dataset:
            # r记录每个样本到每个质心的距离
            distances = [distance(sample, centroid) for centroid in centroids]
            # 找到最近的质心
            nearest = distances.index(min(distances))
            # 将样本添加到最近的质心所在的簇中
            clusters[nearest].append(sample)
        # 更新质心
        centroids = [average(cluster) for cluster in clusters]
    return centroids, clusters


# using 'dataset' and 'k' to test the k-means algorithm
if __name__ == '__main__':
    dataset = [
        [1, 1],
        [1, 2],
        [2, 1],
        [100, 100],
        [100, 101],
        [101, 100],
        [66, 22],
        [66, 23],
        [67, 22],
    ]

    # 定义 k
    k = 3
    centroids, clusters = k_means(dataset, k)
    print(centroids)
    print(clusters)
