import itertools


def loadDataSet():
    return [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]


class Apriori:
    def __init__(self, dataset, support) -> None:
        self.dataset = dataset
        self.support = support
        len = 1
        collections = self.getCollections(dataset, len)
        support_map = self.getSupport(dataset, collections)
        len += 1
        while support_map.values().__len__() > 0:
            collections = self.getCollections(dataset, len)
            support_map = self.getSupport(dataset, collections)
            len += 1
            if support_map != {}:
                self.support_map = support_map

    def get_result(self):
        return self.support_map

    def getCollections(self, dataSet: list, target_len: int):
        collections = set()
        for data in dataSet:
            for item in data:
                collections.add(item)
        # 生成 collections 所有长度为 target_len 的子集
        return [set(i) for i in itertools.combinations(collections, target_len)]

    def getSupport(self, dataSet: list, collections: list):
        support_map = {}
        for collection in collections:
            count = 0
            for data in dataSet:
                if collection.issubset(set(data)):
                    count += 1
            if count >= self.support:
                support_map[tuple(collection)] = count
        return support_map


if __name__ == "__main__":
    dataset = loadDataSet()
    apriori = Apriori(dataset, 2)
    print(apriori.get_result())
