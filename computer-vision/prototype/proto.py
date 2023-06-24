import math
import cv2
import numpy as np

image = cv2.imread('image.jpg')


class Rect:
    def __init__(self, x, y, width, height) -> None:
        self.x, self.y, self.width, self.height = x, y, width, height

    def to_tuple(self):
        return (self.x, self.y, self.width, self.height)


# utils: generate rng


def rng(upper, lower=0):
    return math.floor(np.random.rand() * (upper - lower + 1) + lower)


class CompressiveTracker:
    feature_min_num_rect = 2
    feature_max_num_rect = 4  # number of rectangle from 2 to 4
    featureNum = 50  # number of all weaker classifiers, i.e,feature pool
    rOuterPositive = 4  # radical scope of positive samples
    rSearchWindow = 25  # size of search window
    muPositive = []
    muNegative = []
    sigmaPositive = []
    sigmaNegative = []

    def __init__(self):
        self.learnRate = 0.85  # Learning rate parameter

    def init(self, frame, ROI):
        self.ROI = ROI
        self.frame = frame

    def haar_feature(self, object_box: Rect, num_feature: int):

        features = np.array([])
        features_weight = np.array([])

        for i in range(num_feature):
            num_rect = rng(self.feature_max_num_rect,
                           self.feature_min_num_rect)
            for j in range(num_rect):
                x, y = rng(object_box.width - 3), rng(object_box.height - 3)
                w, h = rng(object_box.width - x -
                           1), rng(object_box.height - y - 1)
                features = np.append(features, Rect(x, y, w, h))

                features_weight = np.append(
                    features_weight, math.pow(-1, rng(2)) / math.sqrt(num_rect))

    def sample_rect(self, image: np.ndarray, rect: Rect):
        x, y, w, h = rect.to_tuple()
        return image[y:y+h, x:x+w]


if __name__ == '__main__':
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
