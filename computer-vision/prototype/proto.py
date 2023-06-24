import math
from typing import List
import cv2
import numpy as np



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
    features =[]
    features_weight = []
    sample_feature_value = np.array([])
    sample_positive_box = []
    sample_negative_box = []
    detect_box = []
    detect_feature_value = np.array([])
    ROI = Rect(0, 0, 0, 0)
    frame = np.array([])
    image_integral = np.array([])
    sample_positive_feature_value = np.array([])
    sample_negative_feature_value = np.array([])
    learnRate = 0.85


    def __init__(self):
        self.learnRate = 0.85  # Learning rate parameter
        

    def init(self, frame, ROI):
        self.ROI = ROI
        self.frame = frame

    def haar_feature(self, object_box: Rect, num_feature: int):
        temp_features = []
        temp_features_weight = []
        for i in range(num_feature):
            num_rect = rng(self.feature_max_num_rect,
                           self.feature_min_num_rect)
            for j in range(num_rect):
                x, y = rng(object_box.width - 3), rng(object_box.height - 3)
                w, h = rng(object_box.width - x -
                           1), rng(object_box.height - y - 1)
                if i >= len(temp_features):
                    temp_features.append([])
                if i >= len(temp_features_weight):
                    temp_features_weight.append([])
                temp_features[i].append(Rect(x, y, w, h))
                temp_features_weight[i].append( rng(2) * 2 - 1)
                
        self.features = temp_features
        self.features_weight =  temp_features_weight

    def sample_rect(self, image: np.ndarray, object_box: Rect, r_inner:float,r_outer:float,max_sample_num:int,sample_box:List[Rect]):
        row_sz, col_sz = image.shape[0] - object_box.height -1, image.shape[1] - object_box.width - 1

        in_radsq , out_radsq = r_inner * r_inner, r_outer*r_outer

        min_row = max(0, int(object_box.y-r_inner))
        max_row = min(row_sz -1 ,int(object_box.y+r_inner))
        min_col = max(0,int(object_box.x-r_inner))
        max_col = min(col_sz -1 ,int(object_box.x+r_inner))

        prob = max_sample_num / (max_row - min_row + 1) / (max_col - min_col + 1)
        i = 0
        sample_box.clear()
        for r in range(min_row,max_row):
            for c in range(min_col,max_col):
                dist = (r-object_box.y)*(r-object_box.y) + (c-object_box.x)*(c-object_box.x)

                if rng(1) < prob and dist < in_radsq and dist > out_radsq:
                    to_append = Rect(c,r,object_box.width,object_box.height)
                    sample_box.append(to_append)
                    i += 1
            
            np.resize(sample_box,i)
        print("Done for sample_rect")
        print(sample_box[:5])

    def get_feature_value(self, image_integral, sample_box: List[Rect]):
        sample_box_size = len(sample_box)
        sample_feature_value = np.zeros((self.featureNum, sample_box_size), dtype=float)

        for i in range(self.featureNum):
            for j in range(sample_box_size):
                temp_value = 0
                for k in range(len(self.features[i])):
                    x_min = sample_box[j].x + self.features[i][k].x
                    x_max = sample_box[j].x + self.features[i][k].x + self.features[i][k].width
                    y_min = sample_box[j].y + self.features[i][k].y
                    y_max = sample_box[j].y + self.features[i][k].y + self.features[i][k].height
                    temp_value += self.features_weight[i][k] * \
                    (image_integral[y_min][x_min] + \
                        image_integral[y_max][x_max] - \
                        image_integral[y_min][x_max] -\
                                image_integral[y_max][x_min])
                sample_feature_value[i][j] = temp_value.sum()
        print("Done for get_feature_value")
        return sample_feature_value

    def classifier_update(self, sample_feature_value: np.ndarray):
        mu = np.zeros((self.featureNum, 1), dtype=float)
        sigma = np.zeros((self.featureNum, 1), dtype=float)
        learn_rate = self.learnRate
        print("sample_feature_value",sample_feature_value)
        for i in range(self.featureNum):
            muTemp, sigmaTemp =  cv2.meanStdDev(sample_feature_value[i])
            sigma[i] = math.sqrt(learn_rate*sigma[i]*sigma[i] + (1-learn_rate)*sigmaTemp*sigmaTemp)*mu[i] \
                + learn_rate*(1-learn_rate)*(mu[i]-muTemp[0])*(mu[i]-muTemp[0])
            mu[i] = learn_rate*mu[i] + (1-learn_rate)*muTemp[0]
        
        return mu, sigma

    def radio_classifier(self, mu_pos:List[float],sigma_pos:List[float],
                         mu_neg:List[float],sigma_neg:List[float],
                         sample_feature_value:np.ndarray):
        sample_box_num = sample_feature_value.shape[1]
        # radio = np.zeros((self.featureNum, sample_box_num), dtype=float)
        radio_max_index = 0
        radio_max = 0
        for j in range(sample_box_num):
            sum_radio = 0
            for i in range(self.featureNum):
            #     pPos = exp( (_sampleFeatureValue.at<float>(i,j)-_muPos[i])*(_sampleFeatureValue.at<float>(i,j)-_muPos[i]) / -(2.0f*_sigmaPos[i]*_sigmaPos[i]+1e-30) ) / (_sigmaPos[i]+1e-30);
			# pNeg = exp( (_sampleFeatureValue.at<float>(i,j)-_muNeg[i])*(_sampleFeatureValue.at<float>(i,j)-_muNeg[i]) / -(2.0f*_sigmaNeg[i]*_sigmaNeg[i]+1e-30) ) / (_sigmaNeg[i]+1e-30);
			# sumRadio += log(pPos+1e-30) - log(pNeg+1e-30);	// equation 4
                p_pos = math.exp(
                    (sample_feature_value[i][j] - mu_pos[i])*(sample_feature_value[i][j] - mu_pos[i]) / -(2.0*sigma_pos[i]*sigma_pos[i]+1e-30)) / (sigma_pos[i]+1e-30)
                p_neg = math.exp(
                    (sample_feature_value[i][j] - mu_neg[i])*(sample_feature_value[i][j] - mu_neg[i]) / -(2.0*sigma_neg[i]*sigma_neg[i]+1e-30)) / (sigma_neg[i]+1e-30)
                sum_radio += math.log(p_pos+1e-30) - math.log(p_neg+1e-30)
            if sum_radio > radio_max:
                radio_max_index = j
                radio_max = sum_radio
        return radio_max_index, radio_max
    
    def init(self, frame, object_box: Rect):
        self.haar_feature(object_box, self.featureNum)
        self.sample_rect(frame, object_box, self.rOuterPositive,
                            0,1000000, self.sample_positive_box)
        self.sample_rect(frame, object_box, self.rSearchWindow *1.5,
                            self.rOuterPositive + 4, 100, self.sample_negative_box)
        
        self.image_integral = cv2.integral(frame)

        self.sample_positive_feature_value= \
            self.get_feature_value(self.image_integral, self.sample_positive_box)
        self.sample_negative_feature_value =\
        self.get_feature_value(self.image_integral, self.sample_negative_box)


        self.muPositive, self.sigmaPositive = \
            self.classifier_update(self.sample_positive_feature_value)
        self.muNegative, self.sigmaNegative = \
            self.classifier_update(self.sample_negative_feature_value)
    
    def process_frame(self, frame,object_box:Rect):
        self.sample_rect(frame, object_box, self.rSearchWindow,
                            self.rOuterPositive, 1000000, self.detect_box)
        
        self.image_integral = cv2.integral(frame)

        self.detect_feature_value = \
            self.get_feature_value(self.image_integral, self.detect_box)


        radio_max_index,_ = \
            self.radio_classifier(self.muPositive, self.sigmaPositive,
                                self.muNegative, self.sigmaNegative,
                                self.detect_feature_value )
        
        self.ROI = self.detect_box[radio_max_index]

        _ = self.sample_rect(frame, self.ROI, self.rOuterPositive,
                            0, 1000000, self.sample_positive_box)
        _ = self.sample_rect(frame, self.ROI, self.rSearchWindow * 1.5,
                            self.rOuterPositive + 4, 100, self.sample_negative_box)
        
        self.sample_positive_feature_value = \
            self.get_feature_value(self.image_integral, self.sample_positive_box, )
        self.sample_negative_feature_value = \
            self.get_feature_value(self.image_integral, self.sample_negative_box, )

        self.muPositive, self.sigmaPositive = \
            self.classifier_update(self.sample_positive_feature_value, )
        self.muNegative, self.sigmaNegative = \
            self.classifier_update(self.sample_negative_feature_value, )

if __name__ == '__main__':
    image = cv2.imread('./computer-vision/prototype/Lenna.jpg')

    tracker = CompressiveTracker()

    tracker.init(image, Rect(100, 100, 50, 50))
    tracker.process_frame(image, Rect(100, 100, 50, 50))
    print(tracker.ROI)
