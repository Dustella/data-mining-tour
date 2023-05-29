# coding=utf8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2
import os
import sys
import scipy.ndimage
import time
import scipy
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key


###################################################### 1. 定义SIFT类 ####################################################
class CSift:
    def __init__(self, num_octave, num_scale, sigma):
        self.sigma = sigma  # 初始尺度因子
        self.num_scale = num_scale  # 层数
        self.num_octave = 3  # 组数，后续重新计算
        self.contrast_t = 0.04  # 弱响应阈值
        self.eigenvalue_r = 10  # hessian矩阵特征值的比值阈值
        self.scale_factor = 1.5  # 求取方位信息时的尺度系数
        self.radius_factor = 3  # 3被采样率
        self.num_bins = 36  # 计算极值点方向时的方位个数
        self.peak_ratio = 0.8  # 求取方位信息时，辅方向的幅度系数

###################################################### 2. 构建尺度空间 ####################################################


def pre_treat_img(img_src, sigma, sigma_camera=0.5):
    # 因为接下来会将图像尺寸放大2倍处理，所以sigma_camera值翻倍
    sigma_mid = np.sqrt(sigma**2 - (2*sigma_camera)**2)
    img = img_src.copy()
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2),
                     interpolation=cv2.INTER_LINEAR)  # 注意dstSize的格式，行、列对应高、宽
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_mid, sigmaY=sigma_mid)
    return img


def get_numOfOctave(img):
    num = round(np.log(min(img.shape[0], img.shape[1]))/np.log(2))-1
    return num


def construct_gaussian_pyramid(img_src, sift: CSift):
    pyr = []
    img_base = img_src.copy()
    for i in range(sift.num_octave):  # 共计构建octave组
        octave = construct_octave(img_base, sift.num_scale,
                                  sift.sigma)  # 构建每一个octave组
        pyr.append(octave)
        img_base = octave[-3]  # 倒数第三层的尺度与下一组的初始尺度相同，对该层进行降采样，作为下一组的图像输入
        img_base = cv2.resize(img_base, (int(
            img_base.shape[1]/2), int(img_base.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
    return pyr


def construct_octave(img_src, s, sigma):
    octave = []
    octave.append(img_src)  # 输入的图像已经进行过GaussianBlur了
    k = 2**(1/s)
    for i in range(1, s+3):  # 为得到S层个极值结果，需要构建S+3个高斯层
        img = octave[-1].copy()
        cur_sigma = k**i*sigma
        pre_sigma = k**(i-1)*sigma
        mid_sigma = np.sqrt(cur_sigma**2 - pre_sigma**2)
        cur_img = cv2.GaussianBlur(
            img, (0, 0), sigmaX=mid_sigma, sigmaY=mid_sigma)
        octave.append(cur_img)
    return octave
###################################################### 3. 寻找初始极值点 ##################################################


def construct_DOG(pyr):
    dog_pyr = []
    for i in range(len(pyr)):  # 对于每一组高斯层
        octave = pyr[i]  # 获取当前组
        dog = []
        for j in range(len(octave)-1):  # 对于当前层
            diff = octave[j+1]-octave[j]
            dog.append(diff)
        dog_pyr.append(dog)
    return dog_pyr


def get_keypoints(gau_pyr, dog_pyr, sift: CSift):
    key_points = []
    threshold = np.floor(0.5 * sift.contrast_t /
                         sift.num_scale * 255)  # 原始图像灰度范围[0,255]
    for octave_index in range(len(dog_pyr)):  # 遍历每一个DoG组
        octave = dog_pyr[octave_index]  # 获取当前组下高斯差分层list
        for s in range(1, len(octave)-1):  # 遍历每一层（第1层到倒数第2层）
            bot_img, mid_img, top_img = octave[s -
                                               1], octave[s], octave[s+1]  # 获取3层图像数据
            board_width = 5
            x_st, y_st = board_width, board_width
            x_ed, y_ed = bot_img.shape[0] - \
                board_width, bot_img.shape[1]-board_width
            for i in range(x_st, x_ed):  # 遍历中间层图像的所有x
                for j in range(y_st, y_ed):  # 遍历中间层图像的所有y
                    flag = is_extreme(bot_img[i-1:i+2, j-1:j+2], mid_img[i-1:i+2,
                                      j-1:j+2], top_img[i-1:i+2, j-1:j+2], threshold)  # 初始判断是否为极值
                    if flag:  # 若初始判断为极值，则尝试拟合获取精确极值位置
                        reu = try_fit_extreme(
                            octave, s, i, j, board_width, octave_index, sift)
                        if reu is not None:  # 若插值成功，则求取方向信息，
                            kp, stemp = reu
                            kp_orientation = compute_orientation(
                                kp, octave_index, gau_pyr[octave_index][stemp], sift)
                            for k in kp_orientation:  # 将带方向信息的关键点保存
                                key_points.append(k)
    return key_points


def is_extreme(bot, mid, top, thr):
    c = mid[1][1]
    temp = np.concatenate([bot, mid, top], axis=0)
    if c > thr:
        index1 = temp > c
        flag1 = len(np.where(index1 == True)[0]) > 0
        return not flag1
    elif c < -thr:
        index2 = temp < c
        flag2 = len(np.where(index2 == True)[0]) > 0
        return not flag2
    return False


def try_fit_extreme(octave, s, i, j, board_width, octave_index, sift: CSift):
    flag = False
    # 1. 尝试拟合极值点位置
    for n in range(5):  # 共计尝试5次
        bot_img, mid_img, top_img = octave[s - 1], octave[s], octave[s + 1]
        g, h, offset = fit_extreme(bot_img[i - 1:i + 2, j - 1:j + 2],
                                   mid_img[i - 1:i + 2, j - 1:j + 2], top_img[i - 1:i + 2, j - 1:j + 2])
        if(np.max(abs(offset)) < 0.5):  # 若offset的3个维度均小于0.5，则成功跳出
            flag = True
            break
        s, i, j = round(s+offset[2]), round(i+offset[1]
                                            ), round(j+offset[0])  # 否则，更新3个维度的值，重新尝试拟合
        # 若超出边界，直接退出
        if i < board_width or i > bot_img.shape[0]-board_width or j < board_width or j > bot_img.shape[1]-board_width or s < 1 or s > len(octave)-2:
            break
    if not flag:
        return None
    # 2. 拟合成功，计算极值
    ex_value = mid_img[i, j]/255+0.5*np.dot(g, offset)  # 求取经插值后的极值
    if np.abs(ex_value)*sift.num_scale < sift.contrast_t:  # 再次进行弱响应剔除
        return None
    # 3. 消除边缘响应
    hxy = h[0:2, 0:2]  # 获取关于x、y的hessian矩阵
    trace_h = np.trace(hxy)  # 求取矩阵的迹
    det_h = det(hxy)  # 求取矩阵的行列式
    # 若hessian矩阵的特征值满足条件（认为不是边缘）
    if det_h > 0 and (trace_h**2/det_h) < ((sift.eigenvalue_r+1)**2/sift.eigenvalue_r):
        kp = cv2.KeyPoint()
        kp.response = abs(ex_value)  # 保存响应值
        i, j = (i+offset[1]), (j+offset[0])  # 更新精确x、y位置
        # 这里保存坐标的百分比位置，免去后续在不同octave上的转换
        kp.pt = j/bot_img.shape[1], i/bot_img.shape[0]
        kp.size = sift.sigma * \
            (2**((s+offset[2])/sift.num_scale)) * \
            2**(octave_index)  # 保存sigma(o,s)
        # 低8位存放octave的index，中8位存放s整数部分，剩下的高位部分存放s的小数部分
        kp.octave = octave_index + s * \
            (2 ** 8) + int(round((offset[2] + 0.5) * 255)) * (2 ** 16)
        return kp, s
    return None


def fit_extreme(bot, mid, top):  # 插值求极值
    arr = np.array([bot, mid, top])/255
    g = get_gradient(arr)
    h = get_hessian(arr)
    rt = -lstsq(h, g, rcond=None)[0]  # 求解方程组
    return g, h, rt


def get_gradient(arr):  # 获取一阶梯度
    dx = (arr[1, 1, 2]-arr[1, 1, 0])/2
    dy = (arr[1, 2, 1] - arr[1, 0, 1])/2
    ds = (arr[2, 1, 1] - arr[0, 1, 1])/2
    return np.array([dx, dy, ds])


def get_hessian(arr):  # 获取三维hessian矩阵
    dxx = arr[1, 1, 2]-2*arr[1, 1, 1] + arr[1, 1, 0]
    dyy = arr[1, 2, 1]-2*arr[1, 1, 1] + arr[1, 0, 1]
    dss = arr[2, 1, 1]-2*arr[1, 1, 1] + arr[0, 1, 1]
    dxy = 0.25*(arr[1, 0, 0]+arr[1, 2, 2]-arr[1, 0, 2] - arr[1, 2, 0])
    dxs = 0.25*(arr[0, 1, 0]+arr[2, 1, 2] - arr[0, 1, 2] - arr[2, 1, 0])
    dys = 0.25*(arr[0, 0, 1]+arr[2, 2, 1] - arr[0, 2, 1] - arr[2, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
###################################################### 4. 计算方位信息  ##################################################


def compute_orientation(kp, octave_index, img, sift: CSift):
    keypoints_with_orientations = []
    # 除去组信息o，不知为何？莫非因为输入图像img已经是进行了降采样的图像，涵盖了o的信息？
    cur_scale = kp.size / (2**(octave_index))
    radius = round(sift.radius_factor*sift.scale_factor*cur_scale)  # 求取邻域半径
    weight_sigma = -0.5 / ((sift.scale_factor*cur_scale) ** 2)  # 高斯加权运算系数
    raw_histogram = np.zeros(sift.num_bins)  # 初始化方位数组
    cx = round(kp.pt[0]*img.shape[1])  # 获取极值点位置x
    cy = round(kp.pt[1]*img.shape)
