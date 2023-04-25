# read string from console
import os


def encrypt(img, message):
    # 将字符串添加到图像文件前部
    length = len(message)
    with open(img, mode='rb') as f:
        data = f.read()
        print(data[:20])


def extract(img):
    # 从图像中提取字符串
    length = img[0, 0, 0]
    message = ""
    for i in range(length):
        message += chr(img[0, 0, i+1])
    return message


if __name__ == '__main__':
    # print("请输入要隐藏的信息：")
    # message = input()
    #
    encrypt("./sec/coffeecat.bmp.bin", '')
