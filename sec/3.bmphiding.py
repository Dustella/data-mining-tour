# read string from console
import os

# default offset should be 54


def get_binary_offset(data: bytes) -> int:
    # 获取图像文件中二进制数据的起始位置
    offset = data[10:11].hex(':').split(':')
    offset.reverse()
    return int(''.join(offset), 16)


def encrypt(img, message):
    # 将字符串添加到图像文件前部
    length = len(message)
    with open(img, mode='rb') as f:
        data = f.read()
        offset = get_binary_offset(data)
    new_index = hex(offset + length).replace('0x',
                                             '')
    print(f'new index: {new_index}')
    new_data = data[0:10]+bytes.fromhex(new_index) + data[11:offset] + \
        message.encode() + data[offset:]
    with open(f'{img}.new.bmp', mode='wb') as f:
        f.write(new_data)


def extract(img):
    # 从图像中提取字符串
    with open(img, mode='rb') as f:
        data = f.read()
    offset = get_binary_offset(data)
    message = data[54:offset]
    print("提取到的信息：")
    print(message)


if __name__ == '__main__':
    print("请输入要隐藏的信息：")
    message = input()

    encrypt("./sec/lenna.bmp", message)
    extract('./sec/lenna.bmp.new.bmp')
