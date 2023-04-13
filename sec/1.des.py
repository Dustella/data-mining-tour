# DES 算法的初始置换
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]


def permute(key, table):
    return [key[i - 1] for i in table]


# 测试
data = '0123456789ABCDEF'
# 十六进制转换二进制
key = ''.join([bin(int(i, 16)).replace('0b', '').rjust(4, '0') for i in data])
# 初始置换
key = permute(key, IP)
# 转换为十六进制
print(''.join([hex(int(''.join(i), 2)).replace('0x', '').upper()
      for i in zip(*[iter(key)] * 4)]))
