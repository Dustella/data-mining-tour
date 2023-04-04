# 对其余所有的非数值型属性，转化成数值型属性，并对缺失值进行众数填充。jiangl
dic = {'b': 0, 'a': 1}
dic.update({'u': 0, 'y': 1, 'l': 2})
dic.update({'g': 0, 'p': 1, 'gg': 2})
dic.update({'c': 0, 'q': 1, 'w': 2, 'i': 3, 'aa': 4, 'ff': 5,
            'k': 6, 'cc': 7, 'm': 8, 'x': 9, 'd': 10, 'e': 11, 'j': 12, 'r': 13})
dic.update({'v': 0, 'h': 1, 'bb': 2, 'ff': 3,
            'j': 4, 'z': 5, 'dd': 6, 'n': 7, 'o': 8})
dic.update({'t': 0, 'f': 1})
dic.update({'g': 0, 's': 1, 'p': 2})

print(dic)
