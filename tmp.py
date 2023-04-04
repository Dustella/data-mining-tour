from math import *

n = 1000
times = [10, 15, 100, 50]
idf_list = []
j = 0
for i in times:
    f = log10(n / (1 + times[j]))
    j += 1
    idf_list.append(f)

for i in range(len(idf_list)):
    print(round(idf_list[i], 4))
