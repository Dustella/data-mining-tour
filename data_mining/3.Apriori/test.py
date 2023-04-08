import pandas as pd
from apriori import Apriori

raw_data = pd.read_table(
    "./data_mining/3.Apriori/Transactions.txt", sep="\t",)

cols = raw_data.columns
final_res = []
tmp_set = []

for line in raw_data.values:
    tmp_set.clear()
    for i in range(len(line)):
        if line[i] == 1:
            tmp_set.append(cols[i])
    final_res.append(set(tmp_set))

print(final_res)
a = Apriori(final_res, 600)
print(a.get_result())
