import pandas as pd

raw_data = pd.read_table(
    "./data_mining/3.Apriori/Transactions.txt", sep="\t",)

print(raw_data.columns[[0, 1, 0, 1]])
for i in raw_data.values:
    pass
    # print(i.dtype)
