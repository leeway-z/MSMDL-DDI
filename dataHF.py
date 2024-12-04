import itertools
from collections import defaultdict
from operator import neg
import random
import math
import pandas as pd
import numpy as np

df_drugs_smiles = pd.read_csv('data/drug_smiles.csv')
DRUG_TO_INDX_DICT = {indx: drug_id for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}
print(len(DRUG_TO_INDX_DICT))
newDrug = []


def sample_dict(d, sample_size):
    keys = random.sample(list(d.keys()), sample_size)
    return {key: d[key] for key in keys}


sample_size = len(DRUG_TO_INDX_DICT) // 5
sampled_dict = sample_dict(DRUG_TO_INDX_DICT, sample_size)
print(sampled_dict)


import pandas as pd
import json
# 文件读取
dataset = pd.read_csv('data/drug_smiles.csv')
print(dataset)
# 所需列名和新老列名映射关系
columns_json_str1 = '{"d1":"d1","d2":"d2","type":"type"}'
columns_dict1 = json.loads(columns_json_str1)
# columns_json_str2 = '{"Drug2_ID":"drug_id","Drug2":"smiles"}'
# columns_dict2 = json.loads(columns_json_str2)
# 列名重命名
df1 = pd.DataFrame(dataset, columns=columns_dict1.keys())
df1.rename(columns=columns_dict1, inplace=True)
# df2 = pd.DataFrame(dataset, columns=columns_dict2.keys())
# df2.rename(columns=columns_dict2, inplace=True)
# print(df2)
# df1.append(df2, ignore_index=True, verify_integrity=False, sort=None)
# print(df1)
# 将重命名之后的数据写入到文件
filepath1 = 'data/twosides.csv'
# filepath2 = 'data/twosides_smiles2.csv'
df_columns = pd.DataFrame([list(df1.columns)])
df_columns.to_csv(filepath1, mode='w', header=False, index=True)
# df_columns.to_csv(filepath2, mode='w', header=False, index=True)

# f1 = df1.drop_duplicates(keep='first')  # 去重
# f2 = df2.drop_duplicates(keep='first')  # 去重

df1.to_csv(filepath1, mode='a', header=False, index=True)

