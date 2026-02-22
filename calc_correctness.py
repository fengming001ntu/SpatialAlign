import pandas as pd
import pickle
from tqdm import tqdm
import os
import numpy as np
import sys
import ast



df = pd.read_csv(sys.argv[1])

df["reward"] = df["reward"].apply(lambda x: ast.literal_eval(x))


result_list = []
for _, row in df.iterrows():
    prompt_id = row["prompt_id"]
    reward = np.asarray(row["reward"])
    result_list.append(reward)
    

value = np.asarray(result_list)

for TH in list(np.arange(0,1,0.1)):
    num_correct = np.sum(value >= TH)
    print(f"{num_correct/value.size:.3f}", end=',')
print()
