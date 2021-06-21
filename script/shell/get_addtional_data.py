import os
import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import requests

title_list=[]
start=0

for i in tqdm(range(start,300),total=300):
    tmp=requests.get(f'https://catalog.data.gov/api/3/action/package_search?rows=1000&start={i*1000}')
    tmp_res=tmp.json()['result']['results']
    if len(tmp_res) == 0:
        break
    title_list.extend([obj['title'] for obj in tmp_res if len(obj['resources'])>0])
    start+=1

df=pd.DataFrame(zip(title_list,['']*len(title_list)),columns=['title','label']).drop('label',axis=1)
df.drop_duplicates(inplace=True)

df.to_csv('data/raw_data/addtional_data/gov_data.csv',index=False)