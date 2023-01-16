import pandas as pd
import numpy as np
import os
#%%
df=pd.read_csv(r'meglasses1.csv')
#%%
dark_glasses_list=os.listdir(r"C:\Users\isaac\OneDrive\Escritorio\dark_glasse")
#%%
meta=pd.read_csv(r'meta_me_glasses.txt',sep=' ',names=['path','lentes_claros'])


#concat meta and df
df=pd.merge(df,meta,on='path')

# df['lentes_claros']=0
df['lentes_oscuros']=0
for x in range(len(df)):
    path=df['path'][x]
    if path in dark_glasses_list:
        df['lentes_claros'][x]=0
        df['lentes_oscuros'][x]=1
#%%
df.to_csv(r'meglasses2.csv',index=False)
