# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 21:04:13 2022

@author: fatih
"""

import pandas as pd
import numpy as np
from scipy import stats

def getfeature(data):
    fmean=np.mean(data)
    fstd=np.std(data)
    fmax=np.max(data)
    fmin=np.min(data)
    fkurtosis=stats.kurtosis(data)
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    fzero=zero_crosses.size/len(data)
    return fmean,fstd,fmax,fmin,fkurtosis,fzero
def extractFeature(raw_data,ws,hop,dfname):
    fmean=[]
    fstd=[]
    fmax=[]
    fmin=[]
    fkurtosis=[]
    fzero=[]
    flabel=[]
    for i in range(ws,len(raw_data),hop):
       m,s,ma,mi,k,z = getfeature(raw_data.iloc[i-ws+1:i,0])
       fmean.append(m)
       fstd.append(s)
       fmax.append(ma)
       fmin.append(mi)
       fzero.append(z)
       fkurtosis.append(k)
       
       flabel.append(dfname)
    rdf = pd.DataFrame(
    {'mean': fmean,
     'std': fstd,
     'max': fmax,
     'min': fmin,
     'kurtosis': fkurtosis,
     'zerocross':fzero,
     'label':flabel
    })
    return rdf
    

df0 = pd.read_csv('sabit.csv', header = None).iloc[:,:6]
df0_rdf=extractFeature(df0,20,10,"0")

df1 = pd.read_csv('kosma.csv', header = None).iloc[:,:6]
df1_rdf=extractFeature(df1,20,10,"1")

df2 = pd.read_csv('ziplama.csv', header = None).iloc[:,:6]
df2_rdf=extractFeature(df2,20,10,"2")

df3 = pd.read_csv('yukari.csv', header = None).iloc[:,:6]
df3_rdf=extractFeature(df3,20,10,"3")

df4 = pd.read_csv('asagi.csv', header = None).iloc[:,:6]
df4_rdf=extractFeature(df3,20,10,"4")

df = pd.concat([df0_rdf, df1_rdf, df2_rdf,df3_rdf,df4_rdf])

df.to_csv(r'project_features.csv', index = None)