# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:49:42 2022

@author: ruiqi
"""

import joblib
import numpy as np
import collections
import arviz
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import os

txtfiles = []
for file in glob.glob("/Simulation/N500/m50/bkmr_mhmc_results/*"):
    txtfiles.append(file)
a1=joblib.load(txtfiles[0])
alls={}

for i in range(1,len(txtfiles)+1):
    txt=txtfiles[i-1]
    alls[i-1]=joblib.load(txt)
    
    
    
alls[0]
s=','.join(["alls[%d][key]"% i for i in range(0,len(txtfiles))])
s

d_comb={key:(eval(s)) for key in alls[0]}



delta_true=np.zeros(a1['gamma'][0].shape)
delta_true[:2,]=1.0


# first true delta
delta=np.stack(d_comb['gamma'])
first_true=[]
for i in range(0,delta.shape[0]):
    first_true.append(np.argwhere((delta_true == delta[i,:,:]).all(1))[0])
    
first_true
np.mean(first_true)
np.std(first_true)
timeused=np.array(first_true)/5000*1020.408
np.mean(timeused)
np.std(timeused)

