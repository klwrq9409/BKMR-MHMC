setwd("/Simulation/Summary_bkmr-mhmc/6ind/haccuracy/N500/m50")
files=list.files("/Simulation/6ind/N500/m50/results/bkmr-mhmc/")
index=gsub("results.joblib_","",files)
for (i in index) {
  outf<-paste0("read_data_",i,".py")
  write("import joblib
import collections
import matplotlib.pyplot as plt
import numpy as np
import json",outf,append=T)
write(sprintf("results=joblib.load('/Simulation/6ind/N500/m50/results/bkmr-mhmc/results.joblib_%s')",i),outf,append=T) 
write(sprintf("l = json.load(open('/Simulation/6ind/N500/m50/data/data_n500_m50_6ind_regular_%s.json'))",i),outf,append=T) 
write("X=np.array(l['X'])
M=l['M'][0]
N=X.shape[0]
y=np.array(l['y']).reshape((N,))
P=1
Z=np.array(l['Z'])",outf,append=T)
}
