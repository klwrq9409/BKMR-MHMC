setwd("/Simulation/Summary_bkmr-mhmc/correlated/N500/m200/correlated/N500/m200")
files=list.files("/Simulation/Correlated/N500/m200/results/bkmr-mhmc/")
index=gsub("results.joblib_highcorr_m200_","",files)
for (i in index) {
  outf<-paste0("read_data_",i,".py")
  write("import joblib
import collections
import matplotlib.pyplot as plt
import numpy as np
import json",outf,append=T)
  write(sprintf("results=joblib.load('/Simulation/Correlated/N500/m200/results/bkmr-mhmc/results.joblib_highcorr_m200_%s')",i),outf,append=T) 
  write(sprintf("l = json.load(open('/Simulation/Correlated/N500/m200/data_n500_m200_6ind_highcorr_%s.json'))",i),outf,append=T) 
  write("X=np.array(l['X'])
M=l['M'][0]
N=X.shape[0]
y=np.array(l['y']).reshape((N,))
P=1
Z=np.array(l['Z'])",outf,append=T)
}
