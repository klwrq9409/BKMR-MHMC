#!/usr/bin/env Rscript
source("SimDATA2.R")
library(jsonlite)
for (i in 1:100) {
  dat <- SimData2(n = 500, M = 50,beta.true = 0.3,hfun = 3)
  # for the bkmr-mhmc
  write_json(dat, path=paste0("Simulation/N500/m50/data_n500_m50_2ind_regular_",i,".json"))
  # for the original bkmr
saveRDS(dat,file =paste0("Simulation/N500/m50/data_n500_m50_2ind_regular_",i,".rds") )
}
