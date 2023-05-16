#!/usr/bin/env Rscript
source("/Simulation/SimDATA.R")
library(jsonlite)
for (i in 1:100) {
  dat <- SimData2(n = 500, M = 50,beta.true = 0.3,hfun = 5)
  write_json(dat, path=paste0("/Simulation/6ind/N500/m50/data/data_n500_m50_6ind_regular_",i,".json"))
saveRDS(dat,file =paste0("/Simulation/6ind/N500/m50/data/data_n500_m50_6ind_regular_",i,".rds") )
}
