#!/usr/bin/env Rscript
library(bkmr)

id <- as.numeric(Sys.getenv("SGE_TASK_ID"))
dat<-readRDS( paste0("/Simulation/family_cluster/data_n500_m50_fam_",id,".rds"))
# knots100 <- fields::cover.design(dat$Z, nd = 100)$design
y <- dat$y
Z <- dat$Z
X <- dat$X
ids<-dat$id
r0=rgamma(dat$M,shape = 2,rate = 1)
assign(paste0("fitkm",id), kmbayes(y = y,Z = Z,X=X, iter = 100000, id=ids,verbose = FALSE,varsel = TRUE,starting.values = list( beta = 0.3, sigsq.eps = 0.5, r =r0 , lambda = 3, delta = 0),control.params=list(r.prior="gamma",mu.lambda=c(10,10),sigma.lambda=c(10,10),mu.r=2,sigma.r=sqrt(2))))

saveRDS(get(paste0("fitkm",id)),file = paste0("/Simulation/family_cluster/fitkm_bkmr_",id,".rds"))




