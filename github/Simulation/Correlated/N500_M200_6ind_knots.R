#!/usr/bin/env Rscript
library(bkmr)

id <- as.numeric(Sys.getenv("SGE_TASK_ID"))
dat<-readRDS( paste0("/Simulation/Correlated/N500/m200/data_n500_m200_6ind_highcorr_",id,".rds"))
knots50 <- fields::cover.design(dat$Z, nd = 50)$design
y <- dat$y
Z <- dat$Z
X <- dat$X
r0=rgamma(dat$M,shape = 2,rate = 1)

assign(paste0("fitkm",id), kmbayes(y = y,Z = Z,X=X,groups = c(1,2,1,3:(dat$M-1)), iter = 100000, verbose = FALSE,varsel = TRUE,starting.values = list( beta = 0.3, sigsq.eps = 0.5, r =r0 , lambda = 3),control.params=list(r.prior="gamma",mu.lambda=10,sigma.lambda=10,mu.r=2,sigma.r=sqrt(2)),knots = knots50))

saveRDS(get(paste0("fitkm",id)),file = paste0("/Simulation/Correlated/N500/m200/results/bkmr/fitkm_bkmr_knots_",id,".rds"))




