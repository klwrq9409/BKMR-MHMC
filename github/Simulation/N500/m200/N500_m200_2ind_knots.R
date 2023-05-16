#!/usr/bin/env Rscript
library(bkmr)
# for the cluster array job run; id corresponds to the index
id <- as.numeric(Sys.getenv("SGE_TASK_ID"))
dat<-readRDS( paste0("Simulation/N500/m200/data_n500_m200_2ind_regular_",id,".rds"))
knots50 <- fields::cover.design(dat$Z, nd = 50)$design
y <- dat$y
Z <- dat$Z
X <- dat$X
r0=rgamma(dat$M,shape = 2,rate = 1)
assign(paste0("fitkm",id), kmbayes(y = y,Z = Z,X=X, iter = 40000, verbose = FALSE,varsel = TRUE,starting.values = list( beta = 0.3, sigsq.eps = 0.5, r =r0 , lambda = 3, delta = 0),control.params=list(r.prior="gamma",mu.lambda=10,sigma.lambda=10,mu.r=2,sigma.r=sqrt(2)),knots = knots50))
saveRDS(get(paste0("fitkm",id)),file = paste0("/Simulation/N500/m200/bkmr_results/fitkm_bkmr_knots_",id,".rds"))




