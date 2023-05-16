source("/Simulation/SimDATA.R")
library(bkmr)
library(jsonlite)
library(kinship2)
kmat<-readRDS("/Simulation/family_cluster/kmat.RDS")
for (iii in 1:100) {
  N=500
  kmat2<-kmat[1:N,1:N]*2
  kmat2<-as.matrix(kmat2)
  id=row.names(kmat2)
  nid <- length(unique(id))
  TT <- matrix(0, length(id), nid)
  for (i in 1:nid) {
    TT[which(id == id[i]), i] <- 1
  }
  L1=t(chol(kmat2))
  tmp=L1%*%t(L1)
  TT2=TT%*%L1
  crossTT <- tcrossprod(TT2)
  crossTT2= TT%*%kmat2%*%t(TT)
  lambda2=0.8
  # set.seed(111)
  dat=SimData2(n=N,M = 50)
  dat$y<- dat$y+ as.numeric(TT%*%L1 %*% rnorm(nid,sd=lambda2))
  dat$crossTT<-crossTT
  dat$id<-id
  write_json(dat, path=paste0("/Simulation/family_cluster/data_n500_m50_fam_",iii,".json"))
  saveRDS(dat,file =paste0("/Simulation/family_cluster/data_n500_m50_fam_",iii,".rds") )
cat(iii)
}



