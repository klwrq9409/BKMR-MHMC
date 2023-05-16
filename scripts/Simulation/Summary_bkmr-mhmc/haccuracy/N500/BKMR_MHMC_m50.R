library(reticulate)
library(bkmr)
use_condaenv("/miniconda3/envs/bkmr_mhmc")
diff=diffsd=c()
intercept=slope=rsq=c()
for (i in 1:100) {
  source_python(paste0("/Simulation/Summary_bkmr-mhmc/haccuracy/N500/m50/read_data_",i,".py"))
  fit<-list()
  P=ncol(X)
  seqs=2001:5000
  fit$beta=results$beta[seqs,1:P,drop=F]
  fit$delta=results$gamma[seqs,,drop=F]
  lambda1=results$beta[seqs,P+2,drop=F]/results$beta[seqs,P+1,drop=F]
  # lambda2=results$beta[seqs,P+3,drop=F]/results$beta[seqs,P+1,drop=F]
  fit$lambda<-lambda1
  # fit$lambda<-cbind(lambda1,lambda2)
  fit$sigsq.eps=results$beta[seqs,P+1]
  # fit$r=results$gamma[seqs,,drop=F]*results$beta[seqs,(P+4):ncol(results$beta),drop=F]
  fit$r=results$gamma[seqs,,drop=F]*results$beta[seqs,(P+3):ncol(results$beta),drop=F]
  
  fit$iter=length(seqs)
  fit$y=as.numeric(y)
  fit$X=X
  fit$Z=Z
  fit$est.h=FALSE
  fit$family = "gaussian"
  class(fit)="bkmrfit"
  
  hi_est <- ComputePostmeanHnew(fit)$postmean
  # hi_est
  HFun3 <- function(z, ind = 1:2) 4*plogis(1/4*(z[ind[1]] + z[ind[2]] + 1/2*z[ind[1]]*z[ind[2]]), 0, 0.3)
  # HFun3 <- function(z, ind = 1:4) 4*plogis(1/4*(z[ind[1]] + z[ind[2]] + 1/2*z[ind[1]]*z[ind[2]]), 0, 0.3)+4*plogis(1/4*(z[ind[3]] + z[ind[4]] + 1/2*z[ind[3]]*z[ind[4]]), 0, 0.3)
  hi_true <- apply(Z, 1, HFun3)
  # plot(hi_true, hi_est, xlab = "True h", ylab = "Estimated h")
  
  lmres<-summary(lm(hi_est~hi_true))
  intercept[i]<-lmres$coefficients[1,1]
  slope[i]<-lmres$coefficients[2,1]
  rsq[i]<-lmres$r.squared
  # diff[i]=mean((hi_est-hi_true)^2)
  # diffsd[i]=sd((hi_est-hi_true)^2)
  cat(i,"\r")
}
mean(intercept)
mean(slope)
mean(rsq)

