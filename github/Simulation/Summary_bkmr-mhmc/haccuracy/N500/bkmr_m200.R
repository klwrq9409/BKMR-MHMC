library(bkmr)
files=list.files(pattern = "*knots","/Simulation/N500/m200/bkmr_results/")

index<-gsub("fitkm_bkmr_knots_","",files)
index<-gsub(".rds","",index)

intercept=slope=rsq=timeused=c()
for (i in index) {
  fit<-readRDS(paste0("/Simulation/N500/m200/bkmr_results/fitkm_bkmr_knots_",i,".rds"))
  timeused[i]<-as.numeric(difftime(fit$time2,fit$time1,units = "mins"))
  hi_est <- ComputePostmeanHnew(fit)$postmean
  # hi_est
  HFun3 <- function(z, ind = 1:2) 4*plogis(1/4*(z[ind[1]] + z[ind[2]] + 1/2*z[ind[1]]*z[ind[2]]), 0, 0.3)
  # HFun3 <- function(z, ind = 1:4) 4*plogis(1/4*(z[ind[1]] + z[ind[2]] + 1/2*z[ind[1]]*z[ind[2]]), 0, 0.3)+4*plogis(1/4*(z[ind[3]] + z[ind[4]] + 1/2*z[ind[3]]*z[ind[4]]), 0, 0.3)
  hi_true <- apply(fit$Z, 1, HFun3)
  # plot(hi_true, hi_est, xlab = "True h", ylab = "Estimated h")
  
  lmres<-summary(lm(hi_est~hi_true))
  lmres
  intercept[i]<-lmres$coefficients[1,1]
  slope[i]<-lmres$coefficients[2,1]
  rsq[i]<-lmres$r.squared
  
  cat(i,"\r")
}
mean(intercept)
mean(slope)
mean(rsq,na.rm = T)
mean(timeused)

rsq
slope
