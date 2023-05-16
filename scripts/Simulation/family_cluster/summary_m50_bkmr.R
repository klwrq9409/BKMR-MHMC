library(bkmr)
b<-c()
timediff<-timeused<-c()
truedelta=c(1,1,rep(0,50-2))
check1<-function(x){
  ifelse(all(x==truedelta),1,0)
  
}
lamb2<-c()
tau<-c()
sigma<-c()
for (i in 1:100) {
  a<-readRDS(paste0("/Simulation/family_cluster/fitkm_bkmr_",i,".rds"))
  ss=apply(a$delta, 1,check1)
  b[i]<-which(ss==1)[1]
  timediff[i]<-as.numeric(difftime(a$time2,a$time1,units = "secs"))
  timeused[i]<-b[i]/a$iter*timediff[i]
  lamb2[i]<-mean(a$lambda[20000:50000,2])
  sigma[i]<-mean(a$sigsq.eps[20000:50000])
  tau[i]<-mean(a$lambda[20000:50000,2]*a$sigsq.eps[20000:50000])
  cat(i)
}
b
mean(b,na.rm = T)# 530.67
sd(b,na.rm = T)
timediff
mean(timediff) 
sd(timediff,na.rm = T)# 11425.6
a$iter # 50000
mean(timeused,na.rm = T)
sd(timeused,na.rm = T)

lamb2
b
mean(lamb2)
mean(tau)
sd(tau)
tau
# MSE
mean((tau-0.64)^2)

mean(sigma)
sd(sigma)
