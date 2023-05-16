library(bkmr)
b<-c()
timediff<-timeused<-c()
truedelta=c(1,1,rep(0,200-2))
check1<-function(x){
  ifelse(all(x==truedelta),1,0)
  
}
for (i in 1:100) {
  a<-readRDS(paste0("/Simulation/N500/m200/fitkm_bkmr_knots_",i,".rds"))
  ss=apply(a$delta, 1,check1)
  b[i]<-which(ss==1)[1]
  timediff[i]<-as.numeric(difftime(a$time2,a$time1,units = "secs"))
  timeused[i]<-b[i]/a$iter*timediff[i]
  cat(i)
}
b
mean(b,na.rm = T)# 
sd(b,na.rm = T)
timediff
mean(timediff) # 
sd(timediff)
a$iter # 20000
mean(timeused,na.rm = T)
sd(timeused,na.rm = T)
