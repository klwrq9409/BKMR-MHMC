library(bkmr)
b<-c()
timediff<-timeused<-c()
truedelta=c(rep(1,6),rep(0,200-6))
check1<-function(x){
  ifelse(all(x==truedelta),1,0)
  
}
for (i in 1:100) {
  a<-readRDS(paste0("/Simulation/6ind/N500/m200/results/bkmr/fitkm_bkmr_knots_",i,".rds"))
  ss=apply(a$delta, 1,check1)
  b[i]<-which(ss==1)[1]
  # ss2=table(ss[which(ss==1)[1]:length(ss)])
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
# mean(b,na.rm = T)/a$iter*mean(timediff) # 190.7518
mean(timeused,na.rm = T)
sd(timeused,na.rm = T)

# length(which(is.na(b)))