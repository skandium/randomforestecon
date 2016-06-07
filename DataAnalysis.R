###################### Replication code for
###################### Random forest for high-dimensional non-linear forecasting
###################### Martin Lumiste
###################### June 7, 2016
###################### Part 2 - data analysis

rm(list=ls())
library(dplyr)
library(foreach)
library(doParallel)
library(randomForest)
library(quantmod)
library(e1071)
library(glmnet)
library(lmtest)
library(TSA)
library(forecast)
library(caret)
library(xtable)
setwd("C:\\Users\\martin\\Desktop\\thesis")
data.cov = read.csv("datacov.csv")[,2:144]
datah1 = read.csv("datah1.csv")[,2:144]
datah2 = read.csv("datah2.csv")[,2:144]
datah4 = read.csv("datah4.csv")[,2:144]
datah8 = read.csv("datah8.csv")[,2:144]
datah12 = read.csv("datah12.csv")[,2:144]
data.pred = list(datah1,datah2,datah4,datah8,datah12)
datach1 = read.csv("datach1.csv")[,2:144]
datach2 = read.csv("datach2.csv")[,2:144]
datach4 = read.csv("datach4.csv")[,2:144]
datach8 = read.csv("datach8.csv")[,2:144]
datach12 = read.csv("datach12.csv")[,2:144]
data.cpred = list(datach1,datach2,datach4,datach8,datach12)
rm(datah1,datah2,datah4,datah8,datah12,datach1,datach2,datach4,datach8,datach12)

# We want to get h-step ahead direct forecasts. Regress y_{t+h} on y_t, y_{t-1}, ..., y_{t-3}
#Here the argument s takes values 1 to 5, for 1,2,4,8,12 step ahead forecasts.
autoreg_ols = function(s,k){
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3))
  x = na.roughfix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  for (i in 0:(length(predictions)-1)){
    ar = lm(y[(1+i):(100+i)]~x[(1+i):(100+i),])
    temp = t(ar$coef) %*% c(1,x[(101+i),])
    predictions[i+1] = as.numeric(temp)
  }
  return(predictions)
}

#Absolute MSE function gives the average error.
mse_abs = function(s,k,fun){
  data.mse = as.data.frame(data.cpred[s])[,k]
  if (s==1){
    return(mean((data.mse[(101):199] - fun(s,k))^2))
  }
  if (s==2){
    return(mean((data.mse[(102):199] - fun(s,k))^2))
  }
  if (s==3){
    return(mean((data.mse[(104):199] - fun(s,k))^2))
  }
  if (s==4){
    return(mean((data.mse[(108):199] - fun(s,k))^2))
  }
  else {
    return(mean((data.mse[(112):199] - fun(s,k))^2))
  }
}

#Relative MSE gives it relative to AR(4)
mse = function(s,k,fun){
  return(mse_abs(s,k,fun)/mse_abs(s,k,autoreg_ols))
}

#Random forest
foress = function(s,k){
  #get data, replace NAs
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3),data.cov[,-k])
  x = na.roughfix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  #Rolling window forecast
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    forest = foreach(ntree=rep(1250,4), .combine=combine, .packages='randomForest') %dopar% {
      set.seed(i)
      randomForest(x[(1+i):(100+i),],y[(1+i):(100+i)],ntree=ntree)
    }
    predictions[i+1] = as.numeric(predict(forest,newdata=x[101+i,],type="response"))
  }
  return(predictions)
}

# cl = makeCluster(4)
# registerDoParallel(4)
# var = c(93,16,90)
# rfdmreal = matrix(NA,nrow=5,ncol=3)
# for (i in 1:5){
#   rfdmreal[i,] = vapply(var,FUN=dm, FUN.VALUE=1,s=i,fun=foress)
# }
# stopCluster(cl)
# registerDoSEQ()

#DFM
dfms = function(s,k){
  #get data, replace NAs
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,-k])
  x = na.roughfix(x)
  #We don't want to include own values in PC computation
  z = cbind(data.cov[,k],lag(data.cov[,k],1),lag(data.cov[,k],2),lag(data.cov[,k],3))
  z = na.roughfix(z)
  x = as.matrix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  #Rolling window forecast
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    pc = prcomp(x)$x
    xpc = pc[,1]
    #xpc = cbind(xpc,apply(xpc,2,Lag))#apply(xpc,2,Lag,2))#apply(xpc,2,Lag,3))
    #xpc = cbind(xpc,lag(xpc,1))
    ar = lm(y[(1+i):(100+i)]~xpc[(1+i):(100+i)]+z[(1+i):(100+i),])
    temp = t(ar$coef) %*% c(1,xpc[(101+i)],z[(101+i),])
    predictions[i+1] = as.numeric(temp)
  }
  return(predictions)
}

#SVM 
svmfuns = function(s,k){
  y = data.pred[[s]][,k]
  x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3),data.cov[,-k])
  x = na.roughfix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    svmval = svm(x[(1+i):(100+i),],y[(1+i):(100+i)])
    predictions[i+1] = as.numeric(predict(svmval,newdata=x[(101+i),]))
  }
  return(predictions)
}

#Tuned (naively) random forest

#Save tuning parameters for random forest separately
# para = function(s,k){
# y = as.data.frame(data.pred[s])[,k]
# x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3),data.cov[,-k])
# x = na.roughfix(x)
# ctrl = trainControl(method = "oob")
# outrf <- train(x[1:100,],y[1:100], method = "rf", tuneLength=10,trControl=ctrl)
# para = as.numeric(outrf$bestTune)
# return(para)
# }
# cl = makeCluster(4)
# registerDoParallel(4)
# para193 = para(1,93)
# para293 = para(2,93)
# para393 = para(3,93)
# para493 = para(4,93)
# para593 = para(5,93)
# 
# para116 = para(1,16)
# para216 = para(2,16)
# para316 = para(3,16)
# para416 = para(4,16)
# para516 = para(5,16)
# 
# para190 = para(1,90)
# para290 = para(2,90)
# para390 = para(3,90)
# para490 = para(4,90)
# para590 = para(5,90)
# stopCluster(cl)
# registerDoSEQ()
# paras = cbind(rbind(para193,para293,para393,para493,para593),
#               rbind(para116,para216,para316,para416,para516),
#               rbind(para190,para290,para390,para490,para590))
# write.csv(paras,file="rftuneparas.csv")

foresst = function(s,k){
  #get data, replace NAs
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3),data.cov[,-k])
  x = na.roughfix(x)
  paras = read.csv("rftuneparas.csv")[,2:4]
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}  
  if(s==4){h=8}
  if(s==5){h=12}
  if(k==93){t=1}
  if(k==16){t=2}
  if(k==90){t=3}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  #Rolling window forecast
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    forest = foreach(ntree=rep(125,4), .combine=combine, .packages='randomForest') %dopar% {
      set.seed(i)
      randomForest(x[(1+i):(100+i),],y[(1+i):(100+i)],ntree=ntree,mtry=paras[s,t])
    }
    predictions[i+1] = as.numeric(predict(forest,newdata=x[101+i,],type="response"))
  }
  return(predictions)
}

#Tuned DFM
dfs = function(s,k){
  #get data, replace NAs
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,-k])
  x = na.roughfix(x)
  #We don't want to include own values in PC computation
  z = cbind(data.cov[,k],lag(data.cov[,k],1),lag(data.cov[,k],2),lag(data.cov[,k],3))
  z = na.roughfix(z)
  x = cbind(x,x^2)
  x = as.matrix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  #Rolling window forecast
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    lasso = glmnet(x[(1+i):(100+i),],y[(1+i):(100+i)],alpha=0.5)
    wot = subset(lasso$lambda,lasso$df>=30)[1]
    coef = as.numeric(coef(lasso,s=wot))[-1]
    pc = prcomp(x[,which(coef!=0)])$x
    lasso2 = glmnet(pc[(1+i):(100+i),],y[(1+i):(100+i)],alpha=0.5)
    wat = subset(lasso2$lambda,lasso2$df>=2)[1]
    coef2 = as.numeric(coef(lasso2,s=wat))[-1]
    xpc = pc[,which(coef2!=0)]
    xpc = xpc[,1]
    #xpc = cbind(xpc,apply(xpc,2,Lag))#apply(xpc,2,Lag,2))#apply(xpc,2,Lag,3))
    #xpc = cbind(xpc,lag(xpc,1))
    ar = lm(y[(1+i):(100+i)]~xpc[(1+i):(100+i)]+z[(1+i):(100+i),])
    temp = t(ar$coef) %*% c(1,xpc[(101+i)],z[(101+i),])
    predictions[i+1] = as.numeric(temp)
  }
  return(predictions)
}

#Tuned SVM
svmfunst = function(s,k){
  y = data.pred[[s]][,k]
  x = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3),data.cov[,-k])
  x = na.roughfix(x)
  ctrl = trainControl(method = "cv")
  outsvm <- train(x[1:100,],y[1:100], method = "svmRadialCost", tuneLength=10,trControl=ctrl)
  para = as.numeric(outsvm$bestTune)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    svmval = svm(x[(1+i):(100+i),],y[(1+i):(100+i)],cost=para)
    predictions[i+1] = as.numeric(predict(svmval,newdata=x[(101+i),]))
  }
  return(predictions)
}

#Diebold-Mariano test
dm = function(s,k,fun){
  data.mse = as.data.frame(data.cpred[s])[,k]
  if (s==1){
    h=1
    e1 = (data.mse[(101):199] - fun(s,k))
    e2 = (data.mse[(101):199] - autoreg_ols(s,k))
  }
  if (s==2){
    h=2
    e1 = (data.mse[(102):199] - fun(s,k))
    e2 = (data.mse[(102):199] - autoreg_ols(s,k))
  }
  if (s==3){
    h=4
    e1 = (data.mse[(104):199] - fun(s,k))
    e2 = (data.mse[(104):199] - autoreg_ols(s,k))
  }
  if (s==4){
    h=8
    e1 = (data.mse[(108):199] - fun(s,k))
    e2 = (data.mse[(108):199] - autoreg_ols(s,k))
  }
  if (s==5){
    h=12
    e1 = (data.mse[(112):199] - fun(s,k))
    e2 = (data.mse[(112):199] - autoreg_ols(s,k))
  }
  dm = dm.test(e1,e2,alternative="less",h=h)
  return(as.numeric(dm$p.value))
}

## GDP, unemployment, inflation forecasting
which(colnames(data.cov)=="GDP251")
which(colnames(data.cov)=="LHUR")
which(colnames(data.cov)=="CPIAUCSL")

########## Tests
reset = function(s,k){
  y = as.data.frame(data.pred[s])[,k]
  x = cbind(data.cov[,-k])
  x = na.roughfix(x)
  #We don't want to include own values in PC computation
  z = cbind(data.cov[,k],lag(data.cov[,k],1),lag(data.cov[,k],2),lag(data.cov[,k],3))
  z = na.roughfix(z)
  #Add squared values
  x = cbind(x,x^2)
  x = as.matrix(x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  lasso = glmnet(x,y,alpha=0.5)
  wot = subset(lasso$lambda,lasso$df>=30)[1]
  coef = as.numeric(coef(lasso,s=wot))[-1]
  pc = prcomp(x[,which(coef!=0)])$x
  lasso2 = glmnet(pc,y,alpha=0.5)
  wat = subset(lasso2$lambda,lasso2$df>=2)[1]
  coef2 = as.numeric(coef(lasso2,s=wat))[-1]
  xpc = pc[,which(coef2!=0)]
  xpc = cbind(xpc, apply(xpc,2,Lag),apply(xpc,2,Lag,k=2),apply(xpc,2,Lag,k=3))
  #ar = lm(y[(1+i):(100+i)]~xpc+z[(1+i):(100+i),])
  test = resettest(y~xpc+z,power=2:10,type="regressor")
  return(list(test$statistic,test$p.value))
}

#Alternative tests considered and omitted from final paper
keenan = function(s,k){
  y = as.ts(as.data.frame(data.pred[s])[,k])
  test = Keenan.test(y)
  return(list(test$test.stat,test$p.value))
}

tsay = function(s,k){
  y = as.ts(as.data.frame(data.pred[s])[,k])
  test = Tsay.test(y)
  return(list(test$test.stat,test$p.value))
}

white = function(s,k){
  y = as.ts(as.data.frame(data.pred[s])[,k])
  set.seed(1)
  test = white.test(y,qstar=2,q=10)
  return(list(test$statistic,test$p.value))
}

terasvirta = function(s,k){
  y = as.ts(as.data.frame(data.pred[s])[,k])
  test = terasvirta.test(y)
  return(list(test$statistic,test$p.value))
}

# RF-PCDM test
foressX = function(s,k){
  #get data, replace NAs
  y = as.data.frame(data.pred[s])[,k]
  z = cbind(data.cov[,k], lag(data.cov[,k],n=1),lag(data.cov[,k],n=2),lag(data.cov[,k],n=3))
  z = na.roughfix(z)
  x = data.cov[,-k]
  x = na.roughfix(x)
  x = prcomp(x)$x
  cov = cbind(z,x)
  if(s==1){h=1}
  if(s==2){h=2}
  if(s==3){h=4}
  if(s==4){h=8}
  if(s==5){h=12}
  predictions = rep(NA,(length(y)-99-h))
  cat("Currently working on variable ", k, fill=TRUE)
  #Rolling window forecast
  for (i in 0:(length(predictions)-1)){
    cat("Iteration: ",i,fill=TRUE)
    forest = foreach(ntree=rep(125,4), .combine=combine, .packages='randomForest') %dopar% {
      set.seed(i)
      randomForest(cov[(1+i):(100+i),],y[(1+i):(100+i)],ntree=ntree)
    }
    predictions[i+1] = as.numeric(predict(forest,newdata=c(z[101+i,],x[101+i,]),type="response"))
  }
  return(predictions)
}

dmpc = function(s,k){
  data.mse = as.data.frame(data.cpred[s])[,k]
  if (s==1){
    h=1
    e1 = (data.mse[(101):199] - foress(s,k))
    e2 = (data.mse[(101):199] - foressX(s,k))
  }
  if (s==2){
    h=2
    e1 = (data.mse[(102):199] - foress(s,k))
    e2 = (data.mse[(102):199] - foressX(s,k))
  }
  if (s==3){
    h=4
    e1 = (data.mse[(104):199] - foress(s,k))
    e2 = (data.mse[(104):199] - foressX(s,k))
  }
  if (s==4){
    h=8
    e1 = (data.mse[(108):199] - foress(s,k))
    e2 = (data.mse[(108):199] - foressX(s,k))
  }
  if (s==5){
    h=12
    e1 = (data.mse[(112):199] - foress(s,k))
    e2 = (data.mse[(112):199] - foressX(s,k))
  }
  dm = dm.test(e1,e2,alternative="less",h=h)
  return(as.numeric(dm$p.value))
}

############## Let us try to reproduce something from table 2. 
#mse(1,93,svmfuns)
# [1] 1.696243 - same as in paper
#mse(1,93,dfms)
#[1] 1.022743
#cl = makeCluster(4)
#registerDoParallel(cl)
#mse(1,93,foress) - will take a while even on four cores
#stopCluster(cl)
#registerDoSEQ()
# [1] 1.125395

#Now table 3
#reset(1,93)
# [[1]]
# RESET 
# 1.358226 
# 
# [[2]]
# [1] 0.07957827

#Seems to work

