###################### Replication code for
###################### Random forest for high-dimensional non-linear forecasting
###################### Martin Lumiste
###################### June 7, 2016
###################### Part 1 - loading data

# Load in data and process it 
rm(list=ls())
library(plyr)
library(dplyr)
library(randomForest)
table1 <- read.csv2("C:/Users/martin/Desktop/thesis/table1.csv", stringsAsFactors=FALSE)
##Save variable names which we are going to transform
data = table1[,table1[6,]!=0]
pred = colnames(table1[,table1[6,]==2])
trans1 = colnames(data[,data[4,]==1])
trans2 = colnames(data[,data[4,]==2])
trans3 = colnames(data[,data[4,]==3])
trans4 = colnames(data[,data[4,]==4])
trans5 = colnames(data[,data[4,]==5])
trans6 = colnames(data[,data[4,]==6])
#Drop NAs and irrelevant information
data = data[1:611,]
data = data[10:611,]
rownames(data) = data[,1]
data = data[,2:93]
#Make monthly into quarterly data
data = data[seq(from=2,to=602,by=3),]

correct_decim <- function(x) {
  as.numeric(gsub(",", ".", x, fixed = TRUE))
}
mon <- colwise(correct_decim)(data)
rownames(mon) = rownames(data)
#Second dataset with quarterly data
table2 <- read.csv2("C:/Users/martin/Desktop/thesis/table2.csv", stringsAsFactors=FALSE)
##Save transformations
data2 = table2[,table2[5,]!=0]
pred2 = colnames(table2[,table2[5,]==2])
trans21 = colnames(data2[,data2[3,]==1])
trans22 = colnames(data2[,data2[3,]==2])
trans23 = colnames(data2[,data2[3,]==3])
trans24 = colnames(data2[,data2[3,]==4])
trans25 = colnames(data2[,data2[3,]==5])
trans26 = colnames(data2[,data2[3,]==6])
#Massage data
data2 = data2[1:209,]
data2 = data2[9:209,]
rownames(data2) = data2[,1]
data2 = data2[,2:52]
qua <- colwise(correct_decim)(data2)
rownames(qua) = rownames(data2)
full = cbind(mon,qua)
rm(data,data2,mon,qua,table1,table2)
#See which series need transforming
getnr = function(x){return(which(colnames(full)==x))}

t1 = as.numeric(sapply(trans1,getnr))
t2 = as.numeric(sapply(trans2,getnr))
t3 = as.numeric(sapply(trans3,getnr))
t4 = as.numeric(sapply(trans4,getnr))
t5 = as.numeric(sapply(trans5,getnr))
t6 = as.numeric(sapply(trans6,getnr))

t21 = as.numeric(sapply(trans21,getnr))
t22 = as.numeric(sapply(trans22,getnr))
t23 = as.numeric(sapply(trans23,getnr))
t24 = as.numeric(sapply(trans24,getnr))
t25 = as.numeric(sapply(trans25,getnr))
t26 = as.numeric(sapply(trans26,getnr))

#functions to transform series
tra1 = function(x){return(full[,x])}
tra2 = function(x){return(full[,x]-lag(full[,x],1))}
tra3 = function(x){return((full[,x] - lag(full[,x],1)) - (lag(full[,x],1) - lag(full[,x],2)))}
tra4 = function(x){return(log(full[,x]))}
tra5 = function(x){return(log(full[,x] / lag(full[,x],1)))}
tra6 = function(x){return(log(full[,x] / lag(full[,x],1)) - log(lag(full[,x],1) / lag(full[,x],2)))}
#And data transformations for predictor series
htra1 = function(x,h){return(lead(full[,x],h))}
htra2 = function(x,h){return(lead(full[,x],h)-full[,x])}
htra3 = function(x,h){return((lead(full[,x],h) - full[,x])/h - (full[,x] - lag(full[,x],1)))}
htra4 = function(x,h){return(log(lead(full[,x],h)))}
htra5 = function(x,h){return(log(lead(full[,x],h)/full[,x]))}
htra6 = function(x,h){return((log(lead(full[,x],h)/full[,x]))/h - log(full[,x]/lag(full[,x],1)))}
#Finally, we need a comparison series to calculate the MSE
ctra1 = function(x,h){return(full[,x])}
ctra2 = function(x,h){return(full[,x]-lag(full[,x],n=h))}
ctra3 = function(x,h){return((full[,x] - lag(full[,x],n=h))/h - (lag(full[,x],n=h) - lag(full[,x],n=(h+1))))}
ctra4 = function(x,h){return(log(full[,x]))}
ctra5 = function(x,h){return(log((full[,x])/lag(full[,x],n=h)))}
ctra6 = function(x,h){return((log((full[,x])/lag(full[,x],n=h)))/h - log(lag(full[,x],n=h)/lag(full[,x],n=(h+1))))}

#Transform the covariates
dat1 = sapply(t1,tra1)
colnames(dat1) = trans1
dat2 = sapply(t2,tra2)
colnames(dat2) = trans2
#dat3 = sapply(t3,tra3) There are no series which need 3rd transformation
#colnames(dat3) = trans3
dat4 = sapply(t4,tra4)
colnames(dat4) = trans4
dat5 = sapply(t5,tra5)
colnames(dat5) = trans5
dat6 = sapply(t6,tra6)
colnames(dat6) = trans6
#dat21 = sapply(t21,tra1)
#colnames(dat21) = trans21
#dat22 = sapply(t22,tra2)
#colnames(dat22) = trans22
#dat23 = sapply(t23,tra3)
#colnames(dat23) = trans23
#dat24 = sapply(t24,tra4)
#colnames(dat24) = trans24
dat25 = sapply(t25,tra5)
colnames(dat25) = trans25
dat26 = sapply(t26,tra6)
colnames(dat26) = trans26
data.cov = as.data.frame(cbind(dat1,dat2,dat4,dat5,dat6,dat25,dat26))
#Remove first 2 rows because of NAs due to some of the transformations
data.cov = data.cov[3:201,]
#Replace remaining NAs(very few) by column medians (Breiman).
data.cov = na.roughfix(data.cov)

#Need to do the same for series to be forecasted h = 1,2,4,8,12 steps ahead
#h=1 
hvec = c(1,2,4,8,12)

data.for = function(h){
wat1 = sapply(t1,htra1,h=h)
wat2 = sapply(t2,htra2,h=h)
#there are no series that need 3rd transformation
wat4 = sapply(t4,htra4,h=h)
wat5 = sapply(t5,htra5,h=h)
wat6 = sapply(t6,htra6,h=h)
#
wat25 = sapply(t25,htra5,h=h)
wat26 = sapply(t26,htra6,h=h)
mat = cbind(wat1,wat2,wat4,wat5,wat6,wat25,wat26)
colnames(mat) = c(trans1,trans2,trans4,trans5,trans6,trans25,trans26)
mat = mat[3:201,]
mat = na.roughfix(mat)
return(as.data.frame(mat))
}
data.pred = lapply(hvec,data.for)
datah1 = as.data.frame(data.pred[1])
datah2 = as.data.frame(data.pred[2])
datah4 = as.data.frame(data.pred[3])
datah8 = as.data.frame(data.pred[4])
datah12 = as.data.frame(data.pred[5])

data.cfor = function(h){
  wat1 = sapply(t1,ctra1,h=h)
  wat2 = sapply(t2,ctra2,h=h)
  #
  wat4 = sapply(t4,ctra4,h=h)
  wat5 = sapply(t5,ctra5,h=h)
  wat6 = sapply(t6,ctra6,h=h)
  #
  wat25 = sapply(t25,ctra5,h=h)
  wat26 = sapply(t26,ctra6,h=h)
  mat = cbind(wat1,wat2,wat4,wat5,wat6,wat25,wat26)
  colnames(mat) = c(trans1,trans2,trans4,trans5,trans6,trans25,trans26)
  mat = mat[3:201,]
  mat = na.roughfix(mat)
  return(as.data.frame(mat))
}
data.cpred = lapply(hvec,data.cfor)
datach1 = as.data.frame(data.cpred[1])
datach2 = as.data.frame(data.cpred[2])
datach4 = as.data.frame(data.cpred[3])
datach8 = as.data.frame(data.cpred[4])
datach12 = as.data.frame(data.cpred[5])

# Finally data is ready for processing

setwd("C:\\Users\\martin\\Desktop\\thesis")
write.csv(data.cov, file="datacov.csv")
write.csv(datah1, file="datah1.csv")
write.csv(datah2, file="datah2.csv")
write.csv(datah4, file="datah4.csv")
write.csv(datah8, file="datah8.csv")
write.csv(datah12, file="datah12.csv")
write.csv(datach1, file="datach1.csv")
write.csv(datach2, file="datach2.csv")
write.csv(datach4, file="datach4.csv")
write.csv(datach8, file="datach8.csv")
write.csv(datach12, file="datach12.csv")
