---
title: "Project"
author: "Fangjiang Yin"
date: "12/7/2017"
output: html_document
---
```{r}
falldata<-read.csv("/Users/wangmengyuan/Desktop/falldetection.csv")
falldata$ACTIVITY<-factor(falldata$ACTIVITY)
set.seed(12345)
as.data.frame(falldata)
table(falldata$ACTIVITY)
n=nrow(falldata)
train<-sample(n,(0.7*n))
traindata<-falldata[train,]
validationdata<-falldata[-train,]

#Multinomial Logistic Rregression 0- Standing 1- Walking 2- Sitting 3- Falling 4- Cramps 5- Running
library(nnet)
fit<-multinom(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION,data=traindata)
summary(fit)
pred1<-predict(fit,newdata=validationdata)
actual<-validationdata$ACTIVITY
cm<-table(actual,pred1)
cm
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
#fit2 for train
pred2<-predict(fit,newdata=traindata)
actual<-traindata$ACTIVITY
cm1<-table(actual,pred2)
cm1
acc<-(cm1[1,1]+cm1[2,2]+cm1[3,3]+cm1[4,4]+cm1[5,5]+cm1[6,6])/sum(cm1)
acc

#Neuralnet
library(caret)
library(neuralnet)
trainnnet<-traindata
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='0')
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='1')
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='2')
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='3')
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='4')
trainnnet<-cbind(trainnnet,traindata$ACTIVITY=='5')
#binarize the categorical output
names(trainnnet)[8]<-'Standing'
names(trainnnet)[9]<-'Walking'
names(trainnnet)[10]<-'Sitting'
names(trainnnet)[11]<-'Falling'
names(trainnnet)[12]<-'Cramps'
names(trainnnet)[13]<-'Running'
nn<-neuralnet(Standing+Walking+Sitting+Falling+Cramps+Running~SL+EEG+BP+HR+CIRCLUATION,data=trainnnet,hidden=c(4), act.fct = "logistic",linear.output = FALSE,lifesign = "minimal",stepmax=1e7)
plot(nn)
nnvalidation<-validationdata[,-2]
#predict1 for validation predict2 for train
nnpredict1<-compute(nn,nnvalidation[-1])$net.result
#put multiple binary output to catergorical output
maxidx<-function(arr){return(which(arr==max(arr)))}
idx<-apply(nnpredict1,c(1),maxidx)
prediction<-c('Standing','Walking','Sitting','Falling','Cramps','Running')[idx]
cm2<-table(nnvalidation$ACTIVITY,prediction)
cm2
acc<-(cm2[1,4]+cm2[2,5]+cm2[3,3]+cm2[4,2]+cm2[5,1])/sum(cm2)
acc
# nnpredict2<-compute(nn,traindata[-1])$net.result
# #put multiple binary output to catergorical output
# maxidx<-function(arr){return(which(arr==max(arr)))}
# idx<-apply(nnpredict2,c(1),maxidx)
# prediction<-c('Standing','Walking','Sitting','Falling','Cramps','Running')[idx]
# cm2<-table(traindata$ACTIVITY,prediction)
# cm2
# acc<-(cm2[1,3]+cm2[2,4]+cm2[3,2]+cm2[4,1])/sum(cm2)
# acc

#KNN
fun <- function(x) { 
  a <- mean(x) 
  b <- sd(x) 
  (x - a)/(b) } 
knntrain<-traindata
knnvalidation<-validationdata
knntrain[,-1] <- apply(knntrain[,-1], 2, fun)
knnvalidation[,-1] <- apply(knnvalidation[,-1], 2, fun)
#input does not include the prediction line
train_input <- as.matrix(knntrain[,-1])
train_output <- as.vector(knntrain[,1]) #vector!!
validate_input <- as.matrix(knnvalidation[,-1])
library(class)
kmax <- 10
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)
library(class)
for (i in 1:kmax){
  prediction <- knn(train_input, train_input,train_output, k=i)
  prediction2 <- knn(train_input, validate_input,train_output, k=i)
  CM1 <- table(knntrain$ACTIVITY, prediction)
  ER1[i] <-(CM1[1,1]+CM1[2,2]+CM1[3,3]+CM1[4,4]+CM1[5,5]+CM1[6,6])/sum(CM1) #train predication error
  CM2 <- table(knnvalidation$ACTIVITY, prediction2) 
  ER2[i] <- (CM2[1,1]+CM2[2,2]+CM2[3,3]+CM2[4,4]+CM2[5,5]+CM2[6,6])/sum(CM2) #validation predication erro
}
plot(ER2,type='l')
z <- which.min(ER2)
cat("Minimum Validation Error k:", z)
prediction <- knn(train_input, train_input,train_output, k=10)
prediction2 <- knn(train_input, validate_input,train_output, k=10)
table(knntrain$ACTIVITY, prediction)
# cat("Maxmum Accuracy:", 1-ER1[10])
table(knnvalidation$ACTIVITY, prediction2) 
cat("Maxmum Accuracy:", 1-ER2[10])

#treeeeeeeeeeeee
library(ISLR)
library(caret)
library(tree)
library(rpart)
library(rpart.plot)
treevalidation<-validationdata[,-2]
treetrain<-traindata[,-2]
tree<-rpart(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION,treetrain,method="class")
summary(tree)
rpart.plot(tree)
#predict
cm<-table(treevalidation$ACTIVITY, predict(tree,treevalidation,type="class"))
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
#prune
tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
bestcp <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
treetrain.pruned <- prune(tree, cp = bestcp)
rpart.plot(treetrain.pruned, extra=104, box.palette="GnBu",branch.lty=3, shadow.col="gray", nn=TRUE)
cm<-table(treevalidation$ACTIVITY, predict(treetrain.pruned,treevalidation,type="class"))
cm
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
# conf.matrix <- round(prop.table(table(treevalidation$ACTIVITY, predict(treetrain.pruned,treevalidation,type="class"))), 2)
# conf.matrix #0.19+0+0.09+0.03+0.10=0.41
```


```{r}
#Naive Bayes
library(e1071)
naivemodel<-naiveBayes(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION,data=traindata)
#predict with validation data
naivevalidation<-validationdata[,-2]
predict<-predict(naivemodel,newdata=naivevalidation[,-1])
cm<-table(naivevalidation$ACTIVITY,predict)
cm
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
```

```{r}
#Bagging 
library(randomForest)
bag.train=randomForest(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION,data=falldata,subset=train,mtry=6,importance=TRUE)
bagvalidation<-validationdata[,-2]
predicted = predict(bag.train,newdata=bagvalidation)
cm<-table(bagvalidation$ACTIVITY,predicted)
cm
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
#RandomForest 
rf.train=randomForest(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION,data=falldata,subset=train,mtry=3,importance=TRUE)
predicted = predict(rf.train,newdata=bagvalidation)
cm<-table(bagvalidation$ACTIVITY,predicted)
cm
acc<-(cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5]+cm[6,6])/sum(cm)
acc
#Boosting
library(gbm)
library(caret)
fitControl <- trainControl(method="repeatedcv",number=5,repeats=1,verboseIter=TRUE)
gbmFit <- train(ACTIVITY~SL+EEG+BP+HR+CIRCLUATION, data=traindata,method="gbm",trControl=fitControl,verbose=FALSE)
gbmFit
summary(gbmFit)
```


