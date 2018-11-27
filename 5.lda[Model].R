library(Hmisc)
library(tidyverse)
library(leaps)
library(randomForest)
library(doParallel)
library(Metrics)
library(Matrix)
library(xgboost)
library(e1071)
library(glmnet)
library(MASS)

load("0.reduce.data.dim.RData")

data.remove.Medical_History_23=data.reduced.train[,names(data.reduced.train)!="Medical_History_23"]
###find mean kappa for 5 folds. for exploration purpose only, not for optimization
set.seed(137)
fold.k=5
folds <- sample(1:fold.k, nrow(data.remove.Medical_History_23), replace = TRUE)
kappa.lda=rep(NA,fold.k)

for (kth.fold in 1:fold.k) {
  fit.lda=lda(Response ~ ., data=data.remove.Medical_History_23[folds!=kth.fold,],method="mle")
  pred.lda=predict(fit.lda, newdata = data.reduced.train[folds==kth.fold,])$class
  kappa.lda[kth.fold]=ScoreQuadraticWeightedKappa(pred.lda,data.reduced.train[folds==kth.fold,]$Response)
}

mean(kappa.lda)
###use best model(only model) to predict and save prediction
fit.lda.rmMedical_History_23=lda(Response ~ ., data=data.remove.Medical_History_23,method="mle")
lda.finalModel.pred=predict(fit.lda.rmMedical_History_23, newdata = data.remove.Medical_History_23)$class
save(lda.finalModel.pred, file = "5.pred.lda.rda")

rm(fit.rf.24.16,ntree.kappa)
save.image(file = "5.lda.RData")
load("5.lda.RData")


###save image
rm(fit.rf.24.16,fit.lda,ntree.kappa,pred.lda,kth.fold)
save.image(file = "img1.lda.RData")
load("img1.lda.RData")












