library(Hmisc)
library(tidyverse)
library(leaps)
library(randomForest)
library(doParallel)
library(Metrics)
library(Matrix)
library(xgboost)
library(e1071)

load("0.reduce.data.dim.RData")

###define function that use svm to return kappa
svm.return.kappa.fun=function(data.train,folds.vector,fold.k,cost,gamma) {
  svm.cv.kappa.vector<- foreach(kth.fold = 1:fold.k, .combine = cbind, .inorder = FALSE) %dopar% {
    fit.svm=e1071::svm(Response~., data=data.train[folds.vector!=kth.fold,], type = "C-classification", kernel="radial", cost=cost, gamma=gamma)
    pred.svm <- predict(fit.svm, newdata = data.train[folds.vector==kth.fold,])
    return(Metrics::ScoreQuadraticWeightedKappa(pred.svm,data.train[folds.vector==kth.fold,]$Response))
  }
  cat(paste(cost,"cost,",gamma,"gamma","\n"))
  return(mean(svm.cv.kappa.vector))
}

svm.return.kappa.fun(data.reduced.train2,folds,5,8,1/45)

##assign index for 5 folds

set.seed(134)
fold.k=5
folds <- sample(1:fold.k, nrow(data.reduced.train), replace = TRUE)

svm.cv.param <- list(cost = c(5,15,20),
                     gamma = 1/c(35,45,55)) %>%
  cross_df()
###grid search cost and gamma(parameter in radio kernel)
registerDoParallel(cores = 16)
svm.cv.kappa=svm.cv.param %>% mutate(kappa=unlist(pmap(svm.cv.param,svm.return.kappa.fun,data.train=data.reduced.train,folds.vector=folds,fold.k=fold.k)))


svm.cv.kappa %>% group_by(cost,gamma) %>% summarize(kappa.mean=mean(kappa)) %>% arrange(desc(kappa.mean))

###use best model to predict and save prediction
fit.svm.20gamma.0p0286gamma=svm(Response~., data=data.reduced.train, kernel="radial", cost=20, gamma=0.0286)
svm.finalModel.pred <- predict(fit.svm.20gamma.0p0286gamma, newdata = data.reduced.train)
save(svm.finalModel.pred, file = "3.pred.svm.rda")
###save image
rm(fit.rf.24.16,ntree.kappa)
save.image(file = "3.svm.RData")
load("3.svm.RData")






















