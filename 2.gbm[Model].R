library(Hmisc)
library(tidyverse)
library(leaps)
library(randomForest)
library(doParallel)
library(Metrics)
library(Matrix)
library(xgboost)


load("0.reduce.data.dim.RData")

###define function that use boost to return kappa

xgboost.return.kappa.fun=function(data.train,folds.vector,fold.k,eta,nrounds) {
  xgboost.cv.kappa.vector<- foreach(kth.fold = 1:fold.k, .combine = cbind, .inorder = FALSE) %dopar% {
    data.train.train.xgbmatrix=xgboost::xgb.DMatrix(Matrix::sparse.model.matrix(Response ~ .,data.train[folds.vector!=kth.fold,])[,-1],
                                           label = as.numeric(data.train[folds.vector!=kth.fold,]$Response)-1)
    data.train.test.xgbmatrix=xgboost::xgb.DMatrix(Matrix::sparse.model.matrix(Response ~ .,data.train[folds.vector==kth.fold,])[,-1],
                                          label = as.numeric(data.train[folds.vector==kth.fold,]$Response)-1)
    fit.xgboost <- xgboost::xgboost(data = data.train.train.xgbmatrix,num_class = 8,objective = "multi:softmax",eval_metric = "merror",
                           eta=eta,nrounds = nrounds,
                           early_stopping_rounds = 3)
    pred.xgboost <- predict(fit.xgboost, newdata = data.train.test.xgbmatrix)
    return(Metrics::ScoreQuadraticWeightedKappa(pred.xgboost,xgboost::getinfo(data.train.test.xgbmatrix,name="label")))
  }
  cat(paste("each tree contributes",eta,",",nrounds,"boosting iterations","\n"))
  return(mean(xgboost.cv.kappa.vector))
}

xgboost.return.kappa.fun(data.reduced.train,folds,5,0.3,40)

##assign index for 5 folds

set.seed(133)
fold.k=5
folds <- sample(1:fold.k, nrow(data.reduced.train), replace = TRUE)

xgboost.cv.param <- list(eta = c(0.3),
                         nrounds = c(40,50,60)) %>%
  cross_df()
###grid search eta(shrinkage at each step) and nrounds(number of trees)
registerDoParallel(cores = 16)
xgboost.cv.kappa=xgboost.cv.param %>% mutate(kappa=unlist(pmap(xgboost.cv.param,xgboost.return.kappa.fun,data.train=data.reduced.train,folds.vector=folds,fold.k=fold.k)))

xgboost.cv.kappa %>% group_by(eta,nrounds) %>% summarize(kappa.mean=mean(kappa)) %>% arrange(desc(kappa.mean))




data.train.xgbmatrix=xgb.DMatrix(sparse.model.matrix(Response ~ .,data.reduced.train)[,-1],
                                                label = as.numeric(data.reduced.train$Response)-1)
###use best model to predict and save prediction
fit.xgboost.0p3eta.50rounds<-xgboost(data = data.train.xgbmatrix,num_class = 8,objective = "multi:softmax",eval_metric = "merror",
                                     eta=0.3,nrounds = 50,
                                     early_stopping_rounds = 3)
xgboost.finalModel.pred <- factor(predict(fit.xgboost.0p3eta.50rounds, newdata = data.train.xgbmatrix)+1)
save(xgboost.finalModel.pred, file = "2.pred.xgboost.rda")

###save image
rm(ntree.kappa)
save.image(file = "2.xgboost.RData")
load("2.xgboost.RData")



