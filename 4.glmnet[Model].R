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

load("0.reduce.data.dim.RData")

data.reduced.train.smatrix=sparse.model.matrix(Response ~ .,data.reduced.train)[,-1]

###define function that use glmnet to return kappa, best lambda is selected inside the function
glmnet.return.kappa.fun=function(data.train,data.train.response,folds.vector,alpha) {
  glmnet.cv.kappa.vector<- foreach(kth.fold = 1:length(unique(folds.vector)), .combine = cbind, .inorder = FALSE) %dopar% {
    fit.lasso=glmnet::glmnet(data.train[folds.vector!=kth.fold,],data.train.response[folds.vector!=kth.fold],family="multinomial",alpha=alpha)
    pred.lasso=predict(fit.lasso,data.train[folds.vector==kth.fold,],type="class")
    kappa.lasso=sapply(1:ncol(pred.lasso),function(x) Metrics::ScoreQuadraticWeightedKappa(pred.lasso[,x],data.train.response[folds.vector==kth.fold]))
    kappa.max.id=order(kappa.lasso,decreasing=T)[1]
    return(c(kappa.cv=kappa.lasso[kappa.max.id],lambda.cv.selected=fit.lasso$lambda[kappa.max.id]))
  }
  cat(paste("alphs",alpha,"\n"))
  return(list(kappa.cv.mean=mean(glmnet.cv.kappa.vector["kappa.cv",]),
             lambda.cv.selected=glmnet.cv.kappa.vector["lambda.cv.selected",][glmnet.cv.kappa.vector["kappa.cv",]==mean(glmnet.cv.kappa.vector["kappa.cv",])]))
}

glmnet.return.kappa.fun(data.reduced.train.smatrix,data.reduced.train$Response,folds,0)

##assign index for 5 folds
set.seed(321)
fold.k=5
folds <- sample(1:fold.k, nrow(data.reduced.train.smatrix), replace = TRUE)

glmnet.cv.param <- list(alpha = c(0.25,0.5,0.75)) %>%
  cross_df()
###grid search alpha(weight on lasso and ridge) and lambda(beta's penalty size)
registerDoParallel(cores = 16)
glmnet.cv.kappa=glmnet.cv.param %>%
  mutate(glm.fun.return=pmap(glmnet.cv.param,glmnet.return.kappa.fun,data.train=data.reduced.train.smatrix,data.train.response=data.reduced.train$Response,folds.vector=folds))

glmnet.cv.kappa %>%
  group_by(alpha) %>% mutate(lambda.selected=unlist(glm.fun.return)[2],kappa.median=unlist(glm.fun.return)[1]) %>%
  dplyr::select(alpha,lambda.selected,kappa.median) %>%
  arrange(desc(kappa.median))

###use best model to predict and save prediction
fit.lasso.0p25alpha=glmnet(data.reduced.train.smatrix,data.reduced.train$Response,family="multinomial",alpha=0.25)
glmnet.finalModel.pred=factor(predict(fit.lasso.0p25alpha, data.reduced.train.smatrix, s = 0.0001563992, type = "class"))
save(glmnet.finalModel.pred, file = "4.pred.glmnet.rda")

###save image
rm(fit.lasso.0p0001563992lambda.0p25alpha)
save.image(file = "4.glmnet.RData")
load("4.glmnet.RData")




