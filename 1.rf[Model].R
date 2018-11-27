library(Hmisc)
library(tidyverse)
library(leaps)
library(randomForest)
library(doParallel)
library(Metrics)


load("0.reduce.data.dim.RData")

###define function that use rf to return kappa
rf.return.kappa.fun=function(data.train,folds.vector,kth.fold,mtry,ntree.percore) {
  fit.rf<-foreach(ntreeRF=rep(ntree.percore,16), .packages="randomForest",.combine=randomForest::combine, .inorder=FALSE) %dopar% {
    randomForest(Response ~ ., data = data.train[folds.vector!=kth.fold,],mtry=mtry,nodeSize=1,ntree=ntreeRF)
  }
  pred.rf <- predict(fit.rf, newdata = data.train[folds.vector==kth.fold,])
  cat(paste(kth.fold,"fold",mtry,"possible variables at nodes,",ntree.percore*16,"trees","\n"))
  return(ScoreQuadraticWeightedKappa(pred.rf,data.train[folds.vector==kth.fold,]$Response))
}

###assign index for 5 folds
set.seed(321)
fold.k=5
folds <- sample(1:fold.k, nrow(data.reduced.train), replace = TRUE)

rf.cv.param <- list(kth.fold = 1:fold.k,
                    mtry = seq(8,8,by=2),
                    ntree.percore = c(8,10)) %>%
  cross_df()

###grid search ntree.percore and mtry(# variables to select from in each node)
registerDoParallel(cores = 16)
rf.cv.kappa=rf.cv.param %>% mutate(kappa=unlist(pmap(rf.cv.param,rf.return.kappa.fun,data.train=data.reduced.train,folds.vector=folds)))

rf.cv.kappa %>% group_by(mtry,ntree.percore) %>% summarize(kappa.mean=mean(kappa)) %>% arrange(desc(kappa.mean))
###use best model to predict and save prediction
fit.rf.8mtry.256ntree<-foreach(ntreeRF=rep(16,16), .packages="randomForest",.combine=randomForest::combine, .inorder=FALSE) %dopar% {
  randomForest(Response ~ ., data = data.reduced.train,mtry=8,nodeSize=1,ntree=ntreeRF)
}
rf.finalModel.pred <- predict(fit.rf.8mtry.256ntree, newdata = data.reduced.train)
save(rf.finalModel.pred, file = "1.pred.rf.rda")

###save image
rm(ntreeRF)
save.image(file = "1.rf.RData")
load("1.rf.RData")







