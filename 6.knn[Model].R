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
library(caret)

load("0.reduce.data.dim.RData")

set.seed(313)
###search k(# of neighbors)
trctrl.knn <- trainControl(method = "cv")
fit.knn <- train(Response ~., data = data.reduced.train, method = "knn", metric = "kappa",
                 trControl=trctrl.knn,
                 preProcess = c("center", "scale"),
                 tuneLength = 5)

fit.knn$finalModel
###use best model to predict and save prediction
knn.finalModel.pred <- predict(fit.knn, newdata = data.reduced.train)
save(knn.finalModel.pred, file = "6.pred.knn.rda")

confusionMatrix(knn.finalModel.pred, data.reduced.train$Response )
###save image
rm(fit.rf.24.16,ntree.kappa)
save.image(file = "6.knn.RData")
load("6.knn.RData")












