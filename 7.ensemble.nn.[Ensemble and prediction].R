library(keras)

load("0.reduce.data.dim.RData")

load("1.pred.rf.rda")
load("2.pred.xgboost.rda")
load("3.pred.svm.rda")
load("4.pred.glmnet.rda")
load("5.pred.lda.rda")
load("6.pred.knn.rda")

###combine prediction from previous models
rf.pred.matrix=model.matrix( ~ rf.finalModel.pred - 1, data=data.frame(rf.finalModel.pred))
xgboost.pred.matrix=model.matrix( ~ xgboost.finalModel.pred - 1, data=data.frame(xgboost.finalModel.pred))
svm.pred.matrix=model.matrix( ~ svm.finalModel.pred - 1, data=data.frame(svm.finalModel.pred))
glmnet.pred.matrix=model.matrix( ~ glmnet.finalModel.pred - 1, data=data.frame(glmnet.finalModel.pred))
lda.pred.matrix=model.matrix( ~ lda.finalModel.pred - 1, data=data.frame(lda.finalModel.pred))
knn.pred.matrix=model.matrix( ~ knn.finalModel.pred - 1, data=data.frame(knn.finalModel.pred))

pred.combined=array(c(sapply(1:8, function(x) c(rf.pred.matrix[,x],
                                                xgboost.pred.matrix[,x],
                                                svm.pred.matrix[,x],
                                                glmnet.pred.matrix[,x],
                                                lda.pred.matrix[,x],
                                                knn.pred.matrix[,x]))),
                    dim=c(nrow(data.reduced.train),6,8))
response=array(as.numeric(data.reduced.train$Response)-1,dim=nrow(data.reduced.train))

###split into train and test

set.seed(129)
train.ind <- sample(c(TRUE, FALSE), nrow(data.reduced.train), rep = TRUE,prob=c(0.7,0.3))
test.ind <- !train.ind


###ensemble nn model
ensemble.nn.model <- keras_model_sequential()
ensemble.nn.model %>%
  layer_flatten(input_shape = c(6, 8)) %>%
  layer_dense(units = 28, activation = 'relu') %>%
  layer_dense(units = 8, activation = 'softmax')

ensemble.nn.model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

ensemble.nn.model %>% fit(pred.combined[train.ind,,], response[train.ind], epochs = 3)
###make prediction
pred.final <- ensemble.nn.model %>% predict_classes(pred.combined[test.ind,,])

###score prediction on test set
score <- ensemble.nn.model %>% evaluate(pred.combined[test.ind,,], response[test.ind])

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")
###save image
save.image(file = "7.ensemble.nn.final.pred.RData")
load("7.ensemble.nn.final.pred.RData")

