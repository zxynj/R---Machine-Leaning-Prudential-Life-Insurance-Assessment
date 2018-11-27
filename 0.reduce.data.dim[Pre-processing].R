library(Hmisc)
library(tidyverse)
library(leaps)
library(randomForest)
library(doSNOW)
library(Metrics)

###load data
train <- read.table("E:/Self study/kaggle - prudential/input/all/train.csv", sep=",", header=TRUE)
test <- read.table("E:/Self study/kaggle - prudential/input/all/test.csv", sep=",", header=TRUE)

###convert to correct type
cat.var.names <- c(paste("Product_Info_", c(1:3,5:7), sep=""),
                   paste("Employment_Info_", c(2,3,5), sep=""),
                   paste("InsuredInfo_", 1:7, sep=""),
                   paste("Insurance_History_", c(1:4,7:9), sep=""),
                   paste("Family_Hist_", c(1), sep=""),
                   paste("Medical_History_", c(3:9,11:14, 16:23, 25:31, 33:41), sep=""))
cont.var.names <- c("Ins_Age", "Ht", "Wt", "BMI",
                    paste("Product_Info_", c(4), sep=""),
                    paste("Employment_Info_", c(1,4,6), sep=""),
                    paste("Insurance_History_", c(5), sep=""),
                    paste("Family_Hist_", c(2:5), sep=""))
disc.var.names <- c(paste("Medical_History_", c(1,2,10,15,24,32), sep=""),
                    paste("Medical_Keyword_", 1:48, sep=""))

train.cat <- as.data.frame(lapply(train[, cat.var.names], factor))
test.cat <- as.data.frame(lapply(test[, cat.var.names], factor))

train.cont <- train[, cont.var.names]
test.cont <- test[, cont.var.names]

train.disc <- train[, disc.var.names]
test.disc <- test[, disc.var.names]

response.cat <- data.frame(Response=factor(train[,"Response"]))

###study missing value percentage
comb=rbind(cbind(data.frame(train=1),train.cat,train.cont,train.disc),
           cbind(data.frame(train=0),test.cat,test.cont,test.disc))
  
missing.perc=colMeans(is.na(comb))
data.frame(var.name=names(missing.perc[missing.perc>0]),
      missing.perc=missing.perc[missing.perc>0]) %>%
  arrange(desc(missing.perc))

###drop variables with too many missing values
data.selected=comb %>%
  select(-c(names(missing.perc[missing.perc>0.2])))

  ### following variables need to be imputed
  ###Employment_Info_6 - cont
  ###Medical_History_1 - disc
  ###Employment_Info_4 - cont
  ###Employment_Info_1 - cont
  ###
  

###fw selection of Employment_Info_6 on other 10 variables using 30% as test

data.selected.naomit=na.omit(data.selected[,order(colnames(data.selected))])
  
set.seed(123)
train.ind <- sample(c(TRUE, FALSE), nrow(data.selected.naomit), rep = TRUE,prob=c(0.7,0.3))
test.ind <- !train.ind

test.mat <- model.matrix(Employment_Info_6 ~ ., data = data.selected.naomit[test.ind, ])
  
regfit.fwd <- regsubsets(Employment_Info_6 ~ ., data = data.selected.naomit[train.ind, ], nvmax = 10, method = "forward")

val.errors <- rep(NA, 10)
for(i in 1:10) {
  coefi <- coef(regfit.fwd, id = i)
  pred <- test.mat[ , names(coefi)] %*% coefi
  val.errors[i] <- mean((data.selected.naomit$Employment_Info_6[test.ind] - pred)^2)
}

plot(val.errors, type = "b")

summary(regfit.fwd)

  ###following variables are kept
  ###Ins_Age - cont
  ###Product_Info_4 - cont
  ###Employment_Info_1 - disc
  ###

a=lm(Employment_Info_6~
       Ins_Age+
       Product_Info_4+
       Employment_Info_1,
     data=data.selected.naomit)
summary(a)

###fw selection of Medical_History_1 on other 10 variables using 30% test
regfit.fwd <- regsubsets(Medical_History_1 ~ ., data = data.selected.naomit[train.ind, ], nvmax = 10, method = "forward")

val.errors <- rep(NA, 10)
for(i in 1:10) {
  coefi <- coef(regfit.fwd, id = i)
  pred <- test.mat[ , names(coefi)] %*% coefi
  val.errors[i] <- mean((data.selected.naomit$Medical_History_1[test.ind] - pred)^2)
}

plot(val.errors, type = "b")
summary(regfit.fwd,max.print=8)
  ###following variables are kept
  ###Medical_History_23.3 - cat
  ###Medical_History_41.3 - cat
  ###Medical_History_29.3 - cat
  ###
  
a=lm(Medical_History_1~
       Medical_History_23+
       Medical_History_29+
       Medical_History_41,
     data=data.selected.naomit)
summary(a)
###fw selection of Employment_Info_4 on other 10 variables using 30% test
regfit.fwd <- regsubsets(Employment_Info_4 ~ ., data = data.selected.naomit[train.ind, ], nvmax = 10, method = "forward")

val.errors <- rep(NA, 10)
for(i in 1:10) {
  coefi <- coef(regfit.fwd, id = i)
  pred <- test.mat[ , names(coefi)] %*% coefi
  val.errors[i] <- mean((data.selected.naomit$Employment_Info_4[test.ind] - pred)^2)
}

plot(val.errors, type = "b")

summary(regfit.fwd)

  ###following variables are kept
  ###Employment_Info_6 - cont
  ###Employment_Info_3 - cat
  ###
  
a=lm(Employment_Info_4~
       Employment_Info_6+
       Employment_Info_3,
     data=data.selected.naomit)
summary(a)
###fw selection of Employment_Info_1 on other 10 variables using 30% test
regfit.fwd <- regsubsets(Employment_Info_1 ~ ., data = data.selected.naomit[train.ind, ], nvmax = 10, method = "forward")

val.errors <- rep(NA, 10)
for(i in 1:10) {
  coefi <- coef(regfit.fwd, id = i)
  pred <- test.mat[ , names(coefi)] %*% coefi
  val.errors[i] <- mean((data.selected.naomit$Employment_Info_1[test.ind] - pred)^2)
}

plot(val.errors, type = "b")

summary(regfit.fwd)
  
  ###following variables are kept
  ###Employment_Info_6 - cont
  ###Employment_Info_3 - cat
  ###Product_Info_4 - cont
  ###

a=lm(Employment_Info_1~
       Employment_Info_6+
       Employment_Info_3+
       Product_Info_4,
     data=data.selected.naomit)
summary(a)

###use Hmisc to impute, drop Medical_History_23, Medical_History_41, Product_Info_3 since they dont have enough unique observations to bootstrap
impute_arg.Employment_Info_6=aregImpute(~Employment_Info_6+Ins_Age+Product_Info_4+Employment_Info_1,data = data.selected, n.impute = 5)

Medical_History_1.pred.extract=with(data.selected,data.frame(Medical_History_1,
                              Medical_History_23.3=ifelse(Medical_History_23==3,1,0),
                              Medical_History_41.3=ifelse(Medical_History_41==3,1,0),
                              Medical_History_29.3=ifelse(Medical_History_29==3,1,0)))
impute_arg.Medical_History_1=aregImpute(~Medical_History_1+Medical_History_23.3+Medical_History_41.3+Medical_History_29.3,data = Medical_History_1.pred.extract, n.impute = 5)

impute_arg.Employment_Info_4=aregImpute(~Employment_Info_4+Employment_Info_6+Employment_Info_3,data = data.selected, n.impute = 5)
impute_arg.Employment_Info_1=aregImpute(~Employment_Info_1+Employment_Info_6+Employment_Info_3+Product_Info_4,data = data.selected, n.impute = 5)

data.selected.imputed=data.selected[,order(colnames(data.selected))]
data.selected.imputed$Employment_Info_6[is.na(data.selected.imputed$Employment_Info_6)==T]=impute_arg.Employment_Info_6$imputed$Employment_Info_6[,5]
data.selected.imputed$Medical_History_1[is.na(data.selected.imputed$Medical_History_1)==T]=impute_arg.Medical_History_1$imputed$Medical_History_1[,5]
data.selected.imputed$Employment_Info_4[is.na(data.selected.imputed$Employment_Info_4)==T]=impute_arg.Employment_Info_4$imputed$Employment_Info_4[,5]
data.selected.imputed$Employment_Info_1[is.na(data.selected.imputed$Employment_Info_1)==T]=impute_arg.Employment_Info_1$imputed$Employment_Info_1[,5]

sum(is.na(data.selected.imputed))
###to reduce computation, we select part of the training variables using importance score in random forest 

data.selected.imputed.train=data.selected.imputed %>% filter(train==1) %>% mutate(Response=response.cat$Response) %>% select(-train)
data.selected.imputed.test=data.selected.imputed %>% filter(train==0)

rm(comb,data.selected,response.cat,test,test.cat,test.cont,test.disc,train,train.cat,train.cont,train.disc,
   cat.var.names,cont.var.names,disc.var.names,missing.perc,data.selected.imputed,impute_arg.Employment_Info_1,
   impute_arg.Employment_Info_4,impute_arg.Employment_Info_6,impute_arg.Medical_History_1,
   Medical_History_1.pred.extract)

set.seed(123)
train.ind <- sample(c(TRUE, FALSE), nrow(data.selected.imputed.train), rep = TRUE,prob=c(0.7,0.3))
test.ind <- !train.ind



registerDoSNOW(makeCluster(16))
###random forest roudn1
kappa=double(23)

for(i in 1:23){
  mtry=i*5
  fit.rf<-foreach(ntreeRF=rep(32,16), .packages="randomForest",.combine=randomForest::combine, .inorder=FALSE) %dopar% {
    randomForest(Response ~ ., data = data.selected.imputed.train[train.ind,],mtry=mtry,nodeSize=50,ntree=ntreeRF)
  }
  pred.rf <- predict(fit.rf, newdata = data.selected.imputed.train[test.ind,])
  kappa[i]=ScoreQuadraticWeightedKappa(pred.rf,data.selected.imputed.train[test.ind,]$Response)
  cat(mtry," ")
}

plot(5*(1:23),kappa,type="b")
###random forest round2 to zoom in
kappa.zoom=double(9)

for(i in 1:9){
  mtry=i+5
  fit.rf<-foreach(ntreeRF=rep(32,16), .packages="randomForest",.combine=randomForest::combine, .inorder=FALSE) %dopar% {
    randomForest(Response ~ ., data = data.selected.imputed.train[train.ind,],mtry=10,nodeSize=1,ntree=ntreeRF)
  }
  pred.rf <- predict(fit.rf, newdata = data.selected.imputed.train[test.ind,])
  kappa.zoom[i]=ScoreQuadraticWeightedKappa(pred.rf,data.selected.imputed.train[test.ind,]$Response)
  cat(mtry," ")
}

plot((1:9)+5,kappa.zoom,type="b")

###random forest round3 to zoom in even more

fit.rf.24.16<-foreach(ntreeRF=rep(24,16), .packages="randomForest",.combine=randomForest::combine, .inorder=FALSE) %dopar% {
  randomForest(Response ~ ., data = data.selected.imputed.train[train.ind,],mtry=10,nodeSize=1,ntree=ntreeRF,importance=TRUE)
}
pred.rf.24.16 <- predict(fit.rf, newdata = data.selected.imputed.train[test.ind,])
kappa.24.16=ScoreQuadraticWeightedKappa(pred.rf,data.selected.imputed.train[test.ind,]$Response)

imp=importance(fit.rf.24.16)
varImpPlot(fit.rf.24.16)

data.frame(imp)[order(data.frame(imp)$MeanDecreaseAccuracy,decreasing = T),]
data.frame(imp)[order(data.frame(imp)$MeanDecreaseGini,decreasing = T),]

ntree.kappa

plot(ntree.kappa,type="b")

rm(data.selected.imputed.train,data.selected.imputed.test,test.ind,train.ind)

###reduce data
data.reduced.train=data.selected.imputed.train %>%
  select(BMI,Employment_Info_1,Employment_Info_6,Ht,Ins_Age,InsuredInfo_3,InsuredInfo_6,Medical_History_1,Medical_History_2,Medical_History_23,
         Medical_History_30,Medical_History_4,Medical_Keyword_3,Product_Info_2,Product_Info_4,Wt,Response)
data.reduced.test=data.selected.imputed.test %>%
  select(BMI,Employment_Info_1,Employment_Info_6,Ht,Ins_Age,InsuredInfo_3,InsuredInfo_6,Medical_History_1,Medical_History_2,Medical_History_23,
         Medical_History_30,Medical_History_4,Medical_Keyword_3,Product_Info_2,Product_Info_4,Wt)

###save reduced dataset
save.image(file = "0.reduce.data.dim.RData")




