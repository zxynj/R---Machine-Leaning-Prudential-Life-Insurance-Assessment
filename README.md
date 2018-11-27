# kaggle---Prudential-Life-Insurance-Assessment

The detail for this kaggle project is [here](https://www.kaggle.com/c/prudential-life-insurance-assessment#description).
TLDR: Use more than 120 variables(continuous, discrete and categorical) to predict the response variable(ordinal categorical).

## Data imputation
After excluding all variables with missing data more than 20%, there are four variables (Employment_Info_6 - cont, Medical_History_1 - disc, Employment_Info_4 - cont and Employment_Info_1 - cont) left with missing values. Then I regress each of them on other variables in the training set and use forward selection to select variables that explain the most. Finally, I impute the missing values using the prediction from the four regression.
R code is in [0.reduce.data.dim[Pre-processing].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/0.reduce.data.dim%5BPre-processing%5D.R).

## Data selection
We now have 117 variables which will create too much computation work in later models, so we need to select only part of them. I build a random forest model on the response variable using these 117 variables and select 17 variables which have the highest importance when building the model.
R code is in [0.reduce.data.dim[Pre-processing].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/0.reduce.data.dim%5BPre-processing%5D.R).

## Individual models
I build 6 individual models and use cross validation to tune their hyperparameters. Then I predict the training set's response variables using the best individual models. Now we have 6 sets of predicted response for our training set.
### Random forest
R code is in [1.rf[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/1.rf%5BModel%5D.R).
### Boosting
R code is in [2.gbm[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/2.gbm%5BModel%5D.R).
### Support vector machine
R code is in [3.svm[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/3.svm%5BModel%5D.R).
### Elastic-Net Regularized GLM - Multinomial logistic regression
R code is in [4.glmnet[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/4.glmnet%5BModel%5D.R).
### Linear Discriminant Analysis
R code is in [5.lda[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/5.lda%5BModel%5D.R).
### Linear Discriminant Analysis
R code is in [6.knn[Model].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/6.knn%5BModel%5D.R).

## Ensemble model
Instead of using the majority vote to ensemble the 6 individual models. I pass their predictions into a neural network model and train it to predict the correct response.
R code is in [7.ensemble.nn.[Ensemble and prediction].R](https://github.com/zxynj/kaggle---Prudential-Life-Insurance-Assessment/blob/master/7.ensemble.nn.%5BEnsemble%20and%20prediction%5D.R).

## Future work
1. Write markdown documents.
2. Since the data is imbalanced, we can assign cost to each response categories for better accuracy.
3. The evaluation metric is the quadratic weighted kappa. However, the loss function for some individual models are class prediction accuracy not kappa. Models might improve if a custom kappa loss function is passed into those models.
4. When training ensemble on the same training data used in individual models, there could be overfitting problem. We shall dig deeper into this.
