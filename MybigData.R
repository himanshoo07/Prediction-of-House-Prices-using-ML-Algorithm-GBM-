library(caret)
library(gbm)
library(tidyverse)
library(mlbench)

data(BostonHousing)

head(BostonHousing)

str(BostonHousing)

bh <- BostonHousing

#splitting into train test

set.seed(1234) # for reproducibility
train.index = createDataPartition(BostonHousing$medv, p = 0.7, list = F)
data.train = BostonHousing[train.index,]
data.test = BostonHousing[-train.index,]

#The distributions of the target variable are similar across the 2 splits:

summary(data.train$medv)
summary(data.test$medv)

#The predictor variables should be rescaled in each subset:
data.train.z =
  data.train %>% select(-medv) %>%
  mutate_if(is_logical,as.character) %>%
  mutate_if(is_double,scale) %>% data.frame()
data.test.z =
  data.test %>% select(-medv) %>%
  mutate_if(is_logical,as.character) %>%
  mutate_if(is_double,scale) %>% data.frame()

# add unscaled Y variable back
data.train.z$medv = data.train$medv
data.test.z$medv = data.test$medv

#A tuning grid can be set for these, and evaluation metric defined:
  ## Set up tuning grid
#caretGrid <- expand.grid(interaction.depth=c(1, 3, 5,6, 7), n.trees = (0:50)*50,
                        # shrinkage=c(0.01, 0.02),
                        # n.minobsinnode=2)
caretGrid <- expand.grid(interaction.depth=c(7,8,9), n.trees = (0:100)*50,
                           shrinkage=c(0.01, 0.02),
                          n.minobsinnode=2)


metric <- "RMSE"

#And the trainControl() function used to define the type of ‘in-model’ sampling and evaluation undertaken
#to iteratively refine the model. It generates a list of parameters that are passed to the train function that
#creates the model. Here a simple 10 fold cross validation will suffice:

trainControl <- trainControl(method="cv", number=10)

## run the model over the grid
set.seed(99)
gbm.caret <- train(medv ~ ., data=data.train.z, distribution="gaussian", method="gbm",
                   trControl=trainControl, verbose=FALSE,
                   tuneGrid=caretGrid, metric=metric, bag.fraction=0.75)


print(gbm.caret)
ggplot(gbm.caret)
# explore the results
names(gbm.caret)
# see best tune
gbm.caret[6]

# see grid results
head(data.frame(gbm.caret[4]))
# check
dim(caretGrid)
dim(data.frame(gbm.caret[4]))

## Find the best parameter combination
# put into a data.frame
grid_df = data.frame(gbm.caret[4])

head(grid_df)

# confirm best model and assign to params object
grid_df[which.min(grid_df$results.RMSE), ]

# assign to params and inspect
#params = grid_df[which.min(grid_df$results.RMSE), 1:4 ]
#params

## Create final model
# because parameters are known, model can be fit without parameter tuning
fitControl <- trainControl(method = "none", classProbs = FALSE)
# extract the values from params
gbmFit <- train(medv ~ ., data=data.train.z, distribution="gaussian", method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                ## only a single model is passed to the
                tuneGrid = data.frame(interaction.depth = 8,
                                      n.trees = 800,
                                      shrinkage = 0.02,
                                      n.minobsinnode = 2),
                metric = metric)

#The final model can be evaluated, in this case using same data as the model was trained on, and predictions
#evaluated against observations as in Figure 1:
  ## Prediction and Model evaluation
  # generate predictions
  pred = predict(gbmFit, newdata = data.test.z)
# plot these against observed
data.frame(Predicted = pred, Observed = data.test.z$medv) %>%
  ggplot(aes(x = Observed, y = Predicted))+ geom_point(size = 1, alpha = 0.5)+
  geom_smooth(method = "loess", col = "red")+
  geom_smooth(method = "lm")


# generate some prediction accuracy measures
postResample(pred = pred, obs = data.test.z$medv)

# examine variable importance
varImp(gbmFit, scale = FALSE)

print.varImp.10 <- function(x = vimp, top = 13) {
  printObj <- data.frame(as.matrix(sortImp(x, top)))
  printObj$name = rownames(printObj)
  printObj
}

df = data.frame(print.varImp.10(varImp(gbmFit)), method = "GBM")

df %>%
  ggplot(aes(reorder(name, Overall), Overall)) +
  geom_col(fill = "dark green") +
  facet_wrap( ~ method, ncol = 3, scales = "fixed") +
  coord_flip() + xlab("") + ylab("Variable Importance") +
  theme(axis.text.y = element_text(size = 7))
