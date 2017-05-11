library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)

test <- read.csv("test.csv")
train <- read.csv("train.csv")

train.test <- train[sample(nrow(train), 300), ]
train.train <- train[!(train$Id %in% train.test$Id), ]

train.test <- train.test[, -c(1)]
train.train <- train.train[, -c(1)]

housing.tree <- rpart(SalePrice~., data=train.train)
housing.pred <- predict(housing.tree, train.test)
fancyRpartPlot(housing.tree)



# submission ID, House.Price

