wh <- read.csv("world_happiness2016.csv")
wh.features <- wh[, -c(1, 2, 3, 4, 5, 6)]

# make a variable where the happiest half = 1, least happy half = 0
wh.features$Happy.Class <- factor(ifelse(wh$Happiness.Rank <= 79, 1, 0))
wh.tree <- rpart(Happy.Class ~ ., data=wh.features)
fancyRpartPlot(wh.tree)

