---
title: "Practical Machine Learning"
output: html_document
---


# Analysis

### Read in training set & Data cleaning

First we read in the dataset. When inspectiting via summary(origData), not shown 
due to volume of output, we see that there are many variables with a 98% NA values.
For a more meaningful analysis we remove those variables, as well as username, timestamp and 'new_window'.
This also takes care of any near zero variance variables. We use the remaining 53 variables for analysis plus the outcome variable.

```{r}
library(caret)
origData <- read.csv("pml-training.csv", na.strings = c("NA", "") )
dim(origData)

cleanData <- origData[ , -grep("^X$|user_name|timestamp|new_window", names(origData)) ]

NAVars <- apply( cleanData, 2, function(x) sum(is.na(x))  )
table(NAVars)
cleanData <- cleanData[ , NAVars == 0 ]
dim(cleanData)

nearZeroVar(cleanData, saveMetrics = F)

```

### Partition into training and testsets

Next we partition the data into a training (70% of the data) and testset (30% of the data), to be used for crossvalidation.

```{r}
set.seed(123)
inTrain <- createDataPartition( cleanData$classe, p = .7, list = F )
training <- cleanData[ inTrain, ]
testing <- cleanData[ -inTrain, ]
dim(training)
dim(testing)
```


### Model Fit

```{r, cache=TRUE}

modFit <- train( classe ~ ., data = training, method = "gbm", verbose = F)
#modFit

modFitRf <- train( classe ~ ., data = training, method = "rf")


confusionMatrix(training$classe, predict( modFit, training ))

confusionMatrix(training$classe, predict( modFitRf, training ))

```

First we try a stochastic gradient boosting method (gbm), to utilize boosting, which should deliver high accuracy. Results were satisfactory, with relatively good accuracy in the training set (99.25%). 
To see if we can get better results we run a second model, in this case a random forrest model (rf). Accuracy was even better, with 100% correct classification in the training set. The perfect prediction could indicate overfitting, however. 

The error is expected to be higher out of sample, due to overfitting the data in the training set. For the GBM model we would expect an error greater than .75%. For the RF model we would expect an error somewhat greater than 0. 

When crossvalidating on the testing set it turns out that the RF still provides the better accuracy ( 99.81% vs 98.56% ). Both models provide very good predictions, but due to the near perfect accuracy we choose the RF model as the final model.

```{r}
confusionMatrix(testing$classe, predict( modFit, testing ))

confusionMatrix(testing$classe, predict( modFitRf, testing ))
```



### Testing Dataset


First we read in the dataset and apply the same data cleaning functions as before.


```{r}
origTestData <- read.csv("pml-testing.csv", na.strings = c("NA", "") )
dim(origTestData)
cleanTestData <- origTestData[ , -grep("^X$|user_name|timestamp|new_window", names(origTestData)) ]

cleanTestData <- cleanTestData[ , NAVars == 0 ]
dim(cleanTestData)

```

Then we predict the classes, using the random forrest model, for each of the test cases, and create text files via the supplied function. The model behaves perfectly, with 100% accuracy of predictions.


```{r}

answers <- predict(modFitRf, cleanTestData)

print(answers)


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


pml_write_files(answers)

```
















