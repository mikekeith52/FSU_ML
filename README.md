
# FSU_ML
This repository includes the lesson on Machine Learning I give annually to the Master's of Applied Economics students at Florida State.

## Overview
I am asked annually to teach a single introduction to Machine Learning lecture to the new cohort at Florida State. The students have backgrounds typically in STATA, SAS, and R; Logistic Modeling; and economic interpretations of societal trends. Building upon that knowledge, this lecture hopes to offer a view to the students to understand how big data can be used to expand their skillsets.  

This is completed in R, with an identical application of the techniques and models in Microsoft Azure Machine Learning Studio. If the files in this directory are copied locally, the ML Script.R program should run and give identical results to what is described in this document.  

## Data
The data (master_panel_set_clean.csv) is a monthly aggregated state-by-state flat view of many indicators, including how many shooting events occurred in that month (from a dataset obtained on Kaggle--not a complete measure of all shooting events, but close enough to be modeled), how many gun licenses were issued for which types of firearms (according to the NICS database maintained by the FBI), how many permits for hunting and recreational shooting were outstanding in the state in a given month, a coincidental economic indicator available on the [St. Louis Federal Reserve website (FRED)](), the unemployment rate (also from FRED), and more. If you want a complete data dictionary or to know how these measures were obtained and applied to the master dataset, please contact me and I'd be happy to provide that information.  

For the sake of the class, I decided to keep only a few variables from the master dataset. These include:
- many_shooting_events: 1 if the state had many shooting events in the given year/month (I chose an arbitrary cutoff around the 25 percentile of shootings for all states in all months), 0 otherwise. **Dependent Variable**
- state_race_white: the percentage of the white population in the state. I used my own estimates so that this would change month-to-month, year-to-yer, from the 2010 census estimates
- UR: the [unemployment rate]() in the given state in the given month
- CI: the [coincidental economic indicator]() in the given state in the given month -- this was as the distance away from the mean of a logged first difference per state--to ensure stationarity and reduce auto-correlation in the panel dataset
- restr_laws: the number of restrictive gun laws in the given state in the given month
- hg_per_cap: the number of handgun permits issued in the state per capita
- summer_months: whether the given month is a summer month (June, July, August)  

|many_shooting_events|state_race_white|UR|CI|restr_laws|hg_per_cap|summer_months|
|---|---|---|---|---|---|---|
|1|0.7032|3.8|0.001583796|14|26.45265|0|
|1|0.6758|7.2|0.00009260545|10|49.67592|0|
|0|0.6378|4.7|0.002719951|14|25.03609|0|
|0|0.7855|3.7|0.001219661|11|30.58637|0|
|0|0.4656|4.5|0.00343214|27|12.04275|0|

This leaves 2,400 observations and 6 predictor variables to predict if many_shooting_events is 1 or 0 for a given obs.  

One of the students in the class, the first time I taught this, made the comment that there was probably some correlation between the statewide indicators applied to the dataset and the number of shootings in that state at a given time. Yes, this is probably true. This lesson isn't supposed to be the final word on how the indicators in this dataset affect gun violence, although the results are interesting. It is meant to be an introduction to Machine Learning techniques. That being said, the predictive power of the models created in the attached R script are siginificantly better than guessing, so there are some insights to be gleaned.  

## Process

### dplyr
First, the data needs to be prepared. This can be done easily with dplyr in the tidyverse suite of libraries:

```R
# Select columns and change data types
library(tidyverse)
modeling_data <- D %>% 
  select(many_shooting_events, state_race_white, UR, CI, restr_laws, 
         hg_per_cap, summer_months) %>%
  mutate(many_shooting_events = factor(many_shooting_events), 
         summer_months = factor(summer_months))
```

### Split data
We can use a 30/70 split:

```R
# Split data
training_index <- sample.split(modeling_data, SplitRatio=0.7)
training <- modeling_data[training_index == T, ]
testing <- modeling_data[training_index == F, ]
```

### Cross-validation with caret (default settings)
Using the caret package, we can choose how our models will be trained:

``` R
# Set training parameters
mytrain<-trainControl(method = "cv", number = 5,
                        verboseIter = TRUE)
```

This will automatically tune our model with 5-fold cross validation.

``` R
# Logit model
logistic_model <- train(many_shooting_events ~ ., data = training,
                        tuneLength = 4,
                        method = 'glm', family = 'binomial'(link = 'logit'),
                        trControl = mytrain)
summary(logistic_model)

# Random forest
random_forest <- train(many_shooting_events ~ ., data = training,
                       tuneLength = 4,
                       method = 'rf', trControl = mytrain)
summary(random_forest)
```

No parameters are tuned for the Logistic Model using this method. For the Random Forest, the mtry is trained and optimal mtry is found to be 4. This means four predictors will be used in each bagged tree.  

### Tuning optimal cutoff values
The students now know the concept of parameter tuning, but they should have the opportunity to see what is going on under the hood. That's why choose one more hypterparamter to tune on each model--the cutoff value. At what point should we round our probability predictions up to 1? The default is 0.5, but we can make that value whatever we want.  

To avoid overfitting, we want to tune on an out-of-sample dataset, but we don't want to do this on our test set. Therefore, we can resplit our test data into a tuning and testing set:

``` R
# split the test set into a tune and test set
tuning_index <- sample.split(testing, SplitRatio=0.5)
tuning <- testing[tuning_index == T, ]
testing2 <- testing[tuning_index == F, ]
```

I created this function in R to tune this hypeprameter:

``` R
# Create accuracy graph - whichever there is more of (0s or 1s) is the 'positive' class
# in this case, the positive class is 0 since there are more 0s than 1s
cutoffs_eval <- function(model, tuning_set, pred_col, iters = 1000, plot = T) {
  message('function evaluates cutoff metrics for two-class estimators')
  message('0/1 only accepted predicted values')
  if (plot == T) {
    library(tidyverse)
    message('plotting with tidyverse')
  }
  w <- which(names(tuning_set) == as.character(pred_col))
  for (i in 1:nrow(tuning_set)){
    if (tuning_set[[w]][i] != 1 & tuning_set[[w]][i] != 0) {
      stop(paste('non 0/1 binary class detected in', 
                 pred_col, 'column'))
    }
  }
  df <- data.frame(cutoff = rep(0, iters), accuracy = rep(0, iters), 
                   specificity = rep(0, iters), sensitivity = rep(0, iters))
  for (i in 1:iters) {
    p <- predict(model, tuning_set, type = 'prob')
    p_class <- ifelse(p[[2]] >= i/iters, 1, 0)
    evaluator <- data.frame(table(as.factor(tuning_set[[w]]), as.factor(p_class)))
    df$cutoff[i] <- i/iters
    if (sum(p_class == 1) == 0) {
      df$accuracy[i] <- sum(tuning_set[[w]] == 0)/length(tuning_set[[w]])
      if (sum(tuning_set[[w]] == 0) > sum(tuning_set[[w]] == 1)) {
        df$sensitivity[i] <- 1
        df$specificity[i] <- 0
      }
      else {
        df$sensitivity[i] <- 0
        df$specificity[i] <- 1 
      }
    }
    else if (sum(p_class == 0) == 0){
      df$accuracy[i] <- sum(tuning_set[[w]] == 1)/length(tuning_set[[w]])
      if (sum(tuning_set[[w]] == 0) > sum(tuning_set[[w]] == 1)) {
        df$sensitivity[i] <- 0
        df$specificity[i] <- 1
      }
      else {
        df$sensitivity[i] <- 1
        df$specificity[i] <- 0 
      }
    }
    else {
      df$accuracy[i] <- (evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 0, 3] + 
                           evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 1, 3]) /
        sum(evaluator$Freq)
      if (sum(tuning_set[[w]] == 0) > sum(tuning_set[[w]] == 1)) {
        df$sensitivity[i] <- evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 0, 3] / 
          (evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 0, 3] + 
             evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 0, 3])
        df$specificity[i] <- evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 1, 3] / 
          (evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 1, 3] + 
             evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 1, 3])
      }
      else {
        df$sensitivity[i] <- evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 0, 3] / 
          (evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 0, 3] + 
             evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 0, 3])
        df$specificity[i] <- evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 1, 3] / 
          (evaluator[evaluator[[2]] == 1 & evaluator[[1]] == 1, 3] + 
             evaluator[evaluator[[2]] == 0 & evaluator[[1]] == 1, 3])  
      }
    }
  }
  if (sum(tuning_set[[w]] == 0) > sum(tuning_set[[w]] == 1)) {
    message('successful evaluation - positive class is 0')
  }
  else {
    message('successful evaluation - positive class is 1')
  }
  if (plot == T) {
    print(df %>%
             gather(key, value, accuracy, sensitivity, specificity) %>%
             ggplot(aes(x = cutoff, y = value, color = key)) +
             geom_point() + 
             geom_smooth() +
             ggtitle('Cutoff Relationships for Model'))
  }
  return(df)
}
```

This function takes 1000 cutoff values between 0 and 1 and predicts on the tuning set on each cutoff value. The final result should leave us with 1 optimal cutoff value. This can be visualized:  

![]()

This is doubly good because it can give insight into the underlying function shapes. The Logistic model displays an S shape for both sensitivity and specificity -- indicative of how the functional exponential form of a logit model is bounded between 0 and 1. The Random Forest is strictly concave -- this is probably due to the decreasing marginal returns of bagging more and more trees. Although our first intuition is to choose the cutoff values where all three lines, sensitivity, specificity, and accuracy, overlap, this actually does not ensure optimal predictive power. The optimal cutoff value is at the absolute maximum of the accuracy curve.  

For the Logistic model, the accuracy curve is too sporadic to trust. We stick with the default value of 0.5 to make predictions. In the Random Forest, the optimal cutoff is 0.515.  

### Making predictions
Now that we have all hyperparameters tuned, we can make predictions. First, we want to evaluate our no-information rate--the metric to beat. In this dataset, that is 0.76. If we just guessed all 0s in the test set, we would be right 76% of the time.    

The total accuracy of the Logistic model on the test set is 0.78--barely better than the no-info rate (and its 95% interval overlaps with the no-info rate; this is not a great model).  

The total accuracy of the tuned Random Forest model is 0.86, significantly better. The students can see the power of Machine Learning vs. what a traditional econometric approach can accomplish.  

## Results
The good thing about simpler models like logit is that you can easily interpret results. We have odds ratio output for the Logistic Regression model. The interpretation of the model's coefficients can be calculated as their exponentiation minus 1. The direction of the magnitude is the same as the coefficient's estimated sign, so positive coefficients are indicative of a phenomenon that is more likely to result in many shooting events in a give month and vice versa. All p-values less than 0.05 can be considered statistically significant at the 95% confidence level.  

```
Coefficients:
                   Estimate Std. Error z value Pr(>|z|)    
(Intercept)      -6.885e-01  6.552e-01  -1.051 0.293352    
state_race_white -1.158e+00  4.927e-01  -2.351 0.018711 *  
UR                1.828e-01  5.761e-02   3.173 0.001510 ** 
CI               -1.457e+02  3.443e+01  -4.231 2.33e-05 ***
restr_laws       -3.663e-02  1.794e-02  -2.041 0.041215 *  
hg_per_cap        1.517e-02  5.687e-03   2.668 0.007623 ** 
summer_months1    5.618e-01  1.457e-01   3.856 0.000115 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 1524.7  on 1370  degrees of freedom
Residual deviance: 1448.3  on 1364  degrees of freedom
AIC: 1462.3

Number of Fisher Scoring iterations: 4
```

All of our predictors are significant at the 95% confidence level and the signs of the coefficients are intuitive--more hand guns issues in a month means greater likelihood of more gun violence, more restrictive gun laws means less chance, etc. You can go down the list and interpret each one.  

In the Random Forest, you can't derive such a simple directional analysis, but you can see which variables were the most important in making predictions. In this case, these were the following:

![]()