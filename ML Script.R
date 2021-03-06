# Machine Learning
rm(list=ls())

# Specify directory
setwd('C:/Users/uger7/OneDrive/Documents/GitHub/FSU_ML')

# Read data
D <- read.csv('master_panel_set_clean.csv')

# Select columns and change data types
library(tidyverse)
modeling_data <- D %>% 
  select(many_shooting_events, state_race_white, UR, CI, restr_laws, 
         hg_per_cap, summer_months) %>%
  mutate(many_shooting_events = factor(many_shooting_events), 
         summer_months = factor(summer_months))

# View data format
str(modeling_data)

# View the first 5 rows
head(modeling_data,5)

# Call packages
library(caret)
library(caTools)
library(margins)

# Set seed
set.seed(20)

# Split data
training_index <- sample.split(modeling_data, SplitRatio=0.7)
training <- modeling_data[training_index == T, ]
testing <- modeling_data[training_index == F, ]

# Set training parameters
mytrain<-trainControl(method = "cv", number = 5,
                        verboseIter = TRUE)

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

# Evaluating - logistic

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

# split the test set into a tune and test set
tuning_index <- sample.split(testing, SplitRatio=0.5)
tuning <- testing[tuning_index == T, ]
testing2 <- testing[tuning_index == F, ]

# Run logistic model through function
logistic_accuracy <- cutoffs_eval(logistic_model, tuning, 'many_shooting_events', 
                                  plot = F)
logistic_accuracy$model <- 'Logistic'

# Run random forest through function
rf_accuracy <- cutoffs_eval(random_forest, tuning, 'many_shooting_events', 
                            plot = F)
rf_accuracy$model <- 'Random Forest' 

# Combine the two
mapped_accuracy <- rbind(logistic_accuracy, rf_accuracy)

# Visualize the relationship between sensitivity, specficity, and accuracy
mapped_graph <- mapped_accuracy %>%
  gather(key, value, accuracy, sensitivity, specificity) %>%
  ggplot(aes(x = cutoff, y = value, color = key)) +
  facet_wrap(~model) +
  geom_point() + 
  geom_smooth() +
  ggtitle('Cutoff Relationships for ML Models')

# Call the graph
mapped_graph

png('cutoff_differnces_effect_on_accuracy.png')
  
# tuning the cutoff of the logistic never works as well as the RF -- just use .5
optimal_logistic_cutoff <- .5

# Optimal cutoff for rf
optimal_rf_cutoff <- min(rf_accuracy$cutoff[rf_accuracy$accuracy == 
                                          max(rf_accuracy$accuracy)])
# Print optimal cutoff for fr
optimal_rf_cutoff

# Evaluating - logit
p1 <- predict(logistic_model, testing2, type = "prob")
p_class1 <- as.factor(ifelse(p1[[2]] >= optimal_logistic_cutoff, 1, 0))
logistic_cm_opt <- confusionMatrix(p_class1, testing2$many_shooting_events)
logistic_cm_opt

# Evaluating - random forest
p2 <- predict(random_forest, testing2, type = 'prob')
p_class2 <- as.factor(ifelse(p2[[2]] >= optimal_rf_cutoff, 1, 0))
rf_cm_opt <- confusionMatrix(p_class2, testing2$many_shooting_events)
rf_cm_opt

# Variable importance - random forest
rf_imp <- varImp(random_forest, scale = FALSE)
rf_imp

# Hanging dot plot
plot(rf_imp, top = 6, main = 'Random Forest Feature Importance Score')
png('rf_variable_importances.png')
