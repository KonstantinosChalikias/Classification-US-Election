##############################
# Machine Learning Project - Classification
# Author: Konstantinos Chalikias
# Description: Analysis of US 2016 election results by county. Donald Trump 
##############################
# Load Libraries
library(devtools)
library(readxl)
library(MASS)
library(dplyr)
library(stringr)
library(ggplot2)
library(MVN)
library(biotools)
library(class)
library(tree)

#Laod Data
rm(list=ls())
setwd("C:/Users/xalik/desktop/Projects/AUEB/Machine Learning/Machine Learning 2")
data <- read_xlsx("stat machine learn project II.xlsx",
                  sheet =1 )
votes <- read_xlsx("stat machine learn project II.xlsx",
                   sheet =2)

##Data Transformations##
votes <- as.data.frame(votes)
data <- as.data.frame(data)
str(votes)
str(data)

#Make a data frame only with trump outcome (0 if he lost the county,else 1) 
county_results <- votes %>%
  distinct(state, county, fips) %>%
  left_join(
    votes %>%
      filter(party == "Republican") %>%
      group_by(state, county) %>%
      summarize(
        trump_fraction = sum(fraction_votes[candidate == "Donald Trump"], na.rm = TRUE),
        .groups = "drop"
      ),
    by = c("state", "county")
  ) %>%
  mutate(outcome = ifelse(trump_fraction > 0.5, 1, 2))

merged_data <- county_results %>% #Merge trump's data frame with county facts
  left_join(
    data %>% select(fips, area_name, state_abbreviation, everything()),
    by = "fips"  
  )

merged_data <- merged_data[complete.cases(merged_data), ]
colnames(merged_data)[8:57]

str(merged_data)
merged_data$outcome <- as.factor(merged_data$outcome) #After cleaning, 2712 counties remain for analysis 

#Plot (Other plots made on powerBI)
ggplot(merged_data, aes(x = trump_fraction)) +
  geom_histogram(binwidth = 0.05, fill = "red", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Trump's Vote Fraction by County",
       x = "Trump's Vote Fraction",
       y = "Number of Counties") 


####### Classification #######
predictors <- paste(colnames(merged_data)[8:57], collapse = " + ")
formula <- as.formula(paste("outcome ~", predictors))

#####LDA#####
lda_mod <- lda(formula, data = merged_data)
p <- predict(lda_mod)

#in sample accuracy 
in_acc_lda <- sum(diag(table(merged_data$outcome,p$class)))/length(merged_data$outcome)

#plots for LDA
df_plot <- data.frame(LD1 = p$x[, 1], Outcome = merged_data$outcome)
ggplot(df_plot, aes(x = LD1, fill = as.factor(Outcome))) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of LD1", x = "LD1", y = "Density", fill = "Outcome") 

ggplot(df_plot, aes(x = as.factor(Outcome), y = LD1, fill = as.factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Boxplot of LD1 by Outcome", x = "Outcome", y = "LD1", fill = "Outcome") 

#calculate Accuracy with train and test data
lda_acc <- NULL
for(i in 1:1000){
  ind <- sample(1:2712 , 2712*.8 , replace = F)
  train <- merged_data[ind,]
  test <- merged_data[-ind,]
  
  m1 <- lda(formula , data = train)
  p1 <- predict(m1 , newdata = test )
  t<- table(test$outcome,p1$class)
  lda_acc <- c(lda_acc,sum(diag(t))/sum(t))
}
mean(lda_acc) #mean accuracy ~~ 71%

###### glm #####
glm_mod <- glm(formula , data = merged_data , family = "binomial")
summary(glm_mod)
glmstep_mod <- step(glm_mod) #Step algorithm to select the best model (according to AIC)
summary(glmstep_mod)

#Models comparison
in_acc_glmfull <- sum(diag(table(merged_data$outcome ,round(glm_mod$fitted))))/length(merged_data$outcome)
in_acc_glmstep <- sum(diag(table(merged_data$outcome ,round(glmstep_mod$fitted))))/length(merged_data$outcome) #About the same accuracy (in sample testing)
anova(glmstep_mod,glm_mod , test = "Chisq") #Deviance 

#Out of sample Check
glmfull_acc <- NULL
for(i in 1:1000){
  ind <- sample(1:2712 , 2712*.8 , replace = F)
  train <- merged_data[ind,]
  test <- merged_data[-ind,]
  
  glm_mod <- glm(formula , data = train , family = "binomial")
  p1 <- predict(glm_mod , newdata = test[-c(1:7)]  , type = "response")
  t<- table(test$outcome,round(p1,0))
  glmfull_acc <- c(glmfull_acc,sum(diag(t))/sum(t))
}
mean(glmfull_acc) #Full model ~~ 71.1% acc

glmstep_acc <- NULL
for(i in 1:1000){
  ind <- sample(1:2712 , 2712*.8 , replace = F)
  train <- merged_data[ind,]
  test <- merged_data[-ind,]
  
  glm_mod <- glm(glmstep_mod$formula , data = train , family = "binomial")
  p1 <- predict(glm_mod , newdata = test[-c(1:7)]  , type = "response")
  t<- table(test$outcome,round(p1,0))
  glmstep_acc <- c(glmstep_acc,sum(diag(t))/sum(t))
}
mean(glmstep_acc) #step model acc ~~ 71.6%
##step model (with less variables) may perform slightly better, but not a significant difference


####### k-nn Nearest Neighbors #######
only_x <- merged_data[8:57]
scale_only_x <- scale(only_x)

#Find best k to use (out of sample testing)
full_res <- NULL #model for k 1:50 and ckeck accuracy
for(i in 1:100){ #100 times for each k to reduce randomness
  ind <- sample(1:2712 , 2712*.8 , replace = F)
  train <- scale_only_x[ind,]
  test <- scale_only_x[-ind,]
  res <- NULL
  for(tk in 1:50){
    k2 <- knn(train ,test , merged_data$outcome[ind] , k = tk )
    t <- table(merged_data$outcome[-ind] , k2)
    tr <- sum(diag(t))/length(merged_data$outcome[-ind])
    res <- c(res,tr)
  }
  full_res <- c(full_res,res)
}
full_res
resr <- NULL
for(i in 1:50){
  resr[i] <- mean(full_res[seq(i,500,50)])
}

df_res <- data.frame(k = 1:50, Accuracy = resr) #Accuracy for every k tested
ggplot(df_res, aes(x = k, y = Accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Accuracy for different values of k",
       x = "k",
       y = "Accuracy") 

head(order(resr ,decreasing = T)) #Best k to use, Arround 15 should be fine 

#Model with k = 15
k_mod <- knn(scale_only_x ,scale_only_x , merged_data$outcome , k =15 )

t <- table(merged_data$outcome,k_mod)
in_acc_knn <- sum(diag(t))/dim(merged_data)[1] #In sample accuracy

ind <- sample(1:2712 , 2712*.8 , replace = F)
train <- scale_only_x[ind,]
test <- scale_only_x[-ind,]

knn_acc <- NULL
for(i in 1:1000){
  ind <- sample(1:2712 , 2712*.8 , replace = F)
  train <- scale_only_x[ind,]
  test <- scale_only_x[-ind,]
  
  k2 <- knn(train ,test , merged_data$outcome[ind] , k = 11 )
  t <- table(merged_data$outcome[-ind] , k2)
  sum(diag(t))/length(merged_data$outcome[-ind])
  knn_acc <- c(knn_acc,sum(diag(t))/sum(t))
}
mean(knn_acc) #model acc 72%

####### All models Comparison #######
results <- data.frame(
  in_sample_accuracy = c(in_acc_lda  ,in_acc_glmfull, in_acc_glmstep , in_acc_knn),
  out_of_sample_accuracy = c(mean(lda_acc), mean(glmfull_acc) , mean(glmstep_acc) , mean(knn_acc))
)
rownames(results) <- c("LDA" , "GLM full" , "GLM step" , "K-nn")
colnames(results) <- c("In Sample Accuracy","Put of Sample Accuracy")
round(results,5)


### Most importand variables ###
sort(lda_mod$scaling)




