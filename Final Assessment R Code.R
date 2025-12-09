#Install necessary packages
install.packages("rnaturalearth")
install.packages("rnaturalearthdata")
install.packages("sf")
install.packages("ggplot2")
install.packages("sp")
install.packages("plyr")
install.packages("viridis")
install.packages("dplyr")
install.packages("fastDummies")
install.packages("neuralnet")
install.packages("tidyverse")
install.packages("ROCR")
install.packages("reticulate")
install.packages("keras")
install.packages("pROC") 
install.packages("rpart")

#Load necessary libraries
library(tidyverse)
library(readr)
library(readxl)
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)
library(dplyr)
library(ggplot2)
library(sp)
library(plyr)
library(fastDummies)
library(neuralnet)
library(ROCR)
library(reticulate)
library(keras)
library(pROC)
library(rpart)


#Exploratory Data Analysis:

##Import data set
loans <- read.csv("~/loan_data.csv")
View(loans)

##Removing missing data
na.omit(loans)


##Histogram:
ggplot(loans, aes(x = credit_score)) + 
  geom_histogram(binwidth = 1, fill = "purple", color = "black") +
  labs(title = "Distribution of Credit Score")


#Classification:

##Logistic Regression:
model <- glm(loan_status ~ loan_int_rate,
             data = loans,
             family = binomial)


print(summary(model))

logistic_curve_plot <- ggplot(loans, aes(x = loan_int_rate, y = loan_status)) +
  
  geom_point(position = position_jitter(height = 0.05, width = 0), alpha = 0.2) +
  
  
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"), 
              se = TRUE, 
              color = "blue") +
  
  
  labs(title = "Logistic Regression Curve: Loan Status vs. Interest Rate",
       x = "Loan Interest Rate",
       y = "Probability of Loan Default (loan_status = 1)") +
  theme_minimal() +
  
  
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  
  geom_vline(xintercept = mean(loans$loan_int_rate, na.rm = TRUE), 
             linetype = "dashed", color = "red")


print(logistic_curve_plot)

##Deep Learning:

###Feedforward Neural Network (FNN) [sample of 3000 had to be used]:
colnames(loans)[colnames(loans) == "loan_status"] <- "Target_Status"


numerical_features <- c('person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                        'credit_score')
categorical_features <- c('person_gender', 'person_education', 'person_home_ownership', 
                          'loan_intent', 'previous_loan_defaults_on_file')


loans[numerical_features] <- scale(loans[numerical_features])


dummy_formula <- as.formula(paste("~ -1 + ", paste(categorical_features, collapse = " + ")))


dummy_vars <- model.matrix(dummy_formula, data = loans)


data_to_model <- cbind(loans[, c("Target_Status", numerical_features)], as.data.frame(dummy_vars))


set.seed(42)

subsample_indices <- sample(seq_len(nrow(data_to_model)), size = 3000)
data_subsample <- data_to_model[subsample_indices, ]


train_indices <- sample(seq_len(nrow(data_subsample)), size = floor(0.8 * nrow(data_subsample)))

train_data <- data_subsample[train_indices, ]
test_data <- data_subsample[-train_indices, ]


valid_names <- make.names(colnames(train_data), unique = TRUE)
colnames(train_data) <- valid_names
colnames(test_data) <- valid_names 


feature_names <- setdiff(colnames(train_data), "Target_Status") 
formula_string <- paste("Target_Status ~", paste(feature_names, collapse = " + "))
f <- as.formula(formula_string)


cat("Starting Model Training with SIMPLIFIED neuralnet (3,000 samples, Hidden: 5)...\n")
nn_model <- neuralnet(f, 
                      data = train_data, 
                      hidden = 5,  
                      linear.output = FALSE, 
                      err.fct = "ce", 
                      act.fct = "logistic", 
                      rep = 1) 


target_col_index <- which(colnames(test_data) == "Target_Status")
test_features <- test_data[, -target_col_index]

nn_prediction <- compute(nn_model, test_features)


predicted_class <- as.numeric(nn_prediction$net.result > 0.5)
actual_class <- test_data$Target_Status
accuracy <- mean(predicted_class == actual_class)

cat("\n--- Feedforward Neural Network Evaluation (Max Speed Attempt) ---\n")
cat("Test Accuracy:", round(accuracy, 4), "\n")

plot(nn_model)

##Decision Trees:
loans <- read.csv("loan_data.csv", stringsAsFactors = FALSE)


names(loans)[names(loans) == "loan_status"] <- "Target_Status"


loans <- loans %>%
  mutate(
    
    person_gender = as.factor(person_gender),
    person_education = as.factor(person_education),
    person_home_ownership = as.factor(person_home_ownership),
    loan_intent = as.factor(loan_intent),
    previous_loan_defaults_on_file = as.factor(previous_loan_defaults_on_file),
    
    
    Target_Status = factor(Target_Status, levels = c(0, 1), labels = c("No_Default", "Default"))
  )


loans <- na.omit(loans)


formula_OHE <- ~ . - Target_Status # Formula to use all columns except the target
X_matrix <- model.matrix(formula_OHE, data = loans)


X_matrix <- X_matrix[, -1]


final_df <- data.frame(Target_Status = loans$Target_Status, X_matrix)



set.seed(42)


train_indices <- sample(
  x = 1:nrow(final_df),
  size = floor(0.7 * nrow(final_df)),
  replace = FALSE
)

train_data <- final_df[train_indices, ]
test_data <- final_df[-train_indices, ]



model_formula <- Target_Status ~ .


dt_model <- rpart(
  formula = model_formula,
  data = train_data,
  method = "class",
  control = rpart.control(minsplit = 20, cp = 0.005, maxdepth = 10)
)




print("--- Decision Tree Structure (Textual) ---")
print(dt_model)


plot(dt_model, uniform = TRUE, main = "Classification Tree for Loan Status")


text(dt_model, use.n = TRUE, all = TRUE, cex = 0.7, pretty = 0) 




predictions <- predict(dt_model, newdata = test_data, type = "class")

# Create the confusion matrix (using base R 'table')
conf_matrix <- table(
  Predicted = predictions,
  Actual = test_data$Target_Status
)

print("--- Confusion Matrix (Predicted vs. Actual) ---")
print(conf_matrix)


TN <- conf_matrix["No_Default", "No_Default"]
FP <- conf_matrix["Default", "No_Default"]
FN <- conf_matrix["No_Default", "Default"]
TP <- conf_matrix["Default", "Default"]
Total <- sum(conf_matrix)


accuracy <- (TP + TN) / Total
precision_default <- TP / (TP + FP)
recall_default <- TP / (TP + FN) 
f1_score_default <- 2 * (precision_default * recall_default) / (precision_default + recall_default)



cat("\n--- Decision Tree Performance Metrics --- \n")
cat(sprintf("Overall Accuracy: %.4f \n", accuracy))
cat("---------------------------------- \n")
cat(sprintf("Default Class (1) Precision: %.4f \n", precision_default))
cat(sprintf("Default Class (1) Recall:    %.4f \n", recall_default))
cat(sprintf("Default Class (1) F1-Score:  %.4f \n", f1_score_default))







