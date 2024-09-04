# Load required libraries
library(caret)
library(randomForest)
library(ggplot2)

# Load datasets
traindata <- read.csv("C:/Users/admin/Downloads/kdd_train/kdd_train.csv")
testdata <- read.csv("C:/Users/admin/Downloads/kdd_test/kdd_test.csv")

# Function to group infrequent categories
group_infrequent <- function(x, threshold = 10) {
  freq_table <- table(x)
  levels_to_keep <- names(freq_table[freq_table >= threshold])
  factor(ifelse(x %in% levels_to_keep, as.character(x), "Other"))
}

# Apply grouping and factor conversion for training data
traindata$protocol_type <- as.factor(traindata$protocol_type)
traindata$service <- group_infrequent(traindata$service)
traindata$flag <- group_infrequent(traindata$flag, threshold = 5)
traindata$labels <- as.factor(ifelse(traindata$labels == "normal", "normal", "attack"))

# Apply grouping and factor conversion for testing data
testdata$protocol_type <- as.factor(testdata$protocol_type)
testdata$service <- group_infrequent(testdata$service)
testdata$flag <- group_infrequent(testdata$flag, threshold = 5)
testdata$labels <- as.factor(ifelse(testdata$labels == "normal", "normal", "attack"))

# Drop columns with too many levels
traindata <- traindata[, !(names(traindata) %in% c("service"))]
testdata <- testdata[, !(names(testdata) %in% c("service"))]

# Ensure that 'labels' column is a factor and there are no missing values
traindata <- na.omit(traindata)
testdata <- na.omit(testdata)

# Normalize 'count' to 0-5 range
normalize_feature <- function(x, old_min, old_max, new_min, new_max) {
  (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
}

traindata$count_normalized <- normalize_feature(traindata$count, old_min = 0, old_max = 250, new_min = 0, new_max = 5)
testdata$count_normalized <- normalize_feature(testdata$count, old_min = 0, old_max = 250, new_min = 0, new_max = 5)

# Align factor levels between training and test data
testdata$protocol_type <- factor(testdata$protocol_type, levels = levels(traindata$protocol_type))
testdata$flag <- factor(testdata$flag, levels = levels(traindata$flag))

# Train the Random Forest model
set.seed(123)
rfmodel <- randomForest(labels ~ ., data = traindata, ntree = 100, mtry = 3, importance = TRUE)

# Print model summary and feature importance
print(rfmodel)
importance(rfmodel)

# Plot feature importance
importance_df <- as.data.frame(importance(rfmodel))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Feature") +
  ylab("Mean Decrease in Gini") +
  ggtitle("Feature Importance in Random Forest Model") +
  theme_minimal()

# Predict on the test dataset
predictions <- predict(rfmodel, newdata = testdata)

# Create confusion matrix
conf_matrix <- confusionMatrix(predictions, testdata$labels)

# Convert confusion matrix to data frame
conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))

# Plot the heatmap
ggplot(data = conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  geom_text(aes(label = Freq), vjust = 1) +
  xlab("Predicted Label") +
  ylab("True Label") +
  ggtitle("Confusion Matrix Heatmap") +
  theme_minimal()
