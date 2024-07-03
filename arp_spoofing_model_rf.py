import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE

df = pd.read_csv('arp_logs.csv')

print(df.head())

# EDA
print(df.describe())

# Check for missing values 
print(df.isnull().sum())
df['Source_IP'].fillna('0.0.0.0', inplace=True)
df['Destination_IP'].fillna('0.0.0.0', inplace=True)

# visualize the distribution of numerical features
df.hist(bins=50, figsize=(20, 15))
plt.show()

# visualize correlations - Exclude non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# data preprocessing
# handle missing values if found
df.dropna(inplace=True)

# encode categorical features
label_encoder = LabelEncoder()
df['Source_MAC'] = label_encoder.fit_transform(df['Source_MAC'])
df['Source_IP'] = label_encoder.fit_transform(df['Source_IP'])
df['Destination_MAC'] = label_encoder.fit_transform(df['Destination_MAC'])
df['Destination_IP'] = label_encoder.fit_transform(df['Destination_IP'])
df['Protocol'] = label_encoder.fit_transform(df['Protocol'])

# define features and target variable
X = df[['Source_MAC', 'Source_IP', 'Destination_MAC', 'Destination_IP', 'Protocol']]
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check for class imbalance
print("Class distribution in training set:")
print(y_train.value_counts())

# handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

print("Best parameters found: ", grid_search.best_params_)


# Train with best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_res, y_train_res)

# Make predictions
y_pred_rf = best_rf.predict(X_test)

# Evaluate the model
print("\nRandomForest Classifier:")
print("\nConfusion Matrix:")
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(conf_matrix_rf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=1))

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf * 100:.2f}%")

# Feature importance plot for RandomForest
feature_importance = best_rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance (RandomForest)')
plt.show()

# Confusion Matrix Graph
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Spoof'], yticklabels=['Benign', 'Spoof'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (RandomForest)')
plt.show()

# Precision-recall curve for RandomForest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
plt.plot(recall_rf, precision_rf, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (RandomForest)')
plt.grid(True)
plt.show()

# F1 score curve for RandomForest
f1_scores_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
plt.figure(figsize=(8, 6))
plt.plot(recall_rf, f1_scores_rf, marker='.')
plt.xlabel('Recall')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve (RandomForest)')
plt.grid(True)
plt.show()

# AUC curve for RandomForest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (RandomForest)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Add a scatter plot for model predictions
plt.figure(figsize=(10, 6))
correct = (y_test == y_pred_rf)
incorrect = ~correct
plt.scatter(range(len(y_test[correct])), y_test[correct], color='blue', label='Correct', alpha=0.6)
plt.scatter(range(len(y_test[incorrect])), y_pred_rf[incorrect], color='red', label='Incorrect', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.title('Correct vs Incorrect Predictions (RandomForest)')
plt.legend()
plt.show()
