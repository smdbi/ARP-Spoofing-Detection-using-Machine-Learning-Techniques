import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE


df = pd.read_csv('converted_output.csv')

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

# Encode categorical features
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

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

print("Best parameters found: ", grid_search.best_params_)



# Train with best parameters
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train_res, y_train_res)

# Make predictions
y_pred_xgb = best_xgb.predict(X_test)

# Evaluate the model
print("\nXGBoost Classifier:")
print("\nConfusion Matrix:")
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print(conf_matrix_xgb)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, zero_division=1))

# calculating accuracy
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy_xgb * 100:.2f}%")

# Feature importance plot for XGBoost
feature_importance = best_xgb.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='orange')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance (XGBoost)')
plt.show()

# Confusion Matrix Graph
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Greens', xticklabels=['Benign', 'Spoof'], yticklabels=['Benign', 'Spoof'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (XGBoost)')
plt.show()

# Precision-recall curve for XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
plt.plot(recall_xgb, precision_xgb, marker='.', color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (XGBoost)')
plt.grid(True)
plt.show()

# F1 score curve for XGBoost
f1_scores_xgb = 2 * (precision_xgb * recall_xgb) / (precision_xgb + recall_xgb)
plt.figure(figsize=(8, 6))
plt.plot(recall_xgb, f1_scores_xgb, marker='.', color='orange')
plt.xlabel('Recall')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve (XGBoost)')
plt.grid(True)
plt.show()

# AUC curve for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'AUC = {roc_auc_xgb:.2f}', color='orange')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (XGBoost)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

