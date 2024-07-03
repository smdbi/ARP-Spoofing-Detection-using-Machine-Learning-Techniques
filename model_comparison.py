import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
import time

df = pd.read_csv('arp_logs.csv')

# Check for missing values
#print(df.isnull().sum())
#df['Source_IP'].fillna('0.0.0.0', inplace=True)
#df['Destination_IP'].fillna('0.0.0.0', inplace=True)


# Data preprocessing
df.dropna(inplace=True)
label_encoder = LabelEncoder()
df['Source_MAC'] = label_encoder.fit_transform(df['Source_MAC'])
df['Source_IP'] = label_encoder.fit_transform(df['Source_IP'])
df['Destination_MAC'] = label_encoder.fit_transform(df['Destination_MAC'])
df['Destination_IP'] = label_encoder.fit_transform(df['Destination_IP'])
df['Protocol'] = label_encoder.fit_transform(df['Protocol'])


# Define features and target variable
X = df[['Source_MAC', 'Source_IP', 'Destination_MAC', 'Destination_IP', 'Protocol']]
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Measure training and testing times
performance_data = []

# RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

start_time = time.time()
grid_search_rf.fit(X_train_res, y_train_res)
train_time_rf = time.time() - start_time

best_rf = grid_search_rf.best_estimator_
start_time = time.time()
y_pred_rf_test = best_rf.predict(X_test)
test_time_rf = time.time() - start_time

y_pred_rf_train = best_rf.predict(X_train_res)
train_accuracy_rf = accuracy_score(y_train_res, y_pred_rf_train)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf_test)
f1_rf = f1_score(y_test, y_pred_rf_test, average='weighted')

performance_data.append(['RandomForest', train_time_rf, test_time_rf, train_accuracy_rf, test_accuracy_rf, f1_rf])

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)

start_time = time.time()
grid_search_xgb.fit(X_train_res, y_train_res)
train_time_xgb = time.time() - start_time

best_xgb = grid_search_xgb.best_estimator_
start_time = time.time()
y_pred_xgb_test = best_xgb.predict(X_test)
test_time_xgb = time.time() - start_time

y_pred_xgb_train = best_xgb.predict(X_train_res)
train_accuracy_xgb = accuracy_score(y_train_res, y_pred_xgb_train)
test_accuracy_xgb = accuracy_score(y_test, y_pred_xgb_test)
f1_xgb = f1_score(y_test, y_pred_xgb_test, average='weighted')

performance_data.append(['XGBoost', train_time_xgb, test_time_xgb, train_accuracy_xgb, test_accuracy_xgb, f1_xgb])

# Create a DataFrame with the performance data
performance_df = pd.DataFrame(performance_data, columns=['Model', 'Training Time (s)', 'Testing Time (s)', 'Train Accuracy', 'Test Accuracy', 'F1 Score'])

# Plot histogram of Train Accuracy, Test Accuracy, and F1 Score
metrics = ['Train Accuracy', 'Test Accuracy', 'F1 Score']
rf_metrics = [train_accuracy_rf, test_accuracy_rf, f1_rf]
xgb_metrics = [train_accuracy_xgb, test_accuracy_xgb, f1_xgb]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars_rf = ax.bar(x - width/2, rf_metrics, width, label='RandomForest', color='blue')
bars_xgb = ax.bar(x + width/2, xgb_metrics, width, label='XGBoost', color='orange')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

annotate_bars(bars_rf)
annotate_bars(bars_xgb)

plt.ylim(0, 1)
plt.show()

# Plot feature importance comparison
rf_importances = best_rf.feature_importances_
xgb_importances = best_xgb.feature_importances_
feature_names = X.columns

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.35
x = np.arange(len(feature_names))

bars_rf = ax.bar(x - width/2, rf_importances, width, label='RandomForest', color='blue')
bars_xgb = ax.bar(x + width/2, xgb_importances, width, label='XGBoost', color='orange')

ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.legend()

annotate_bars(bars_rf)
annotate_bars(bars_xgb)

plt.show()

# Plot Training Time vs Testing Time
plt.figure(figsize=(10, 6))
plt.scatter(performance_df['Training Time (s)'], performance_df['Testing Time (s)'], s=100, c='blue', alpha=0.5)
plt.title('Training Time vs Testing Time')
plt.xlabel('Training Time (s)')
plt.ylabel('Testing Time (s)')
for i, model in enumerate(performance_df['Model']):
    plt.text(performance_df['Training Time (s)'][i], performance_df['Testing Time (s)'][i], model, fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()
