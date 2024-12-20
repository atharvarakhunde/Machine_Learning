import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load training dataset
data_train = pd.read_csv('Credit_Card_Fraud_detection/fraudTrain.csv')
data_train = data_train.sample(frac=0.1, random_state=42)  # Sample for faster testing

# Load testing dataset
data_test = pd.read_csv('Credit_Card_Fraud_detection/fraudTest.csv')

# Drop irrelevant columns in both datasets
irrelevant_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'dob', 'trans_num', 'street', 'job']
data_train = data_train.drop(irrelevant_cols, axis=1, errors='ignore')
data_test = data_test.drop(irrelevant_cols, axis=1, errors='ignore')

# Separate features and target variable
X_train = data_train.drop(['is_fraud'], axis=1)
y_train = data_train['is_fraud']
X_test = data_test.drop(['is_fraud'], axis=1)
y_test = data_test['is_fraud']

# Identify high-cardinality columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
high_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() > 100]
low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() <= 100]

# Label encode high-cardinality columns
le = LabelEncoder()
for col in high_cardinality_cols:
    X_train[col] = le.fit_transform(X_train[col])
    if col in X_test.columns:  # Prevent errors during unseen labels in test data
        X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# One-hot encode low-cardinality columns
X_train = pd.get_dummies(X_train, columns=low_cardinality_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_cardinality_cols, drop_first=True)

# Align train and test columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols].astype(np.float32))
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols].astype(np.float32))

# Apply SMOTE to training data
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,  # Increase number of trees
    max_depth=15,      # Increase tree depth for better learning
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set with adjusted threshold
rf_probs = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.4  # Lower threshold for higher Recall
rf_preds = (rf_probs >= threshold).astype(int)

# Evaluate the Random Forest model
print("Random Forest Classification Report on Test Data:")
print(classification_report(y_test, rf_preds))
print("Random Forest AUC-ROC Score on Test Data:", roc_auc_score(y_test, rf_probs))

# Feature Importance Visualization
importances = rf_model.feature_importances_
features = X_train.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(10), importances[sorted_indices[:10]], align="center")
plt.xticks(range(10), features[sorted_indices[:10]], rotation=90)
plt.show()

# Train XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set with XGBoost
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs >= threshold).astype(int)

# Evaluate the XGBoost model
print("XGBoost Classification Report on Test Data:")
print(classification_report(y_test, xgb_preds))
print("XGBoost AUC-ROC Score on Test Data:", roc_auc_score(y_test, xgb_probs))
