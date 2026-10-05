import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"dataset\StressLevelDataset.csv")

# Display first few rows
print(df.head())

print(df.shape)
print(df.isnull().sum())

# View basic statistics
print(df.describe())

# Separate features from target
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial Random Forest to rank feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Get Top 5 important features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top5 = X.columns[indices][:5]
print("Selected Important Features:", list(top5))

X_top5 = df[top5]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_top5, y, test_size=0.2, random_state=42, stratify=y
)

# Scale again
X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

# Train final model
final_model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=15)
final_model.fit(X_train2_scaled, y_train2)

y_pred = final_model.predict(X_test2_scaled)

print("Accuracy:", accuracy_score(y_test2, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test2, y_pred))
print("\nClassification Report:\n", classification_report(y_test2, y_pred))

importances_final = final_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': top5, 'importance': importances_final})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("\nFeature Importance in Final Model:")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance in Final Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Generate confusion matrix
cm = confusion_matrix(y_test2, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Low Stress', 'Medium Stress', 'High Stress'],
            yticklabels=['Low Stress', 'Medium Stress', 'High Stress'])
plt.xlabel('Predicted Stress Level')
plt.ylabel('Actual Stress Level')
plt.title(f'Confusion Matrix - Accuracy: {accuracy_score(y_test2, y_pred):.4f}')
plt.tight_layout()
plt.show()

# Save model and scaler
os.makedirs("ml models", exist_ok=True)
joblib.dump(final_model, r"ml models\stress_model.pkl")
joblib.dump(scaler, r"ml models\scaler.pkl")
print("\n✓ Model and scaler saved successfully!")