import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # You can change this model later
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # for saving the model

# Load dot and dash datasets
dot_df = pd.read_csv("dot_data.csv")
dash_df = pd.read_csv("dash_data.csv")

# Add labels: 0 for dot, 1 for dash
dot_df["label"] = 0
dash_df["label"] = 1

# Combine the datasets
data = pd.concat([dot_df, dash_df], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = data[["duration", "velocity"]]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model for real-time use
joblib.dump(model, "tap_classifier_model.pkl")
print("✅ Model saved as tap_classifier_model.pkl")
