import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

train_df = pd.read_csv("../data/train_features.csv")
val_df = pd.read_csv("../data/val_features.csv")

X_train = train_df.drop(columns=["label"]).values
y_train_labels = train_df["label"].values

X_val = val_df.drop(columns=["label"]).values
y_val_labels = val_df["label"].values

# Encode string labels (angry, happy, etc) into integers 0..K-1
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_labels)
y_val = label_encoder.transform(y_val_labels)

from sklearn.model_selection import StratifiedKFold

# Common CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Best params from your grid search
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

# Train the model
print("Training Random Forest with best hyperparameters...")
model.fit(X_train, y_train)

# Evaluate on train set
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train accuracy: {train_acc:.4f}")

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy: {val_acc:.4f}")

print("\nClassification report (validation):")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

print("Confusion matrix (validation):")
print(confusion_matrix(y_val, y_val_pred))

model_path = os.path.join("..", "models", "emotion_model.joblib")
encoder_path = os.path.join("..", "models", "label_encoder.joblib")

joblib.dump(model, model_path)
joblib.dump(label_encoder, encoder_path)

print("Saved:", model_path)
print("Saved:", encoder_path)