
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("dataset/emp_attrition.csv")
# 2. Remove unnecessary columns
df = df.drop(
    ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"],
    axis=1
)
print("\nModel saved successfully!")
# 3. Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"No": 0, "Yes": 1})

# 4. Identify column types
categorical_columns = X.select_dtypes(
    include=["str", "object"]
).columns

numerical_columns = X.select_dtypes(
    include=["number"]
).columns

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# 6. Create ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)
print("Classifier settings:", model.named_steps["classifier"])
# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
))

# 13. Save trained model
joblib.dump(model, "models/attrition_model.pkl")

print("\nModel saved successfully!")
