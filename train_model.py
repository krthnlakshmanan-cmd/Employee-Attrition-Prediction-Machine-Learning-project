
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("dataset/emp_attrition.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove unnecessary columns
X = df.drop(columns=[
    "Attrition",
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
])

# 3. Separate target
y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# 4. Identify categorical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns

# 5. Convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# 6. Create ML model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train the model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
))

# 11. Save the new model separately
joblib.dump(
    model,
    "models/attrition_model_new.pkl"
)

print("New model saved successfully!")
