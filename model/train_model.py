# -----------------------------------------------------DATA PREPROCESSING SECTION---------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
data = pd.read_csv("./dataset/student_dataset.csv")

print("Dataset shape:", data.shape)

print("\nMissing values:")
print(data.isnull().sum())

data["Target"] = data["Target"].apply(
    lambda x: 1 if x == "Dropout" else 0
)

print("\nTarget after conversion:")
print(data["Target"].value_counts())

columns_to_drop = [
    "Application mode",
    "Application order",
    "Course",
    "Nationality"
]

data = data.drop(columns=columns_to_drop)

print("\nColumns after dropping:")
print(data.columns)

print("\nData types:")
print(data.dtypes)

X = data.drop("Target", axis=1)
y = data["Target"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

data.to_csv("./dataset/processed_student_data.csv", index=False)

print("\nProcessed dataset saved as 'processed_student_data.csv'")

#----------------------------------------------------TRAINING MODEL SECTION-------------------------------------------------------

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nBefore SMOTE:")
print("Class distribution:", y_train.value_counts())


smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Class distribution:", y_train.value_counts())

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

joblib.dump(best_model, "model.pkl")

print("\nModel saved as model.pkl")

#----------------------------------------------------MODEL COMPARISON SECTION-------------------------------------------------------
print("\n================ MODEL COMPARISON ================")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_acc)
print(classification_report(y_test, dt_pred))

# Random Forest (already computed)
rf_acc = accuracy

print("\nRandom Forest Accuracy:", rf_acc)

# Save results for UI
results_df = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
    "Accuracy": [rf_acc, lr_acc, dt_acc]
})

results_df.to_csv("model_results.csv", index=False)

print("\nModel comparison saved as 'model_results.csv'")