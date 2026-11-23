import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("data/sales_data.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

print("===== SALES ANALYSIS =====")

print("Total Revenue:", df["Revenue"].sum())
print("Total Profit:", df["Profit"].sum())
print("Total Orders:", df["Order ID"].nunique())
print("Total Quantity Sold:", df["Quantity"].sum())

# 2. Top Selling Products
print("\n===== TOP SELLING PRODUCTS =====")
print(df.groupby("Product Name")["Quantity"].sum().sort_values(ascending=False))

# 3. Revenue by Category
print("\n===== REVENUE BY CATEGORY =====")
print(df.groupby("Category")["Revenue"].sum())

# 4. Region-wise Sales
print("\n===== REGION WISE SALES =====")
print(df.groupby("Region")["Revenue"].sum())

# 5. Feature Engineering
df["Month"] = df["Order Date"].dt.month
df["Day"] = df["Order Date"].dt.day
df["DayOfWeek"] = df["Order Date"].dt.dayofweek

# 6. Select Features and Target
# Cost and Profit are excluded to prevent target leakage.
features = [
    "Product Name",
    "Category",
    "Quantity",
    "Unit Price",
    "Region",
    "Month",
    "Day",
    "DayOfWeek"
]

X = df[features]
y = df["Revenue"]

# 7. Preprocessing
categorical_features = ["Product Name", "Category", "Region"]
numerical_features = [
    "Quantity", "Unit Price", "Month", "Day", "DayOfWeek"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# 8. Create Regression Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# 9. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Train Model
model.fit(X_train, y_train)

# 11. Make Predictions
y_pred = model.predict(X_test)

# 12. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 13. Save Trained Model
joblib.dump(model, "outputs/sales_prediction_model.pkl")

# 14. Actual vs Predicted Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Sales Revenue")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# 15. Save Prediction Results
results = pd.DataFrame({
    "Actual Revenue": y_test.values,
    "Predicted Revenue": y_pred
})

results.to_csv("outputs/prediction_results.csv", index=False)

print("\nProject analysis completed successfully!")
print("Model saved in outputs folder.")
print("Prediction graph saved in outputs folder.")
print("Prediction results saved in outputs folder.")