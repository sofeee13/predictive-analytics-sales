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


# 1. Load and Clean Dataset

df = pd.read_csv("data/sales_data.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Convert Order Date into datetime format
df["Order Date"] = pd.to_datetime(
    df["Order Date"], errors="coerce"
)

# Convert numerical columns into numeric format
numeric_columns = [
    "Quantity",
    "Unit Price",
    "Revenue",
    "Cost",
    "Profit"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Remove rows with missing required values
required_columns = [
    "Order Date",
    "Product Name",
    "Category",
    "Quantity",
    "Unit Price",
    "Revenue",
    "Region"
]

df = df.dropna(subset=required_columns)

print("===== DATA CLEANING =====")
print("Cleaned Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())


# 2. Sales Analysis

print("\n===== SALES ANALYSIS =====")

print("Total Revenue:", df["Revenue"].sum())
print("Total Profit:", df["Profit"].sum())
print("Total Orders:", df["Order ID"].nunique())
print("Total Quantity Sold:", df["Quantity"].sum())


# 3. Top Selling Products

print("\n===== TOP SELLING PRODUCTS =====")

top_products = df.groupby("Product Name")["Quantity"].sum()

print(top_products.sort_values(ascending=False))


# 4. Revenue by Category

print("\n===== REVENUE BY CATEGORY =====")

category_revenue = df.groupby("Category")["Revenue"].sum()

print(category_revenue)


# 5. Region-wise Sales

print("\n===== REGION WISE SALES =====")

region_sales = df.groupby("Region")["Revenue"].sum()

print(region_sales)


# 6. Feature Engineering

df["Month"] = df["Order Date"].dt.month
df["Day"] = df["Order Date"].dt.day
df["DayOfWeek"] = df["Order Date"].dt.dayofweek


# 7. Select Features and Target

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


# 8. Data Preprocessing

categorical_features = [
    "Product Name",
    "Category",
    "Region"
]

numerical_features = [
    "Quantity",
    "Unit Price",
    "Month",
    "Day",
    "DayOfWeek"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# 9. Create Regression Model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    )
])


# 10. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 11. Train Model

model.fit(X_train, y_train)


# 12. Make Predictions

y_pred = model.predict(X_test)


# 13. Model Evaluation

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# 14. Save Model and Results

os.makedirs("outputs", exist_ok=True)

joblib.dump(
    model,
    "outputs/sales_prediction_model.pkl"
)

results = pd.DataFrame({
    "Actual Revenue": y_test.values,
    "Predicted Revenue": y_pred
})

results.to_csv(
    "outputs/prediction_results.csv",
    index=False
)


# 15. Visualize Actual vs Predicted Values

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Sales Revenue")

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.close()


# 16. Completion

print("\n===== PROJECT COMPLETED =====")
print("Regression model trained successfully.")
print("Model saved in outputs folder.")
print("Prediction results saved.")
print("Actual vs Predicted graph saved.")