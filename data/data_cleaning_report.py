import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("data/sales_data.csv")

print("===== ORIGINAL DATA =====")
print("Rows and Columns:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())


# 2. Remove Duplicate Rows
df = df.drop_duplicates()


# 3. Clean Missing Values
# Numeric columns: fill missing values with median
numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# Text columns: fill missing values with "Unknown"
text_columns = df.select_dtypes(
    include="object"
).columns

for column in text_columns:
    df[column] = df[column].fillna("Unknown")


# 4. Clean and Convert Date
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )


# 5. Clean Text Values
for column in text_columns:
    df[column] = df[column].astype(str).str.strip()


# 6. Save Cleaned Dataset
df.to_csv(
    "outputs/cleaned_sales_data.csv",
    index=False
)


# 7. Generate Automated Summary Report
report = df.describe(include="all").transpose()

report.to_csv(
    "outputs/sales_summary_report.csv"
)


# 8. Generate Revenue by Category Report
if "Category" in df.columns and "Revenue" in df.columns:
    category_report = df.groupby(
        "Category"
    )["Revenue"].sum().reset_index()

    category_report.to_csv(
        "outputs/revenue_by_category.csv",
        index=False
    )

    # Create visualization
    plt.figure(figsize=(8, 5))

    plt.bar(
        category_report["Category"],
        category_report["Revenue"]
    )

    plt.title("Revenue by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "outputs/revenue_by_category.png"
    )

    plt.close()


# 9. Final Data Quality Report
print("\n===== CLEANED DATA =====")
print("Rows and Columns:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

print("\n===== REPORTS GENERATED =====")
print("Cleaned dataset saved.")
print("Sales summary report saved.")
print("Revenue by category report saved.")
print("Revenue visualization saved.")

print("\nData Cleaning & Reporting Automation Completed!")