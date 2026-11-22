import pandas as pd

# Load the dataset
df = pd.read_csv("data/sales_data.csv")

# Convert Order Date into date format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# 1. Total Revenue
total_revenue = df["Revenue"].sum()

# 2. Total Profit
total_profit = df["Profit"].sum()

# 3. Total Orders
total_orders = df["Order ID"].nunique()

# 4. Total Quantity Sold
total_quantity = df["Quantity"].sum()

# Display results
print("===== SALES ANALYSIS =====")

print("Total Revenue:", total_revenue)
print("Total Profit:", total_profit)
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)

# 5. Top Selling Products
print("\n===== TOP SELLING PRODUCTS =====")

top_products = df.groupby("Product Name")["Quantity"].sum()

print(top_products.sort_values(ascending=False))

# 6. Revenue by Category
print("\n===== REVENUE BY CATEGORY =====")

category_revenue = df.groupby("Category")["Revenue"].sum()

print(category_revenue)

# 7. Region-wise Sales
print("\n===== REGION WISE SALES =====")

region_sales = df.groupby("Region")["Revenue"].sum()

print(region_sales)