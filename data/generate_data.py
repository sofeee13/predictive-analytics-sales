import pandas as pd
import random
from datetime import datetime, timedelta

# Sample sales data
products = {
    "Laptop": ("Electronics", 50000),
    "Mobile": ("Electronics", 20000),
    "Headphones": ("Accessories", 2000),
    "Keyboard": ("Accessories", 1500),
    "Chair": ("Furniture", 5000),
    "Table": ("Furniture", 8000)
}

regions = ["North", "South", "East", "West"]

data = []

for i in range(500):

    product = random.choice(list(products.keys()))
    category, price = products[product]

    quantity = random.randint(1, 5)
    cost_per_unit = price * 0.70

    revenue = quantity * price
    cost = quantity * cost_per_unit
    profit = revenue - cost

    order_date = datetime(2026, 1, 1) + timedelta(
        days=random.randint(0, 299)
    )

    data.append({
        "Order ID": i + 1,
        "Order Date": order_date,
        "Product Name": product,
        "Category": category,
        "Quantity": quantity,
        "Unit Price": price,
        "Revenue": revenue,
        "Cost": cost,
        "Profit": profit,
        "Region": random.choice(regions)
    })

# Convert into DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("data/sales_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Total records:", len(df))