import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Sales Predictive Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Predictive Analytics Dashboard")
st.write("Sales analysis and machine learning prediction")

# Load dataset
df = pd.read_csv("data/sales_data.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# KPI metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{df['Revenue'].sum():,.2f}")
col2.metric("Total Profit", f"₹{df['Profit'].sum():,.2f}")
col3.metric("Total Orders", df["Order ID"].nunique())
col4.metric("Quantity Sold", int(df["Quantity"].sum()))

st.divider()

# Revenue by Category
st.subheader("Revenue by Category")
category_revenue = df.groupby("Category")["Revenue"].sum()
st.bar_chart(category_revenue)

# Region-wise Revenue
st.subheader("Region-wise Revenue")
region_revenue = df.groupby("Region")["Revenue"].sum()
st.bar_chart(region_revenue)

# Monthly Revenue
st.subheader("Monthly Revenue Trend")
monthly_revenue = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Revenue"].sum()
monthly_revenue.index = monthly_revenue.index.astype(str)
st.line_chart(monthly_revenue)

# Top products
st.subheader("Top Selling Products")
top_products = df.groupby("Product Name")["Quantity"].sum()
st.bar_chart(top_products.sort_values(ascending=False).head(10))

st.divider()

# Machine Learning Prediction
st.subheader("🤖 Sales Revenue Prediction")

model_path = "outputs/sales_prediction_model.pkl"

if os.path.exists(model_path):

    model = joblib.load(model_path)

    product = st.selectbox(
        "Select Product",
        sorted(df["Product Name"].unique())
    )

    category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1
    )

    price = st.number_input(
        "Unit Price",
        min_value=0.0,
        value=float(df["Unit Price"].median())
    )

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    order_date = st.date_input("Order Date")

    if st.button("Predict Revenue"):

        prediction_data = pd.DataFrame([{
            "Product Name": product,
            "Category": category,
            "Quantity": quantity,
            "Unit Price": price,
            "Region": region,
            "Month": order_date.month,
            "Day": order_date.day,
            "DayOfWeek": order_date.weekday()
        }])

        prediction = model.predict(prediction_data)[0]

        st.success(
            f"Predicted Revenue: ₹{prediction:,.2f}"
        )

else:
    st.warning(
        "Model not found. Run python analysis.py first."
    )

st.divider()

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.caption("Developed using Python, Pandas, Scikit-learn and Streamlit")