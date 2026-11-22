import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard title
st.title("📊 Sales & Revenue Analysis Dashboard")
st.write("Analyze sales performance and business insights.")

# Data upload
st.sidebar.header("Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv("data/sales_data.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Sidebar filters
st.sidebar.header("Dashboard Filters")

categories = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

regions = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

# Filter data
filtered_df = df[
    (df["Category"].isin(categories)) &
    (df["Region"].isin(regions))
]

# KPI calculations
total_revenue = filtered_df["Revenue"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
total_quantity = filtered_df["Quantity"].sum()

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Total Profit", f"₹{total_profit:,.0f}")
col3.metric("Total Orders", total_orders)
col4.metric("Units Sold", total_quantity)

st.divider()

# Monthly sales trend
st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .set_index("Order Date")
    .resample("ME")["Revenue"]
    .sum()
    .reset_index()
)

fig1 = px.line(
    monthly_sales,
    x="Order Date",
    y="Revenue",
    markers=True,
    title="Monthly Revenue"
)

st.plotly_chart(fig1, use_container_width=True)

# Product analysis
col5, col6 = st.columns(2)

with col5:
    st.subheader("Revenue by Product")

    product_sales = (
        filtered_df.groupby("Product Name")["Revenue"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        product_sales,
        x="Product Name",
        y="Revenue",
        title="Product-wise Revenue"
    )

    st.plotly_chart(fig2, use_container_width=True)

with col6:
    st.subheader("Sales by Category")

    category_sales = (
        filtered_df.groupby("Category")["Revenue"]
        .sum()
        .reset_index()
    )

    fig3 = px.pie(
        category_sales,
        names="Category",
        values="Revenue",
        title="Category-wise Revenue"
    )

    st.plotly_chart(fig3, use_container_width=True)

# Top products
st.subheader("Top 5 Selling Products")

top_products = (
    filtered_df.groupby("Product Name")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

st.dataframe(top_products, use_container_width=True)

# Dataset preview
with st.expander("View Sales Dataset"):
    st.dataframe(filtered_df, use_container_width=True)