📊 Predictive Analytics Using Historical Sales Data

📌 Project Overview

This project focuses on analysing historical sales data and predicting sales revenue using Machine Learning techniques.

The project uses Python for data analysis, data preprocessing, visualization, and predictive modelling. An interactive Streamlit dashboard is developed to present business insights and sales predictions.

🎯 Objectives

- Analyse historical sales data.
- Identify sales trends and patterns.
- Understand product and regional performance.
- Build a regression-based prediction model.
- Evaluate model performance.
- Present insights through an interactive dashboard.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Plotly

📂 Project Structure

thiranex/
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── sales_data.csv
│   └── generate_data.py
└── outputs/
    ├── sales_prediction_model.pkl
    ├── actual_vs_predicted.png
    └── prediction_results.csv

📈 Dataset Description

The dataset contains 500 sales records and 10 columns:

- Order ID
- Order Date
- Product Name
- Category
- Quantity
- Unit Price
- Revenue
- Cost
- Profit
- Region

🤖 Machine Learning Model

Random Forest Regressor is used to predict sales revenue.

Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

📊 Dashboard Features

- Total Revenue
- Total Profit
- Total Orders
- Quantity Sold
- Category-wise Revenue
- Region-wise Revenue
- Monthly Revenue Trends
- Top Selling Products
- Interactive Sales Revenue Prediction

⚙️ Installation and Execution

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project folder

cd thiranex

3. Install dependencies

pip install -r requirements.txt

4. Run the analysis

python analysis.py

5. Launch the dashboard

streamlit run app.py

📌 Expected Outcome

A machine learning-based sales analytics solution that helps identify historical sales patterns, evaluate predictive performance, and generate revenue predictions through an interactive dashboard.

👩‍💻 Project Category

Predictive Analytics | Machine Learning | Data Analysis

📄 License

This project is created for educational and academic purposes.