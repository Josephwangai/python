import streamlit_authenticator as stauth
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from report_generator import generate_pdf_report
import tempfile
import base64
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🛒 Sales Basket", layout="wide")

# ----------- Login Setup -----------
names = ['Akbar Hossain']
usernames = ['akbar']
passwords = ['sales123']

hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    "usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "salesbasket_login",
    "abc123",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login", location="sidebar")

if authentication_status is False:
    st.error("❌ Incorrect username or password")
elif authentication_status is None:
    st.warning("🔐 Please log in to continue")

if authentication_status:
    def show_logo():
        try:
            with open("assets/logo.png", "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px">
                        <img src="data:image/png;base64,{encoded}" width="60"/>
                        <div>
                            <h1 style="margin-bottom:5px;margin-top:0">🛒 Sales Basket</h1>
                            <p style="margin-top:0;color:gray;">
                                Discover top-selling products, market basket insights, and sales forecasts.
                            </p>
                        </div>
                    </div>
                    <hr style="margin-top:5px; margin-bottom:25px;"/>
                """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.title("🛒 Sales Basket")

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, encoding='ISO-8859-1')
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        return df

    def detect_outliers(df):
        z_scores = (df['Revenue'] - df['Revenue'].mean()) / df['Revenue'].std()
        return df[(z_scores > -3) & (z_scores < 3)]

    def filter_by_date(df, date_range):
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        return df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]

    def top_selling_items(df):
        qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        rev = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
        return qty, rev

    def generate_mba_rules(df, filter_item=None):
        top_items = df['StockCode'].value_counts().head(15).index.tolist()
        df = df[df['StockCode'].isin(top_items)]
        basket = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0).astype(bool)
        itemsets = apriori(basket, min_support=0.001, use_colnames=True)
        rules = association_rules(itemsets, metric="confidence", min_threshold=0.01)
        rules = rules.sort_values(by='confidence', ascending=False)
        if filter_item:
            rules = rules[rules['antecedents'].astype(str).str.contains(filter_item) |
                          rules['consequents'].astype(str).str.contains(filter_item)]
        return rules

    def map_stockcodes_to_descriptions(df, rules):
        mapping = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
        def decode_frozenset(fset):
            return [mapping.get(code, code) for code in fset]
        rules['antecedents_desc'] = rules['antecedents'].apply(lambda x: decode_frozenset(list(x)))
        rules['consequents_desc'] = rules['consequents'].apply(lambda x: decode_frozenset(list(x)))
        return rules

    def monthly_revenue(df):
        return df.set_index('InvoiceDate').resample('M')['Revenue'].sum()

    def forecast_sales_arima(series, periods):
        train = series[:-periods]
        test = series[-periods:]
        model = ARIMA(train, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        forecast.index = test.index
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        return forecast, mae, rmse

    def forecast_sales_prophet(series, periods):
        prophet_df = series.reset_index()
        prophet_df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(prophet_df[:-periods])
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        forecast_series = forecast.set_index('ds')['yhat'][-periods:]
        actual = series[-periods:]
        mae = mean_absolute_error(actual, forecast_series)
        rmse = np.sqrt(mean_squared_error(actual, forecast_series))
        return forecast_series, mae, rmse

    def plot_bar(data, title, xlabel):
        fig, ax = plt.subplots()
        data.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        st.pyplot(fig)

    def plot_forecast(actual, forecast):
        fig, ax = plt.subplots()
        actual.plot(label='Actual', ax=ax, marker='o')
        forecast.plot(label='Forecast', ax=ax, marker='x', linestyle='--', color='orange')
        ax.set_title("📈 Sales Forecast")
        ax.set_ylabel("Revenue (£)")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

    st.sidebar.title("📂 Upload Your Sales CSV")
    uploaded_file = st.sidebar.file_uploader("Choose sales_data.csv", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        df = detect_outliers(df)
        show_logo()

        min_date, max_date = df['InvoiceDate'].min(), df['InvoiceDate'].max()
        date_range = st.sidebar.date_input("📅 Filter by Date", [min_date, max_date], min_value=min_date, max_value=max_date)
        df = filter_by_date(df, date_range)

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Revenue", f"£{df['Revenue'].sum():,.0f}")
        col2.metric("🧾 Unique Orders", df['InvoiceNo'].nunique())
        col3.metric("📦 Unique Products", df['Description'].nunique())

        # ---------- Top Sellers with Price Range ----------
        st.header("💰 Top-Selling Products by Price Range")

        min_price = float(df['UnitPrice'].min())
        max_price = float(df['UnitPrice'].max())

        price_range = st.slider("Select Price Range (£)",
                                min_value=round(min_price, 2),
                                max_value=round(max_price, 2),
                                value=(10.0, 30.0), step=1.0)

        filtered_df = df[(df['UnitPrice'] >= price_range[0]) & (df['UnitPrice'] <= price_range[1])]

        # Compute top-selling by quantity and revenue in selected price range
        qty = filtered_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        rev = filtered_df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

        st.markdown(f"### 📦 Top-Selling Items between £{price_range[0]} - £{price_range[1]}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### 🔢 By Quantity")
            st.dataframe(qty)
            fig, ax = plt.subplots()
            qty.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f"Top 10 by Quantity (£{price_range[0]} - £{price_range[1]})")
            ax.set_xlabel("Quantity")
            ax.invert_yaxis()
            st.pyplot(fig)

        with col2:
            st.write("#### 💷 By Revenue")
            st.dataframe(rev)
            fig, ax = plt.subplots()
            rev.plot(kind='barh', ax=ax, color='orange')
            ax.set_title(f"Top 10 by Revenue (£{price_range[0]} - £{price_range[1]})")
            ax.set_xlabel("Revenue (£)")
            ax.invert_yaxis()
            st.pyplot(fig)



        st.header("🔮 Sales Forecast")
        sales_series = monthly_revenue(df)

        if sales_series.empty or sales_series.sum() == 0:
            st.warning("No revenue data available for forecasting in the selected date range.")
        else:
            forecast_periods = st.slider("📆 Months to Forecast", 3, 12, 6)
            method = st.selectbox("Select Forecasting Model", ["ARIMA", "Prophet"])

            try:
                if method == "ARIMA":
                    forecast, mae, rmse = forecast_sales_arima(sales_series, forecast_periods)
                else:
                    forecast, mae, rmse = forecast_sales_prophet(sales_series, forecast_periods)

                plot_forecast(sales_series[-forecast_periods:], forecast)
                st.success(f"✅ Forecast MAE: {mae:.2f} | RMSE: {rmse:.2f}")

                forecast_df = forecast.reset_index().rename(columns={'index': 'Date', 0: 'Forecast'})
                st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")

        st.header("🛒 Market Basket Analysis")
        filter_input = st.text_input("Filter Rules by StockCode (optional)")
        rules = generate_mba_rules(df, filter_input)
        if rules.empty:
            st.warning("No rules found for the selected product/filter.")
        else:
            rules = map_stockcodes_to_descriptions(df, rules)
            top_rule = rules.sort_values(by='lift', ascending=False).iloc[0]
            st.markdown(f"""
                <div style='background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:10px'>
                    <b>🔎 Top Rule:</b><br>
                    If <code>{', '.join(top_rule['antecedents_desc'])}</code> then buy <code>{', '.join(top_rule['consequents_desc'])}</code><br>
                    💡 <b>Lift:</b> {top_rule['lift']:.2f} | <b>Confidence:</b> {top_rule['confidence']:.2f}
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(rules[['antecedents_desc', 'consequents_desc', 'support', 'confidence', 'lift']].head(10))
            csv_rules = rules.to_csv(index=False)
            st.download_button("📥 Download MBA Rules CSV", csv_rules, file_name="mba_rules.csv")