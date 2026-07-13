"""
Streamlit dashboard for Task 7 of the sales forecasting project.
Basically reuses the same logic from analysis.ipynb (XGBoost forecast, anomaly
detection, clustering) but wraps it in a simple 4-page app so someone can
click around instead of reading a notebook.

Run locally with: streamlit run app.py
(train.csv needs to be in the same folder)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Data loading and shared feature engineering (cached so the app stays fast)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        return "Fall"

    df["Season"] = df["Month"].apply(month_to_season)
    return df


def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    return "Fall"


def make_supervised_features(series):
    d = series.reset_index()
    d.columns = ["Month", "Sales"]
    d["Lag1"] = d["Sales"].shift(1)
    d["Lag2"] = d["Sales"].shift(2)
    d["Lag3"] = d["Sales"].shift(3)
    d["RollingMean3"] = d["Sales"].shift(1).rolling(3).mean()
    d["MonthNum"] = d["Month"].dt.month
    d["Quarter"] = d["Month"].dt.quarter
    d["Season"] = d["MonthNum"].apply(month_to_season)
    d = pd.get_dummies(d, columns=["Season"], drop_first=True)
    return d.dropna().reset_index(drop=True)


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


@st.cache_data
def forecast_with_xgboost(series, horizon=3):
    """Recursive XGBoost forecast. Returns (forecast_series, mae, rmse) computed against
    a 3-month holdout, matching the evaluation approach used in analysis.ipynb."""
    series = series.asfreq("MS", fill_value=0)
    if len(series) < 10:
        return None, None, None

    train_series = series.iloc[:-horizon]
    test_series = series.iloc[-horizon:]

    feats = make_supervised_features(train_series)
    if len(feats) < 4:
        return None, None, None
    f_cols = [c for c in feats.columns if c not in ["Month", "Sales"]]

    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(feats[f_cols], feats["Sales"])

    hist = list(train_series.values)
    preds = []
    for step in range(horizon):
        lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]
        roll3 = np.mean(hist[-3:])
        fm = test_series.index[step]
        season = month_to_season(fm.month)
        row = {"Lag1": lag1, "Lag2": lag2, "Lag3": lag3, "RollingMean3": roll3,
               "MonthNum": fm.month, "Quarter": (fm.month - 1) // 3 + 1}
        for col in f_cols:
            if col.startswith("Season_"):
                row[col] = 1 if col == f"Season_{season}" else 0
        row_df = pd.DataFrame([row])[f_cols]
        pred = max(model.predict(row_df)[0], 0)
        preds.append(pred)
        hist.append(pred)

    pred_series = pd.Series(preds, index=test_series.index)
    return pred_series, mae(test_series.values, pred_series.values), rmse(test_series.values, pred_series.values)


@st.cache_data
def run_clustering(df):
    subcat_monthly = (
        df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    )
    rows = []
    for subcat, g in subcat_monthly.groupby("Sub-Category"):
        g = g.sort_values("Order Date")
        total_sales = g["Sales"].sum()
        volatility = g["Sales"].std()
        yearly = g.set_index("Order Date")["Sales"].resample("YS").sum()
        yoy = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100 if len(yearly) >= 2 and yearly.iloc[0] > 0 else 0
        sub_orders = df[df["Sub-Category"] == subcat]
        avg_order_value = sub_orders["Sales"].sum() / len(sub_orders)
        rows.append({"Sub-Category": subcat, "Total Sales Volume": total_sales,
                      "YoY Growth Rate (%)": yoy, "Sales Volatility": volatility,
                      "Average Order Value": avg_order_value})

    feats = pd.DataFrame(rows).set_index("Sub-Category")
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    feats["Cluster"] = km.fit_predict(X)

    profile = feats.groupby("Cluster")[["Total Sales Volume", "YoY Growth Rate (%)", "Sales Volatility"]].mean()
    profile["Volatility Ratio"] = profile["Sales Volatility"] / profile["Total Sales Volume"]
    growth_sorted = profile.sort_values("YoY Growth Rate (%)", ascending=False)
    top, bottom = growth_sorted.index[0], growth_sorted.index[-1]
    remaining = [c for c in profile.index if c not in (top, bottom)]
    labels = {top: "Growing Demand", bottom: "Declining Demand"}
    if len(remaining) == 1:
        labels[remaining[0]] = "High Volume, Stable Demand"
    elif len(remaining) > 1:
        rp = profile.loc[remaining]
        combined = rp["Total Sales Volume"].rank(ascending=False) + rp["Volatility Ratio"].rank(ascending=True)
        for c in remaining:
            labels[c] = "High Volume, Stable Demand" if combined[c] == combined.min() else "Low Volume, High Volatility"

    feats["Cluster Label"] = feats["Cluster"].map(labels)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    feats["PCA1"], feats["PCA2"] = coords[:, 0], coords[:, 1]
    return feats


@st.cache_data
def detect_anomalies(df):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum().to_frame("Total Sales")
    weekly = weekly.asfreq("W", fill_value=0)

    iso_feats = weekly[["Total Sales"]].copy()
    iso_feats["RollingMean"] = iso_feats["Total Sales"].rolling(4, min_periods=1).mean()
    iso_feats["RollingStd"] = iso_feats["Total Sales"].rolling(4, min_periods=1).std().fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["iso_anomaly"] = iso.fit_predict(iso_feats) == -1

    roll_mean = weekly["Total Sales"].rolling(4, min_periods=1).mean()
    roll_std = weekly["Total Sales"].rolling(4, min_periods=1).std().fillna(0)
    z = ((weekly["Total Sales"] - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)
    weekly["z_score"] = z
    weekly["zscore_anomaly"] = z.abs() > 2

    return weekly


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

df = load_data()

st.sidebar.title("Sales Forecasting Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ============================== PAGE 1 ======================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.multiselect("Filter by Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    with col2:
        selected_category = st.multiselect("Filter by Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(selected_region) & df["Category"].isin(selected_category)]

    st.subheader("Total Sales by Year")
    yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
    fig = px.bar(yearly, x="Year", y="Sales", text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    fig2 = px.line(monthly, x="Order Date", y="Sales", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category")
    region_cat = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig3 = px.bar(region_cat, x="Region", y="Sales", color="Category", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

# ============================== PAGE 2 ======================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    dimension = st.selectbox("Select dimension", ["Category", "Region"])
    if dimension == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())
    selected_value = st.selectbox(f"Select {dimension}", options)

    horizon_months = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    segment_df = df[df[dimension] == selected_value]
    segment_ts = segment_df.set_index("Order Date").resample("MS")["Sales"].sum()

    with st.spinner("Training forecasting model..."):
        forecast, model_mae, model_rmse = forecast_with_xgboost(segment_ts, horizon=3)

    if forecast is None:
        st.warning("Not enough history for this segment to produce a reliable forecast.")
    else:
        forecast_display = forecast.iloc[:horizon_months]

        fig = go.Figure()
        history_recent = segment_ts.iloc[-12:]
        fig.add_trace(go.Scatter(x=history_recent.index, y=history_recent.values, mode="lines", name="History"))
        fig.add_trace(go.Scatter(x=forecast_display.index, y=forecast_display.values, mode="lines+markers", name="Forecast (XGBoost)"))
        fig.update_layout(title=f"{horizon_months}-Month Forecast: {selected_value} ({dimension})", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Accuracy (evaluated on a 3-month holdout)")
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"${model_mae:,.0f}")
        m2.metric("RMSE", f"${model_rmse:,.0f}")

        st.caption("Using XGBoost here since it came out best in my model comparison in analysis.ipynb (Task 3) - SARIMA and Prophet are compared there too if you want to see the numbers.")

# ============================== PAGE 3 ======================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly = detect_anomalies(df)
    iso_anoms = weekly[weekly["iso_anomaly"]]
    z_anoms = weekly[weekly["zscore_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["Total Sales"], mode="lines", name="Weekly Sales", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=iso_anoms.index, y=iso_anoms["Total Sales"], mode="markers", name="Isolation Forest Anomaly",
                              marker=dict(color="red", size=10)))
    fig.add_trace(go.Scatter(x=z_anoms.index, y=z_anoms["Total Sales"], mode="markers", name="Z-Score Anomaly",
                              marker=dict(color="orange", size=8, symbol="triangle-up")))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Dates")
    combined = weekly[weekly["iso_anomaly"] | weekly["zscore_anomaly"]].copy()
    combined = combined.reset_index().rename(columns={"index": "Week", "Order Date": "Week"})
    combined["Flagged by Isolation Forest"] = combined["iso_anomaly"]
    combined["Flagged by Z-Score"] = combined["zscore_anomaly"]
    st.dataframe(combined[["Week", "Total Sales", "Flagged by Isolation Forest", "Flagged by Z-Score"]].sort_values("Week"),
                 use_container_width=True)

# ============================== PAGE 4 ======================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    clustered = run_clustering(df)

    fig = px.scatter(clustered.reset_index(), x="PCA1", y="PCA2", color="Cluster Label",
                      text="Sub-Category", title="Product Sub-Category Demand Clusters (PCA-reduced)")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Categories by Demand Cluster")
    display_cols = ["Total Sales Volume", "YoY Growth Rate (%)", "Sales Volatility", "Average Order Value", "Cluster Label"]
    st.dataframe(clustered[display_cols].sort_values("Cluster Label"), use_container_width=True)
