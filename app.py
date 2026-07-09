import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# pyrefly: ignore [missing-import]
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit Page Config
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling for Dark Theme & Glassmorphism
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* General theme styling */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif;
            background-color: #0b0d16;
            color: #f7fafc;
        }
        
        /* Main background adjust */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f1220;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #cbd5e0;
            font-weight: 600;
        }
        
        /* Cards & Glassmorphism */
        .premium-card {
            background: rgba(22, 27, 49, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .premium-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Metric container cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(22, 27, 49, 0.9) 0%, rgba(15, 18, 32, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            text-align: left;
        }
        
        .metric-card:hover {
            border-color: rgba(66, 153, 225, 0.4);
            transform: translateY(-1px);
        }
        
        .metric-blue { border-left: 4px solid #3182ce; }
        .metric-purple { border-left: 4px solid #805ad5; }
        .metric-orange { border-left: 4px solid #dd6b20; }
        .metric-green { border-left: 4px solid #38a169; }
        
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #a0aec0;
            margin-bottom: 6px;
            font-weight: 500;
        }
        
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
        }
        
        .metric-desc {
            font-size: 0.75rem;
            color: #718096;
            margin-top: 4px;
        }
        
        /* Title styling */
        .gradient-title {
            background: linear-gradient(120deg, #63b3ed 0%, #b794f4 50%, #f6ad55 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.2rem;
            margin-bottom: 4px;
        }
        .subtitle {
            color: #a0aec0;
            font-size: 1rem;
            margin-bottom: 24px;
        }
        
        /* Highlight labels */
        .badge {
            background-color: rgba(66, 153, 225, 0.15);
            color: #63b3ed;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid rgba(66, 153, 225, 0.25);
        }
        
        /* Table styles override */
        div[data-testid="stTable"] table {
            background-color: #161b31 !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        div[data-testid="stDataFrame"] {
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        /* Tabs active styling */
        button[data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply CSS
inject_custom_css()

# Common functions to format Plotly charts to look ultra-premium
def premium_plotly_layout(fig, title="", height=400):
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 18, 'family': 'Outfit', 'color': '#ffffff', 'weight': 'bold'}
        },
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#a0aec0", family="Outfit"),
            title=dict(font=dict(color="#a0aec0", family="Outfit"))
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#a0aec0", family="Outfit"),
            title=dict(font=dict(color="#a0aec0", family="Outfit"))
        ),
        legend=dict(
            bgcolor="rgba(15, 18, 32, 0.6)",
            bordercolor="rgba(255,255,255,0.05)",
            borderwidth=1,
            font=dict(color="#cbd5e0", family="Outfit")
        )
    )
    return fig

# Custom Data Source Selection & Column Mapping
st.sidebar.markdown("---")
st.sidebar.subheader("Data Source Configuration")
data_option = st.sidebar.radio("Select Data Source:", ["Demo Data (SuperStore.csv)", "Upload Custom CSV"])

raw_df = None
date_col = 'Order Date'
ship_col = 'Ship Date'
sales_col = 'Sales'
cat_col = 'Category'
subcat_col = 'Sub-Category'
region_col = 'Region'
order_col = 'Order ID'

if data_option == "Demo Data (SuperStore.csv)":
    try:
        raw_df = pd.read_csv('SuperStore.csv')
    except Exception as e:
        st.sidebar.error(f"Error loading demo file: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            columns = raw_df.columns.tolist()
            
            # Simple heuristics to auto-map columns
            def auto_detect(keywords, default_idx=0):
                for kw in keywords:
                    for c in columns:
                        if kw.lower() in c.lower():
                            return c
                return columns[default_idx] if len(columns) > default_idx else columns[0]
                
            det_date = auto_detect(['order date', 'order_date', 'date', 'orderdate'], 0)
            det_sales = auto_detect(['sales', 'revenue', 'amount', 'value', 'price'], 1 if len(columns) > 1 else 0)
            det_cat = auto_detect(['category', 'product_category', 'cat'], 2 if len(columns) > 2 else 0)
            det_region = auto_detect(['region', 'geography', 'country', 'state', 'city'], 3 if len(columns) > 3 else 0)
            det_subcat = auto_detect(['sub-category', 'sub_category', 'subcategory', 'subcat'], 4 if len(columns) > 4 else 0)
            det_order = auto_detect(['order id', 'order_id', 'orderid', 'id'], 5 if len(columns) > 5 else 0)
            det_ship = auto_detect(['ship date', 'ship_date', 'shipping_date'], 6 if len(columns) > 6 else 0)
            
            # Allow user to verify mappings
            st.sidebar.markdown("**Verify Column Mapping**")
            with st.sidebar.expander("Adjust Column Mappings", expanded=False):
                date_col = st.selectbox("Date Column", columns, index=columns.index(det_date))
                sales_col = st.selectbox("Sales Value Column", columns, index=columns.index(det_sales))
                cat_col = st.selectbox("Category Column", columns, index=columns.index(det_cat))
                region_col = st.selectbox("Region Column", columns, index=columns.index(det_region))
                subcat_col = st.selectbox("Sub-Category Column", columns, index=columns.index(det_subcat))
                order_col = st.selectbox("Order ID Column", columns, index=columns.index(det_order))
                
                has_ship = st.checkbox("Dataset has Shipping Dates?", value=(det_ship in columns))
                if has_ship:
                    ship_col = st.selectbox("Ship Date Column", columns, index=columns.index(det_ship) if det_ship in columns else 0)
                else:
                    ship_col = None
                    
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

if raw_df is None:
    st.info("Welcome! Please upload your dataset using the sidebar to begin, or select 'Demo Data (SuperStore.csv)' to explore the system with demo metrics.")
    st.stop()

# Helper function to assign season
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

# Cache processing so it runs quickly
@st.cache_data
def process_data(df_raw, date_c, ship_c, sales_c, cat_c, subcat_c, region_c, order_c):
    df_clean = pd.DataFrame()
    
    # Try different date parsing formats
    df_clean['Order Date'] = pd.to_datetime(df_raw[date_c], errors='coerce')
    if ship_c and ship_c in df_raw.columns:
        df_clean['Ship Date'] = pd.to_datetime(df_raw[ship_c], errors='coerce')
    else:
        # Default ship date to order date + 4 days
        df_clean['Ship Date'] = df_clean['Order Date'] + pd.to_timedelta(4, unit='D')
        
    df_clean['Sales'] = pd.to_numeric(df_raw[sales_c], errors='coerce').fillna(0)
    df_clean['Category'] = df_raw[cat_c].astype(str)
    df_clean['Region'] = df_raw[region_c].astype(str)
    df_clean['Sub-Category'] = df_raw[subcat_c].astype(str)
    df_clean['Order ID'] = df_raw[order_c].astype(str)
    
    df_clean = df_clean.dropna(subset=['Order Date']).sort_values('Order Date').reset_index(drop=True)
    
    df_clean['Year'] = df_clean['Order Date'].dt.year
    df_clean['Month'] = df_clean['Order Date'].dt.month
    df_clean['Week Number'] = df_clean['Order Date'].dt.isocalendar().week
    df_clean['Day of Week'] = df_clean['Order Date'].dt.dayofweek
    df_clean['Quarter'] = df_clean['Order Date'].dt.quarter
    df_clean['Season'] = df_clean['Month'].apply(get_season)
    df_clean['YearMonth'] = df_clean['Order Date'].dt.to_period('M')
    return df_clean

try:
    df = process_data(raw_df, date_col, ship_col, sales_col, cat_col, subcat_col, region_col, order_col)
except Exception as e:
    st.error(f"Error processing data: {e}. Please check your column mappings in the sidebar.")
    st.stop()

# Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product demand Segments"],
    label_visibility="collapsed"
)

# Page-specific parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Page Parameters")

if page == "Sales Overview":
    st.sidebar.info("Use the filters inside the main dashboard panel to slice historical trends.")
    
elif page == "Forecast Explorer":
    with st.sidebar.expander("SARIMA Tuning Options", expanded=False):
        p_param = st.slider("p (autoregressive)", 0, 3, 1)
        d_param = st.slider("d (differencing)", 0, 2, 1)
        q_param = st.slider("q (moving average)", 0, 3, 1)
        st.markdown("**Seasonal Parameters**")
        P_param = st.slider("P (seasonal AR)", 0, 2, 1)
        D_param = st.slider("D (seasonal diff)", 0, 1, 1)
        Q_param = st.slider("Q (seasonal MA)", 0, 2, 1)
        m_param = st.number_input("m (seasonal period)", value=12, min_value=1, max_value=24, step=1)
        
elif page == "Anomaly Report":
    anomaly_method = st.sidebar.radio("Choose Anomaly Method:", ["Isolation Forest", "Rolling Z-Score"])
    with st.sidebar.expander("Sensitivity Controls", expanded=False):
        if anomaly_method == "Isolation Forest":
            contamination_rate = st.slider("Contamination (expected fraction of outliers)", 0.01, 0.15, 0.05, step=0.01)
        else:
            z_threshold = st.slider("Z-Score Threshold (deviations)", 1.5, 3.5, 2.0, step=0.1)
            rolling_window = st.slider("Rolling Window Size (weeks)", 4, 16, 8, step=1)
            
elif page == "Product demand Segments":
    with st.sidebar.expander("Clustering Configurations", expanded=False):
        num_clusters = st.slider("Number of Clusters", 2, 6, 4)

# Data Diagnostics in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Data Health & Diagnostics")
st.sidebar.markdown(f"**Total Records:** `{len(df):,}`")
if len(df) > 0:
    date_min = df['Order Date'].min().strftime('%Y-%m-%d')
    date_max = df['Order Date'].max().strftime('%Y-%m-%d')
    st.sidebar.markdown(f"**Timeline:** `{date_min}` to `{date_max}`")
    st.sidebar.markdown(f"**Regions:** `{df['Region'].nunique()}`")
    st.sidebar.markdown(f"**Categories:** `{df['Category'].nunique()}`")
    st.sidebar.markdown(f"**Sub-Categories:** `{df['Sub-Category'].nunique()}`")

# ----------------- PAGE 1: SALES OVERVIEW DASHBOARD -----------------
if "Sales Overview" in page:
    st.markdown('<div class="gradient-title">Sales Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive historical sales insights and seasonal trend analysis</div>', unsafe_allow_html=True)
    
    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Filters")
    
    categories = df['Category'].unique().tolist()
    regions = df['Region'].unique().tolist()
    
    selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)
    selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
    
    if not selected_categories or not selected_regions:
        st.warning("Please select at least one Category and Region in the sidebar filters.")
    else:
        # Filter Data
        filtered_df = df[
            (df['Category'].isin(selected_categories)) & 
            (df['Region'].isin(selected_regions))
        ]
        
        # Calculate KPI values
        total_sales_val = filtered_df['Sales'].sum()
        total_orders_val = filtered_df['Order ID'].nunique()
        
        # Monthly average calculation
        monthly_agg = filtered_df.set_index("Order Date")["Sales"].resample("MS").sum()
        monthly_avg_val = monthly_agg.mean() if not monthly_agg.empty else 0
        
        # Top Sub-Category
        if not filtered_df.empty:
            subcat_sales = filtered_df.groupby("Sub-Category")["Sales"].sum()
            top_subcat = subcat_sales.idxmax()
            top_subcat_val = subcat_sales.max()
        else:
            top_subcat = "N/A"
            top_subcat_val = 0
            
        # Display KPI cards in columns using HTML injection
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(
                f"""
                <div class="metric-card metric-blue">
                    <div class="metric-label">Total Revenue</div>
                    <div class="metric-val">${total_sales_val:,.2f}</div>
                    <div class="metric-desc">Filtered sales total</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with kpi_col2:
            st.markdown(
                f"""
                <div class="metric-card metric-purple">
                    <div class="metric-label">Monthly Average</div>
                    <div class="metric-val">${monthly_avg_val:,.2f}</div>
                    <div class="metric-desc">Avg sales per month</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with kpi_col3:
            st.markdown(
                f"""
                <div class="metric-card metric-orange">
                    <div class="metric-label">Total Orders</div>
                    <div class="metric-val">{total_orders_val:,}</div>
                    <div class="metric-desc">Unique order counts</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with kpi_col4:
            st.markdown(
                f"""
                <div class="metric-card metric-green">
                    <div class="metric-label">Top Sub-Category</div>
                    <div class="metric-val" style="font-size: 1.4rem; padding-top: 6px; padding-bottom: 6px;">{top_subcat}</div>
                    <div class="metric-desc">${top_subcat_val:,.2f} sales</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            # Total sales by year
            yearly_sales = filtered_df.groupby("Year")["Sales"].sum().reset_index()
            yearly_sales["Year"] = yearly_sales["Year"].astype(str)
            
            fig_year = px.bar(
                yearly_sales, 
                x="Year", 
                y="Sales", 
                color="Sales",
                color_continuous_scale=["#1e3c72", "#63b3ed"],
                labels={"Sales": "Sales ($)", "Year": "Year"}
            )
            fig_year.update_coloraxes(showscale=False)
            premium_plotly_layout(fig_year, "Total Sales by Year", height=380)
            st.plotly_chart(fig_year, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            # Monthly sales trend
            monthly_trend = filtered_df.set_index("Order Date")["Sales"].resample("MS").sum().reset_index()
            
            fig_trend = px.line(
                monthly_trend, 
                x="Order Date", 
                y="Sales",
                markers=True,
                labels={"Sales": "Sales ($)", "Order Date": "Month"}
            )
            fig_trend.update_traces(
                line=dict(color="#b794f4", width=3.5),
                marker=dict(color="#ffffff", size=7, line=dict(color="#b794f4", width=1.5))
            )
            premium_plotly_layout(fig_trend, "Monthly Sales Trend", height=380)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Export raw filtered data as CSV
        st.download_button(
            label="Export Filtered Sales Data (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_sales_data.csv',
            mime='text/csv'
        )

# ----------------- PAGE 2: FORECAST EXPLORER -----------------
elif "Forecast Explorer" in page:
    st.markdown('<div class="gradient-title">Demand Forecast Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict future sales and evaluate SARIMA model metrics</div>', unsafe_allow_html=True)
    
    # Cached Forecast Calculator to avoid recalculating on slider changes
    @st.cache_resource
    def run_sarima_forecast(seg_type, seg_val, p, d, q, P, D, Q, m):
        if seg_type == "Category":
            sub_df = df[df["Category"] == seg_val]
        else:
            sub_df = df[df["Region"] == seg_val]
            
        # Resample to monthly
        series = sub_df.set_index("Order Date")["Sales"].resample("MS").sum()
        
        # Evaluation split (last 3 months)
        train_series = series.iloc[:-3]
        test_series = series.iloc[-3:]
        
        # Fit train model (SARIMA)
        if len(series) >= (m + 6):
            model_eval = SARIMAX(
                train_series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_eval = model_eval.fit(disp=False)
            eval_preds = fit_eval.forecast(steps=3)
            
            # Fit full model for final forecasting
            model_full = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_full = model_full.fit(disp=False)
            full_forecast_3m = fit_full.forecast(steps=3)
        else:
            # Fall back to non-seasonal ARIMA
            model_eval = SARIMAX(
                train_series,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_eval = model_eval.fit(disp=False)
            eval_preds = fit_eval.forecast(steps=3)
            
            model_full = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_full = model_full.fit(disp=False)
            full_forecast_3m = fit_full.forecast(steps=3)
        
        return series, test_series, eval_preds, full_forecast_3m, mean_absolute_error(test_series, eval_preds), math.sqrt(mean_squared_error(test_series, eval_preds))

    # Selector Form Controls
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    fc_col1, fc_col2, fc_col3 = st.columns(3)
    
    with fc_col1:
        segment_type = st.selectbox("Select Forecast Dimension:", ["Category", "Region"])
    
    with fc_col2:
        if segment_type == "Category":
            segment_options = df['Category'].unique().tolist()
        else:
            segment_options = df['Region'].unique().tolist()
        segment_val = st.selectbox(f"Select Specific {segment_type}:", segment_options)
        
    # Generate dynamic month list for user horizon selection
    sub_sel_df = df[df["Category"] == segment_val] if segment_type == "Category" else df[df["Region"] == segment_val]
    series_temp = sub_sel_df.set_index("Order Date")["Sales"].resample("MS").sum()
    
    if len(series_temp) >= 3:
        last_date = series_temp.index[-1]
        future_months = [(last_date + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(1, 4)]
    else:
        future_months = ["Month 1", "Month 2", "Month 3"]
        
    with fc_col3:
        # Date range slider formatted to show horizon months dynamically
        horizon = st.select_slider(
            "Select Forecast Horizon:",
            options=[1, 2, 3],
            value=3,
            format_func=lambda x: f"{x} Month{'s' if x > 1 else ''} Ahead ({', '.join(future_months[:x]) if x > 1 else future_months[0]})"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate Forecast
    with st.spinner("Fitting SARIMA model and forecasting..."):
        series, test_series, eval_preds, full_forecast_3m, mae, rmse = run_sarima_forecast(
            segment_type, segment_val, 
            p_param, d_param, q_param, 
            P_param, D_param, Q_param, m_param
        )
    
    # Restrict future predictions to selected horizon
    future_preds = full_forecast_3m.iloc[:horizon]
    
    # Generate Chart
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    # Only show last 24 months of actuals for cleaner view, but keep whole history in mind
    recent_actuals = series.iloc[-24:]
    
    fig_fc = go.Figure()
    
    # 1. Historical Actuals (Training period)
    fig_fc.add_trace(go.Scatter(
        x=recent_actuals.index[:-3],
        y=recent_actuals.values[:-3],
        name="Historical Actuals",
        mode="lines+markers",
        line=dict(color="#3182ce", width=2.5),
        marker=dict(size=4)
    ))
    
    # 2. Holdout Test Actuals (Last 3 Months)
    fig_fc.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series.values,
        name="Test Period (Actual)",
        mode="lines+markers",
        line=dict(color="#38a169", width=2.5),
        marker=dict(size=6, symbol="square")
    ))
    
    # 3. Holdout Test Predictions (Evaluating SARIMA)
    fig_fc.add_trace(go.Scatter(
        x=eval_preds.index,
        y=eval_preds.values,
        name="Test Period (SARIMA Prediction)",
        mode="lines+markers",
        line=dict(color="#dd6b20", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond")
    ))
    
    # 4. Future Forecast
    # Connect last actual point to the forecast
    connect_dates = [series.index[-1]] + list(future_preds.index)
    connect_values = [series.values[-1]] + list(future_preds.values)
    
    fig_fc.add_trace(go.Scatter(
        x=connect_dates,
        y=connect_values,
        name="Future Horizon Forecast",
        mode="lines+markers",
        line=dict(color="#b794f4", width=3, dash="dot"),
        marker=dict(size=8, symbol="circle")
    ))
    
    # Visual forecast divider line
    fig_fc.add_vline(x=series.index[-1].timestamp() * 1000, line_width=1.5, line_dash="dash", line_color="#a0aec0")
    fig_fc.add_annotation(
        x=series.index[-1],
        y=max(series.values) * 0.95,
        text="Forecast Start",
        showarrow=False,
        xanchor="left",
        font=dict(color="#a0aec0", size=10)
    )
    
    premium_plotly_layout(fig_fc, f"SARIMA Model Forecast: {segment_val} Sales", height=450)
    fig_fc.update_layout(xaxis_title="Timeline", yaxis_title="Sales ($)")
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display MAE & RMSE Metrics below the chart
    test_start = test_series.index[0].strftime('%B %Y')
    test_end = test_series.index[-1].strftime('%B %Y')
    st.markdown("### Model Quality & Accuracy Assessment")
    st.markdown(f"Below are the error metrics computed on the 3-month holdout test set ({test_start} - {test_end}).")
    
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.markdown(
            f"""
            <div class="metric-card metric-orange">
                <div class="metric-label">Mean Absolute Error (MAE)</div>
                <div class="metric-val">${mae:,.2f}</div>
                <div class="metric-desc">Avg absolute deviation from test actuals</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with met_col2:
        st.markdown(
            f"""
            <div class="metric-card metric-purple">
                <div class="metric-label">Root Mean Squared Error (RMSE)</div>
                <div class="metric-val">${rmse:,.2f}</div>
                <div class="metric-desc">Standard deviation of residuals</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with met_col3:
        # Calculate MAPE
        mape_val = np.mean(np.abs(test_series.values - eval_preds.values) / test_series.values) * 100
        st.markdown(
            f"""
            <div class="metric-card metric-blue">
                <div class="metric-label">Mean Absolute Percentage Error (MAPE)</div>
                <div class="metric-val">{mape_val:.2f}%</div>
                <div class="metric-desc">Average percentage forecast error</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Export forecast values
    forecast_download_df = pd.DataFrame({
        "Timeline": future_preds.index.strftime('%Y-%m-%d'),
        "Forecasted Sales ($)": future_preds.values.round(2)
    })
    st.download_button(
        label="📥 Export 3-Month Forecast Report (CSV)",
        data=forecast_download_df.to_csv(index=False).encode('utf-8'),
        file_name=f'demand_forecast_{segment_val.replace(" ", "_")}.csv',
        mime='text/csv'
    )

# ----------------- PAGE 3: ANOMALY REPORT -----------------
elif "Anomaly Report" in page:
    st.markdown('<div class="gradient-title">Weekly Sales Anomaly Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Identify historical sales spikes and sudden drops using Isolation Forest and Rolling Z-Score</div>', unsafe_allow_html=True)
    
    # Anomaly algorithm selector info
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    algo_col1, algo_col2 = st.columns([1, 2])
    with algo_col1:
        st.markdown(f"**Active Method:** `{anomaly_method}`")
    with algo_col2:
        if anomaly_method == "Isolation Forest":
            st.info(
                f"**Isolation Forest**: Builds an ensemble of isolation trees. "
                f"Inputs are weekly sales volume and WoW percentage change. Flagged weeks represent anomalous sales levels "
                f"or highly sudden jumps/drops (contamination set to {contamination_rate * 100:.1f}%)."
            )
        else:
            st.info(
                f"**Rolling Z-Score**: Computes standard deviations from an {rolling_window}-week centered rolling mean. "
                f"Adapts to long-term business growth. Weeks exceeding **|Z| > {z_threshold:.1f}** are flagged as anomalies."
            )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Run anomaly detection
    @st.cache_data
    def run_anomalies(method_name, contam, z_t, r_w):
        weekly_sales = df.set_index("Order Date")["Sales"].resample("W").sum()
        weekly_sales = weekly_sales.asfreq("W", fill_value=0)
        
        if method_name == "Isolation Forest":
            features_df = pd.DataFrame({"sales": weekly_sales})
            features_df["pct_change"] = features_df["sales"].pct_change().fillna(0)
            features_df["pct_change"] = features_df["pct_change"].replace([np.inf, -np.inf], 0)
            
            iso = IsolationForest(n_estimators=200, contamination=contam, random_state=42)
            iso.fit(features_df[["sales", "pct_change"]])
            
            features_df["is_anomaly"] = iso.predict(features_df[["sales", "pct_change"]]) == -1
            features_df["score"] = iso.decision_function(features_df[["sales", "pct_change"]])
            
            anomalies = features_df[features_df["is_anomaly"]].copy()
            return weekly_sales, anomalies, features_df
            
        else: # Rolling Z-Score
            roll_mean = weekly_sales.rolling(window=r_w, center=True, min_periods=max(2, r_w//2)).mean()
            roll_std = weekly_sales.rolling(window=r_w, center=True, min_periods=max(2, r_w//2)).std()
            
            z_scores = (weekly_sales - roll_mean) / roll_std
            is_anomaly = z_scores.abs() > z_t
            
            features_df = pd.DataFrame({
                "sales": weekly_sales,
                "rolling_mean": roll_mean,
                "rolling_std": roll_std,
                "z_score": z_scores,
                "is_anomaly": is_anomaly
            })
            
            anomalies = features_df[features_df["is_anomaly"]].copy()
            return weekly_sales, anomalies, features_df
            
    weekly_sales, anomalies_df, full_features = run_anomalies(
        anomaly_method, 
        contamination_rate if anomaly_method == "Isolation Forest" else 0.05, 
        z_threshold if anomaly_method == "Rolling Z-Score" else 2.0, 
        rolling_window if anomaly_method == "Rolling Z-Score" else 8
    )
    
    # Plotly Anomaly Plot
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    fig_anom = go.Figure()
    
    # Weekly sales line
    fig_anom.add_trace(go.Scatter(
        x=weekly_sales.index,
        y=weekly_sales.values,
        name="Weekly Sales Volume",
        mode="lines",
        line=dict(color="#2e86ab", width=1.5)
    ))
    
    # Rolling mean for Z-Score
    if anomaly_method == "Rolling Z-Score":
        fig_anom.add_trace(go.Scatter(
            x=full_features.index,
            y=full_features["rolling_mean"],
            name=f"{rolling_window}-Week Rolling Mean",
            mode="lines",
            line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dash")
        ))
        
    # Anomaly scatter points
    marker_color = "#d62828" if anomaly_method == "Isolation Forest" else "#f18f01"
    marker_symbol = "x" if anomaly_method == "Isolation Forest" else "diamond"
    marker_size = 10 if anomaly_method == "Isolation Forest" else 8
    
    fig_anom.add_trace(go.Scatter(
        x=anomalies_df.index,
        y=anomalies_df["sales"],
        name="Detected Anomaly",
        mode="markers",
        marker=dict(
            color=marker_color,
            symbol=marker_symbol,
            size=marker_size,
            line=dict(color="black", width=1)
        )
    ))
    
    premium_plotly_layout(fig_anom, f"Weekly Sales - {anomaly_method} Anomalies", height=450)
    fig_anom.update_layout(xaxis_title="Timeline", yaxis_title="Sales ($)")
    st.plotly_chart(fig_anom, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Table of anomalies
    st.markdown(f"### Detected Anomalies List ({len(anomalies_df)} weeks flagged)")
    
    table_df = anomalies_df.reset_index()
    table_df["Order Date"] = table_df["Order Date"].dt.strftime("%Y-%m-%d")
    
    if anomaly_method == "Isolation Forest":
        table_df = table_df[["Order Date", "sales", "pct_change", "score"]].rename(
            columns={
                "sales": "Weekly Sales ($)",
                "pct_change": "WoW Change (%)",
                "score": "Anomaly Score (lower = more anomalous)"
            }
        )
        table_df["WoW Change (%)"] = (table_df["WoW Change (%)"] * 100).round(2).map(lambda x: f"{x:+.2f}%")
        table_df["Weekly Sales ($)"] = table_df["Weekly Sales ($)"].round(2).map(lambda x: f"${x:,.2f}")
        table_df["Anomaly Score (lower = more anomalous)"] = table_df["Anomaly Score (lower = more anomalous)"].round(4)
        table_df = table_df.sort_values("Anomaly Score (lower = more anomalous)")
    else:
        table_df = table_df[["Order Date", "sales", "rolling_mean", "z_score"]].rename(
            columns={
                "sales": "Weekly Sales ($)",
                "rolling_mean": "Rolling Mean ($)",
                "z_score": "Z-Score"
            }
        )
        table_df["Weekly Sales ($)"] = table_df["Weekly Sales ($)"].round(2).map(lambda x: f"${x:,.2f}")
        table_df["Rolling Mean ($)"] = table_df["Rolling Mean ($)"].round(2).map(lambda x: f"${x:,.2f}")
        table_df["Z-Score"] = table_df["Z-Score"].round(2)
        table_df = table_df.sort_values("Z-Score", key=lambda s: s.abs(), ascending=False)
        
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Export flagged anomalies
    st.download_button(
        label="📥 Export Flagged Anomalies Report (CSV)",
        data=table_df.to_csv(index=False).encode('utf-8'),
        file_name=f'sales_anomalies_{anomaly_method.replace(" ", "_")}.csv',
        mime='text/csv'
    )

# ----------------- PAGE 4: PRODUCT DEMAND SEGMENTS -----------------
elif "Product demand Segments" in page:
    st.markdown('<div class="gradient-title">Product Demand Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Segment sub-categories into demand groups using K-Means and PCA dimensionality reduction</div>', unsafe_allow_html=True)
    
    # Run Clustering Cache
    @st.cache_data
    def run_demand_clustering(n_clust):
        # Feature 1: Total Sales Volume
        total_sales = df.groupby("Sub-Category")["Sales"].sum().rename("total_sales_volume")
        
        # Feature 2: YoY Sales Growth Rate
        yearly_sales = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
        
        first_year, last_year = yearly_sales.columns.min(), yearly_sales.columns.max()
        n_years = last_year - first_year
        if n_years <= 0:
            n_years = 1
            
        def compound_growth(row):
            start, end = row[first_year], row[last_year]
            if start <= 0:
                return 0.0
            return ((end / start) ** (1 / n_years) - 1) * 100
            
        growth_rate = yearly_sales.apply(compound_growth, axis=1).rename("yoy_growth_rate_pct")
        
        # Feature 3: Sales Volatility
        monthly_sales = df.groupby(["Sub-Category", "YearMonth"])["Sales"].sum()
        volatility = monthly_sales.groupby("Sub-Category").std().rename("sales_volatility").fillna(0)
        
        # Feature 4: Average Order Value
        order_value = (
            df.groupby(["Sub-Category", "Order ID"])["Sales"].sum()
            .groupby("Sub-Category").mean()
            .rename("avg_order_value")
        )
        
        features_df = pd.concat([total_sales, growth_rate, volatility, order_value], axis=1).fillna(0)
        
        # Scale features
        feature_cols = ["total_sales_volume", "yoy_growth_rate_pct", "sales_volatility", "avg_order_value"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df[feature_cols])
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clust, n_init=10, random_state=42)
        features_df["cluster"] = kmeans.fit_predict(X_scaled)
        
        # Profile clusters and label them
        cluster_profile = features_df.groupby("cluster")[feature_cols].mean()
        
        def label_clusters(profile):
            labels = {}
            growth_sorted = profile["yoy_growth_rate_pct"].sort_values(ascending=False)
            growth_cluster = growth_sorted.index[0]
            labels[growth_cluster] = "Growing Demand"
            
            lowest_growth_cluster = growth_sorted.index[-1]
            if growth_sorted.iloc[-1] < 0:
                labels[lowest_growth_cluster] = "Declining Demand"
                
            remaining = [c for c in profile.index if c not in labels]
            if remaining:
                vol_median = profile.loc[remaining, "total_sales_volume"].median() if len(remaining) > 1 else profile.loc[remaining[0], "total_sales_volume"]
                for c in remaining:
                    high_volume = profile.loc[c, "total_sales_volume"] >= vol_median
                    is_volatile = profile.loc[c, "sales_volatility"] >= profile["sales_volatility"].median()
                    if high_volume and not is_volatile:
                        labels[c] = "High Volume, Stable Demand"
                    elif not high_volume and is_volatile:
                        labels[c] = "Low Volume, Volatile Demand"
                    elif high_volume and is_volatile:
                        labels[c] = "High Volume, Volatile Demand"
                    else:
                        labels[c] = "Low Volume, Stable Demand"
            return pd.Series(labels)
            
        cluster_labels = label_clusters(cluster_profile)
        features_df["cluster_label"] = features_df["cluster"].map(cluster_labels)
        
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_scaled)
        features_df["pca_1"] = pca_coords[:, 0]
        features_df["pca_2"] = pca_coords[:, 1]
        
        return features_df, cluster_labels
        
    features, cluster_labels = run_demand_clustering(num_clusters)
    
    # 2D PCA Scatter plot
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    # Dynamic sub-categories retriever for card UI
    def get_subcategories_in_segment(seg_name):
        subset = features[features["cluster_label"] == seg_name]
        return ", ".join(subset.index.tolist()) if not subset.empty else "No sub-categories in this segment"
    
    # Color palette supporting dynamic labeling options
    color_map = {
        "Growing Demand": "#a23b72",
        "Declining Demand": "#c73e1d",
        "Low Volume, Stable Demand": "#2e86ab",
        "High Volume, Volatile Demand": "#f18f01",
        "High Volume, Stable Demand": "#38a169",
        "Low Volume, Volatile Demand": "#805ad5"
    }
    
    fig_cluster = px.scatter(
        features.reset_index(),
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_name="Sub-Category",
        text="Sub-Category",
        color_discrete_map=color_map,
        labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2", "cluster_label": "Demand Cluster"}
    )
    fig_cluster.update_traces(
        marker=dict(size=25, opacity=0.85, line=dict(color="#ffffff", width=1)),
        textposition="top center",
        textfont=dict(size=9, family="Outfit", color="#ffffff")
    )
    premium_plotly_layout(fig_cluster, "Product Sub-Category Segments (PCA Projection Space)", height=550)
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stocking Strategy Recommendations Table
    st.markdown("### Stocking Strategy & Actionable Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown(
            f"""
            <div class="premium-card" style="height: 250px; overflow-y: auto;">
                <h4 style="margin: 0; color: #a23b72;"><span class="badge" style="background-color: rgba(162, 59, 114, 0.15); color: #a23b72; border-color: rgba(162, 59, 114, 0.25);">GROWING DEMAND</span></h4>
                <p style="margin-top: 10px; font-weight: 600; color: #ffffff; font-size: 0.95rem;">Sub-Categories: {get_subcategories_in_segment("Growing Demand")}</p>
                <p style="font-size: 0.85rem; color: #cbd5e0; line-height: 1.5;">
                    <b>Stocking Strategy: Aggressive Growth Buffer.</b><br>
                    Extreme growth rate and high volume indicate rapid adoption. 
                    Establish a larger safety stock buffer, coordinate closely with suppliers on lead times, 
                    and secure bulk material supply contracts to prevent growth constraints.
                </p>
            </div>
            <div class="premium-card" style="height: 250px; overflow-y: auto;">
                <h4 style="margin: 0; color: #2e86ab;"><span class="badge" style="background-color: rgba(46, 134, 171, 0.15); color: #2e86ab; border-color: rgba(46, 134, 171, 0.25);">LOW VOLUME, STABLE DEMAND</span></h4>
                <p style="margin-top: 10px; font-weight: 600; color: #ffffff; font-size: 0.95rem;">Sub-Categories: {get_subcategories_in_segment("Low Volume, Stable Demand")}</p>
                <p style="font-size: 0.85rem; color: #cbd5e0; line-height: 1.5;">
                    <b>Stocking Strategy: Automated Reorder Points (ROP).</b><br>
                    Low volume but highly stable and predictable demand. Use automated Min-Max inventory 
                    replenishment. Keep minimal safety stock since volatility is low, freeing up valuable storage 
                    space and cash.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with rec_col2:
        st.markdown(
            f"""
            <div class="premium-card" style="height: 250px; overflow-y: auto;">
                <h4 style="margin: 0; color: #f18f01;"><span class="badge" style="background-color: rgba(241, 143, 1, 0.15); color: #f18f01; border-color: rgba(241, 143, 1, 0.25);"> HIGH VOLUME, VOLATILE DEMAND</span></h4>
                <p style="margin-top: 10px; font-weight: 600; color: #ffffff; font-size: 0.95rem;">Sub-Categories: {get_subcategories_in_segment("High Volume, Volatile Demand")}</p>
                <p style="font-size: 0.85rem; color: #cbd5e0; line-height: 1.5;">
                    <b>Stocking Strategy: Dynamic Safety Stock & Rolling Forecasting.</b><br>
                    High-revenue contribution but unpredictable ordering sizes. Maintain robust safety stocks 
                    to cover demand spikes. Run monthly rolling forecasts and share customer demand pipelines 
                    directly with distribution partners to smooth ordering signals.
                </p>
            </div>
            <div class="premium-card" style="height: 250px; overflow-y: auto;">
                <h4 style="margin: 0; color: #c73e1d;"><span class="badge" style="background-color: rgba(199, 62, 29, 0.15); color: #c73e1d; border-color: rgba(199, 62, 29, 0.25);"> DECLINING DEMAND</span></h4>
                <p style="margin-top: 10px; font-weight: 600; color: #ffffff; font-size: 0.95rem;">Sub-Categories: {get_subcategories_in_segment("Declining Demand")}</p>
                <p style="font-size: 0.85rem; color: #cbd5e0; line-height: 1.5;">
                    <b>Stocking Strategy: Just-in-Time (JIT) / Run-out Clearance.</b><br>
                    Negative YoY growth and high volatility. Transition to order-to-order procurement 
                    or drop-shipping where possible. Run promotional campaigns or bundle liquidations to discharge 
                    remaining stock and prevent capital obsolescence.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Demand cluster details table
    st.markdown("### Sub-Category Demand Profiles Table")
    
    details_df = features.reset_index()[["Sub-Category", "cluster_label", "total_sales_volume", "yoy_growth_rate_pct", "sales_volatility", "avg_order_value"]]
    details_df = details_df.rename(
        columns={
            "cluster_label": "Demand Segment",
            "total_sales_volume": "Total Sales ($)",
            "yoy_growth_rate_pct": "YoY Growth Rate (%)",
            "sales_volatility": "Sales Volatility ($)",
            "avg_order_value": "Avg Order Value ($)"
        }
    )
    details_df["Total Sales ($)"] = details_df["Total Sales ($)"].round(2).map(lambda x: f"${x:,.2f}")
    details_df["YoY Growth Rate (%)"] = details_df["YoY Growth Rate (%)"].round(2).map(lambda x: f"{x:+.2f}%")
    details_df["Sales Volatility ($)"] = details_df["Sales Volatility ($)"].round(2).map(lambda x: f"${x:,.2f}")
    details_df["Avg Order Value ($)"] = details_df["Avg Order Value ($)"].round(2).map(lambda x: f"${x:,.2f}")
    
    st.dataframe(details_df.sort_values("Demand Segment"), use_container_width=True, hide_index=True)

    # Export clustering results
    st.download_button(
        label="📥 Export Product Demand Profiles & Strategy (CSV)",
        data=details_df.to_csv(index=False).encode('utf-8'),
        file_name='product_demand_profiles.csv',
        mime='text/csv'
    )
