import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for navbar and general styling with enhanced interactive card colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }

    /* Navbar styles */
    .navbar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }

    .nav-item {
        font-size: 1.2rem;
        font-weight: 600;
        color: #6b7280;
        cursor: pointer;
        padding-bottom: 0.25rem;
        transition: color 0.3s ease;
    }

    .nav-item:hover {
        color: #3b82f6;
    }

    .nav-item-selected {
        color: #3b82f6;
        border-bottom: 3px solid #3b82f6;
        cursor: default;
    }

    /* Enhanced interactive metric cards with gradient background and shadow */
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.5);
        text-align: center;
        transition: all 0.3s ease;
        border: none;
        user-select: none;
    }

    .metric-card:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.8);
        transform: translateY(-6px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.85;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Enhanced cards for general content */
    .card {
        background: linear-gradient(135deg, #f9fafb 0%, #e0e7ff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        border: none;
        margin: 1rem 0;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
        transform: translateY(-5px);
    }

    /* Prediction card with vibrant gradient and shadow */
    .prediction-card {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(67, 56, 202, 0.6);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }

    .prediction-card:hover {
        box-shadow: 0 12px 45px rgba(67, 56, 202, 0.85);
        transform: translateY(-8px);
    }

    .prediction-result {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    /* Style Streamlit buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.3s ease;
        user-select: none;
    }

    .stButton > button:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        rfm = pd.read_csv('customer_segments.csv')
        return rfm
    except FileNotFoundError:
        np.random.seed(42)
        n_customers = 4339
        segments = ['Champions', 'Loyal Customers', 'At-Risk Customers', 'Occasional Buyers']
        data = {
            'CustomerID': range(1, n_customers + 1),
            'Recency': np.random.randint(1, 375, n_customers),
            'Frequency': np.random.randint(1, 50, n_customers),
            'Monetary': np.random.uniform(10, 10000, n_customers),
            'Segment': np.random.choice(segments, n_customers, p=[0.15, 0.35, 0.25, 0.25])
        }
        return pd.DataFrame(data)

@st.cache_resource
def load_models():
    try:
        scaler = load('scaler.joblib')
        kmeans = load('kmeans_model.joblib')
        return scaler, kmeans
    except FileNotFoundError:
        return None, None

rfm = load_data()
scaler, kmeans = load_models()

cluster_names = {
    0: "Loyal Customers",
    1: "At-Risk Customers",
    2: "Champions",
    3: "Occasional Buyers"
}

segment_colors = {
    'Champions': '#ef4444',
    'Loyal Customers': '#10b981',
    'At-Risk Customers': '#f59e0b',
    'Occasional Buyers': '#3b82f6'
}

st.markdown('<h1 class="main-title">📊 Customer Analytics Dashboard</h1>', unsafe_allow_html=True)

# Navbar items
nav_items = ["📊 Dashboard", "🔮 Predictions", "📈 Analytics", "📥 Export"]
selected_nav = st.radio("", nav_items, horizontal=True)

# Precompute some totals for metric cards used in dashboard or analytics
total_customers = len(rfm)
total_orders = rfm['Frequency'].sum()
total_revenue = rfm['Monetary'].sum()
total_days = rfm['Recency'].max()

filtered_rfm = rfm.copy()

if selected_nav == "📊 Dashboard":
    # Swap: Show Analytics cards here
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">👥 Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-label">🔄 Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-label">💰 Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_days}</div>
            <div class="metric-label">📅 Total Days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Customer Segments</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        segments = filtered_rfm['Segment'].unique()
        selected_segments = st.multiselect(
            "Select segments to analyze:",
            segments,
            default=segments
        )
    with col2:
        viz_type = st.selectbox(
            "Visualization type:",
            ["Scatter Plot", "3D View", "Distributions"]
        )
    dashboard_data = filtered_rfm[filtered_rfm['Segment'].isin(selected_segments)]

    if dashboard_data.empty:
        st.warning("Please select at least one segment.")
    else:
        if viz_type == "Scatter Plot":
            X = dashboard_data[['Recency', 'Frequency', 'Monetary']].values
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Segment': dashboard_data['Segment'].values,
                'Recency': dashboard_data['Recency'].values,
                'Frequency': dashboard_data['Frequency'].values,
                'Monetary': dashboard_data['Monetary'].values
            })
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Segment',
                title="Customer Segments (Principal Component Analysis)",
                color_discrete_map=segment_colors,
                hover_data=['Recency', 'Frequency', 'Monetary']
            )
            fig.update_layout(
                height=600,
                plot_bgcolor='white',
                font_family="Inter"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "3D View":
            fig = px.scatter_3d(
                dashboard_data,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Segment',
                title="Customer Behavior in 3D",
                color_discrete_map=segment_colors
            )
            fig.update_layout(height=600, font_family="Inter")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Recency", "Frequency", "Monetary", "Segment Count"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
            for segment in selected_segments:
                segment_data = dashboard_data[dashboard_data['Segment'] == segment]
                color = segment_colors[segment]
                fig.add_trace(
                    go.Histogram(x=segment_data['Recency'], name=segment,
                                 marker_color=color, opacity=0.7),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Histogram(x=segment_data['Frequency'], name=segment,
                                 marker_color=color, opacity=0.7, showlegend=False),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Histogram(x=segment_data['Monetary'], name=segment,
                                 marker_color=color, opacity=0.7, showlegend=False),
                    row=2, col=1
                )
            segment_counts = dashboard_data['Segment'].value_counts()
            fig.add_trace(
                go.Pie(labels=segment_counts.index, values=segment_counts.values,
                       marker_colors=[segment_colors[seg] for seg in segment_counts.index],
                       showlegend=False),
                row=2, col=2
            )
            fig.update_layout(height=700, font_family="Inter")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Segment Summary</div>', unsafe_allow_html=True)
        summary = dashboard_data.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            'Segment': 'count'
        }).round(2)
        summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Customer_Count']
        st.dataframe(summary, use_container_width=True)

elif selected_nav == "🔮 Predictions":
    st.markdown('<div class="section-header">Customer Segment Prediction</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    st.markdown("""
        <style>
        [data-testid="stVerticalBlock"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        st.markdown("**Enter Customer Data**")
        recency_input = st.number_input("Recency (days)", min_value=0, max_value=1000, value=30)
        frequency_input = st.number_input("Frequency (purchases)", min_value=0, max_value=1000, value=5)
        monetary_input = st.number_input("Monetary (total spent)", min_value=0.0, max_value=100000.0, value=500.0)
        predict_button = st.button("🔮 Predict Segment", use_container_width=True)

    with col2:
        st.markdown("**Input Summary**")
        st.metric("Recency", f"{recency_input} days")
        st.metric("Frequency", f"{frequency_input}")
        st.metric("Monetary", f"${monetary_input:.0f}")

    if predict_button:
        if scaler is None or kmeans is None:
            st.error("Prediction models not found.")
        else:
            new_customer = np.array([[recency_input, frequency_input, monetary_input]])
            new_customer_scaled = scaler.transform(new_customer)
            cluster_pred = kmeans.predict(new_customer_scaled)[0]
            segment_name = cluster_names.get(cluster_pred, "Unknown")

            st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction Result</h3>
                <div class="prediction-result">🎯 {segment_name}</div>
                <p>This customer belongs to the {segment_name} segment based on their RFM profile.</p>
            </div>
            """, unsafe_allow_html=True)

elif selected_nav == "📈 Analytics":
    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)

    # Swap: Show Dashboard cards here (Avg Recency, Avg Frequency, Avg Spend, Total Revenue)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_recency = filtered_rfm['Recency'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_recency:.0f}</div>
            <div class="metric-label">📅 Avg Recency (days)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_frequency = filtered_rfm['Frequency'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_frequency:.1f}</div>
            <div class="metric-label">🔄 Avg Frequency</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_monetary = filtered_rfm['Monetary'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_monetary:.0f}</div>
            <div class="metric-label">💰 Avg Spend</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-label">💎 Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        corr_data = filtered_rfm[['Recency', 'Frequency', 'Monetary']].corr()
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        segment_revenue = filtered_rfm.groupby('Segment')['Monetary'].sum().sort_values()
        fig = px.bar(
            x=segment_revenue.values,
            y=segment_revenue.index,
            orientation='h',
            title="Revenue by Segment",
            color=segment_revenue.values,
            color_continuous_scale="blues"
        )
        fig.update_layout(showlegend=False, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("*Statistical Summary*")
    summary_stats = filtered_rfm[['Recency', 'Frequency', 'Monetary']].describe()
    st.dataframe(summary_stats, use_container_width=True)

elif selected_nav == "📥 Export":
    st.markdown('<div class="section-header">Data Export</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        export_segments = st.multiselect(
            "Select segments to export:",
            filtered_rfm['Segment'].unique(),
            default=filtered_rfm['Segment'].unique()
        )
        export_data = filtered_rfm[filtered_rfm['Segment'].isin(export_segments)]

        if not export_data.empty:
            st.markdown("*Preview*")
            st.dataframe(export_data.head(), use_container_width=True)

            csv = export_data.to_csv(index=False).encode('utf-8')
            st.markdown("""
<style>
/* Your existing styles here ... */

/* Style the download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    width: 100%;
    cursor: pointer;
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.5);
    user-select: none;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.8);
    transform: translateY(-3px);
}
</style>
""", unsafe_allow_html=True)

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='customer_segments.csv',
                mime='text/csv',
                use_container_width=True
            )

    with col2:
        st.markdown("""
                <style>
                [data-testid="stVerticalBlock"] {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                border: 1px solid #e5e7eb;
                margin: 1rem 0;
                }
                </style>
                """, unsafe_allow_html=True)

        st.markdown("**Export Summary**")
        if not export_data.empty:
            st.metric("Records", len(export_data))
            st.metric("Segments", len(export_segments))
            st.metric("Total Value", f"${export_data['Monetary'].sum():,.0f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        Customer Analytics Dashboard
        <div> Made By <strong>Khizar Ishtiaq </strong></div>
    </div>
    """,
    unsafe_allow_html=True
)

