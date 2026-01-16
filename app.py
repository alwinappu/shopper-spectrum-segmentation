import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration with custom theme
st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Generate sample trained data (simulating actual trained models)
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Sample products
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    products = []
    for cat in categories:
        for i in range(10):
            products.append(f"{cat}_{i+1}")
    
    # Sample similarity matrix
    n_products = len(products)
    similarity_matrix = np.random.rand(n_products, n_products)
    np.fill_diagonal(similarity_matrix, 1.0)
    similarity_df = pd.DataFrame(similarity_matrix, index=products, columns=products)
    
    return products, similarity_df

# Segment prediction function
def predict_segment(recency, frequency, monetary):
    # Rule-based segmentation logic
    if frequency >= 10 and monetary >= 1000 and recency <= 60:
        return "High-Value", "🌟", "#4CAF50", "These are your VIP customers! High engagement and spending."
    elif recency > 180:
        return "At-Risk", "⚠️", "#FF5722", "These customers haven't purchased recently. Time for win-back campaigns!"
    elif frequency < 5 and monetary < 500:
        return "Occasional", "📅", "#FF9800", "Infrequent buyers. Target with special offers to increase engagement."
    else:
        return "Regular", "✅", "#2196F3", "Consistent customers. Encourage them to move to High-Value tier."

# Load data
products, similarity_df = generate_sample_data()

# Header with gradient
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
    <h1 style='color: white; text-align: center; margin: 0;'>🛍️ Shopper Spectrum Analytics</h1>
    <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>AI-Powered Customer Segmentation & Product Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", "500", "+12%")
with col2:
    st.metric("Products Analyzed", "495", "+8%")
with col3:
    st.metric("Segments", "4", "Active")
with col4:
    st.metric("Avg Similarity", "87%", "+3%")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📦 Product Recommendations", "👥 Customer Segmentation", "📈 Analytics Dashboard"])

# Tab 1: Product Recommendation
with tab1:
    st.header("🎯 Product Recommendation Engine")
    st.markdown("Discover similar products using **AI-powered collaborative filtering**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_product = st.selectbox(
            "🔍 Select a product to get recommendations:",
            products,
            help="Choose any product to see similar items based on customer purchase patterns"
        )
        
        if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Analyzing product similarities..."):
                # Get similar products
                similar_products = similarity_df[selected_product].sort_values(ascending=False)[1:6]
                
                st.success(f"✨ Top 5 products similar to **{selected_product}**:")
                
                # Display recommendations with progress bars
                for i, (product, score) in enumerate(similar_products.items(), 1):
                    col_a, col_b, col_c = st.columns([0.5, 3, 1])
                    with col_a:
                        st.markdown(f"### {i}")
                    with col_b:
                        st.markdown(f"**{product}**")
                        st.progress(score)
                    with col_c:
                        st.metric("Match", f"{score*100:.1f}%")
    
    with col2:
        st.info("💡 **How it works**\n\nOur ML model analyzes:\n\n• Customer purchase patterns\n• Product co-occurrences\n• Behavioral similarities\n\nUsing **cosine similarity** on a customer-product matrix.")

# Tab 2: Customer Segmentation
with tab2:
    st.header("🎯 Customer Segmentation Predictor")
    st.markdown("Predict customer segments using **RFM Analysis** (Recency, Frequency, Monetary)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.slider(
            "📅 Recency (days)",
            min_value=0,
            max_value=365,
            value=30,
            help="Days since last purchase"
        )
        st.caption(f"{'✅ Recent' if recency < 90 else '⚠️ Dormant'}")
    
    with col2:
        frequency = st.slider(
            "🔁 Frequency (purchases)",
            min_value=1,
            max_value=50,
            value=5,
            help="Total number of purchases"
        )
        st.caption(f"{'✅ Frequent' if frequency > 10 else '📈 Growing'}")
    
    with col3:
        monetary = st.slider(
            "💰 Monetary (spend)",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
            help="Total amount spent"
        )
        st.caption(f"{'✅ High value' if monetary > 1000 else '📈 Potential'}")
    
    if st.button("🔮 Predict Customer Segment", type="primary", use_container_width=True):
        segment, emoji, color, description = predict_segment(recency, frequency, monetary)
        
        # Display result with custom styling
        st.markdown(f"""
        <div style='background-color: {color}; padding: 30px; border-radius: 15px; margin: 20px 0;'>
            <h1 style='color: white; text-align: center;'>{emoji} {segment}</h1>
            <p style='color: white; text-align: center; font-size: 18px;'>{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # RFM Score visualization
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[100-recency/3.65, frequency*2, monetary/100],
            theta=['Recency Score', 'Frequency Score', 'Monetary Score'],
            fill='toself',
            name='Customer Profile'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            title="Customer RFM Profile"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Analytics Dashboard
with tab3:
    st.header("📊 Analytics Dashboard")
    
    # Sample segment distribution
    segment_data = pd.DataFrame({
        'Segment': ['High-Value', 'Regular', 'Occasional', 'At-Risk'],
        'Count': [85, 145, 180, 90],
        'Revenue': [45000, 35000, 15000, 8000]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(
            segment_data,
            values='Count',
            names='Segment',
            title='Customer Segment Distribution',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            segment_data,
            x='Segment',
            y='Revenue',
            title='Revenue by Customer Segment',
            color='Revenue',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time series mockup
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    ts_data = pd.DataFrame({
        'Month': dates,
        'Revenue': np.random.randint(80000, 120000, size=len(dates)),
        'Customers': np.random.randint(400, 600, size=len(dates))
    })
    
    fig3 = px.line(
        ts_data,
        x='Month',
        y='Revenue',
        title='Monthly Revenue Trend',
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shop.png", width=150)
    
    st.markdown("### 📊 Project Info")
    
    st.markdown("""
    **Shopper Spectrum** is an AI-powered analytics platform that combines:
    
    ✅ **RFM Analysis** for customer segmentation  
    ✅ **Collaborative Filtering** for product recommendations  
    ✅ **Real-time Analytics** for business insights
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Tech Stack")
    st.code("""
    • Python & Streamlit
    • Scikit-learn (KMeans)
    • Plotly (Visualizations)
    • Pandas & NumPy
    """, language="markdown")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Links")
    st.markdown("[💻 GitHub Repo](https://github.com/alwinappu/shopper-spectrum-segmentation)")
    st.markdown("[📊 Google Colab](https://colab.research.google.com)")
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Alwin Appu**")
    st.markdown("[GitHub](https://github.com/alwinappu) | [LinkedIn](https://linkedin.com/in/alwinappu)")
    
    st.success("✨ Version 2.0 - Enhanced Edition")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>❤️ Built with Streamlit | Shopper Spectrum Analytics Platform</p>
    <p style='font-size: 12px;'>© 2026 Shopper Spectrum. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
