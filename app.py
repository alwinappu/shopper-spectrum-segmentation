import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration with custom theme
st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 8px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# Generate trained models on first run
@st.cache_resource
def generate_models():
    """Generate trained models if they don't exist"""
    # Generate synthetic data
    np.random.seed(42)
    n_transactions = 5000
    
    invoice_numbers = [f'INV{str(i).zfill(6)}' for i in range(1, n_transactions+1)]
    stock_codes = [f'ITEM{np.random.randint(1000, 9999)}' for _ in range(n_transactions)]
    descriptions = [f'Product_{np.random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"])}_{np.random.randint(1, 100)}' for _ in range(n_transactions)]
    quantities = np.random.randint(1, 20, n_transactions)
    unit_prices = np.round(np.random.uniform(5, 200, n_transactions), 2)
    customer_ids = np.random.randint(10000, 10500, n_transactions)
    countries = np.random.choice(['UK', 'Germany', 'France', 'Spain', 'USA'], n_transactions)
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    invoice_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_transactions)]
    
    df = pd.DataFrame({
        'InvoiceNo': invoice_numbers,
        'StockCode': stock_codes,
        'Description': descriptions,
        'Quantity': quantities,
        'InvoiceDate': invoice_dates,
        'UnitPrice': unit_prices,
        'CustomerID': customer_ids,
        'Country': countries
    })
    
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # RFM Analysis
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)
    
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    def assign_segment(row):
        if row['Frequency'] > cluster_summary['Frequency'].mean() and row['Monetary'] > cluster_summary['Monetary'].mean():
            return 'High-Value'
        elif row['Recency'] > cluster_summary['Recency'].mean():
            return 'At-Risk'
        elif row['Frequency'] < cluster_summary['Frequency'].mean():
            return 'Occasional'
        else:
            return 'Regular'
    
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    cluster_to_segment = rfm.groupby('Cluster')['Segment'].agg(lambda x: x.mode()[0]).to_dict()
    
    # Recommendation System
    customer_product_matrix = df.groupby(['CustomerID', 'Description'])['Quantity'].sum().unstack(fill_value=0)
    item_similarity = cosine_similarity(customer_product_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, 
                                      index=customer_product_matrix.columns,
                                      columns=customer_product_matrix.columns)
    product_list = customer_product_matrix.columns.tolist()
    
    return scaler, kmeans_final, cluster_to_segment, item_similarity_df, product_list

# Load models
scaler, kmeans, cluster_labels, item_similarity, product_list = generate_models()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Project Info")
    st.markdown("**Shopper Spectrum** combines:")
    st.markdown("• RFM-based customer segmentation")
    st.markdown("• Item-based collaborative filtering")
    st.markdown("• Interactive analytics dashboard")
    
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("• Python, Pandas, NumPy")
    st.markdown("• Scikit-learn (KMeans, Cosine Similarity)")
    st.markdown("• Streamlit")
    
    st.markdown("### 💻 Full Project")
    st.markdown("Complete code with trained models available in:")
    st.markdown("• [Google Colab Notebook](https://colab.research.google.com/)")
    st.markdown("• [GitHub Repository](https://github.com/alwinappu/shopper-spectrum-segmentation)")
    
    st.markdown("👨‍💻 **Developer**: Alwin Appu")
    st.markdown("[GitHub](https://github.com/alwinappu) | [LinkedIn](https://linkedin.com/in/alwinappu)")

# Main content
st.title("🛍️ Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendations")
st.markdown("---")

tab1, tab2 = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation"])

with tab1:
    st.header("Product Recommendation System")
    st.write("Get personalized product recommendations based on item similarity")
    
    selected_product = st.selectbox("Select a product:", product_list)
    
    if st.button("Get Recommendations", key="recommend"):
        if selected_product:
            similar_products = item_similarity[selected_product].sort_values(ascending=False)[1:6]
            st.success(f"Top 5 products similar to **{selected_product}**:")
            for i, (product, score) in enumerate(similar_products.items(), 1):
                st.write(f"{i}. **{product}** (Similarity: {score:.3f})")
        else:
            st.warning("Please select a product")

with tab2:
    st.header("Customer Segmentation Predictor")
    st.write("Predict customer segment based on RFM values")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=500, value=5)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=50000.0, value=500.0)
    
    if st.button("Predict Segment", key="predict"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")
        
        st.success(f"### Customer Segment: **{segment}**")
        
        if segment == "High-Value":
            st.info("🌟 High-Value customers are frequent buyers with high spending. Focus on retention and loyalty programs.")
        elif segment == "Regular":
            st.info("✅ Regular customers make consistent purchases. Encourage them to increase frequency.")
        elif segment == "Occasional":
            st.info("📅 Occasional buyers purchase infrequently. Use targeted campaigns to increase engagement.")
        elif segment == "At-Risk":
            st.warning("⚠️ At-Risk customers haven't purchased recently. Implement win-back campaigns.")

st.markdown("---")
st.caption("❤️ Built with Streamlit | Shopper Spectrum Analytics")
