import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛍️", layout="wide")

# Title
st.title("🛍️ Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendations")
st.markdown("---")

# Info message
st.info("🚧 **Demo Mode**: This is a demonstration version. Full functionality with trained models available in the Colab notebook.")

# Create tabs
tab1, tab2 = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation"])

# Tab 1: Product Recommendation (Demo)
with tab1:
    st.header("Product Recommendation System")
    st.write("Get personalized product recommendations based on item similarity")
    
    # Sample products for demo
    demo_products = [
        "Product_Electronics_42",
        "Product_Clothing_18",
        "Product_Books_25",
        "Product_Home_67",
        "Product_Sports_33"
    ]
    
    selected_product = st.selectbox("Select a product:", demo_products)
    
    if st.button("Get Recommendations", key="recommend"):
        st.success(f"Top 5 products similar to **{selected_product}**:")
        
        # Demo recommendations
        st.write("1. **Product_Electronics_45** (Similarity: 0.892)")
        st.write("2. **Product_Electronics_31** (Similarity: 0.856)")
        st.write("3. **Product_Home_12** (Similarity: 0.834)")
        st.write("4. **Product_Electronics_78** (Similarity: 0.812)")
        st.write("5. **Product_Sports_56** (Similarity: 0.789)")
        
        st.info("💡 These are demo recommendations. Upload model files for real predictions.")

# Tab 2: Customer Segmentation (Demo)
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
        # Simple rule-based demo logic
        if frequency > 10 and monetary > 1000:
            segment = "High-Value"
            description = "🌟 High-Value customers are frequent buyers with high spending. Focus on retention and loyalty programs."
            st.success(f"### Customer Segment: **{segment}**")
            st.info(description)
        elif recency > 180:
            segment = "At-Risk"
            description = "⚠️ At-Risk customers haven't purchased recently. Implement win-back campaigns."
            st.success(f"### Customer Segment: **{segment}**")
            st.warning(description)
        elif frequency < 5:
            segment = "Occasional"
            description = "📅 Occasional buyers purchase infrequently. Use targeted campaigns to increase engagement."
            st.success(f"### Customer Segment: **{segment}**")
            st.info(description)
        else:
            segment = "Regular"
            description = "✅ Regular customers make consistent purchases. Encourage them to increase frequency."
            st.success(f"### Customer Segment: **{segment}**")
            st.info(description)
        
        st.info("💡 This is a demo prediction. Upload model files for ML-based segmentation.")

st.markdown("---")
st.caption("❤️ Built with Streamlit | Shopper Spectrum Analytics")

# Sidebar info
with st.sidebar:
    st.header("📊 Project Info")
    st.markdown("""
    **Shopper Spectrum** combines:
    - RFM-based customer segmentation
    - Item-based collaborative filtering
    - Interactive analytics dashboard
    
    ### 🛠️ Tech Stack
    - Python, Pandas, NumPy
    - Scikit-learn (KMeans, Cosine Similarity)
    - Streamlit
    
    ### 💻 Full Project
    Complete code with trained models available in:
    - [Google Colab Notebook](https://colab.research.google.com)
    - [GitHub Repository](https://github.com/alwinappu/shopper-spectrum-segmentation)
    """)
    
    st.markdown("---")
    st.markdown("👨‍💻 **Developer**: Alwin Appu")
    st.markdown("[GitHub](https://github.com/alwinappu) | [LinkedIn](https://linkedin.com/in/alwinappu)")
