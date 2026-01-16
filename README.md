# 🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendations

## 🎯 Project Overview

A comprehensive e-commerce analytics solution that combines **RFM-based customer segmentation** with **item-based collaborative filtering** to provide actionable insights and personalized product recommendations.

## ✨ Features

### 1. Customer Segmentation
- **RFM Analysis**: Classify customers based on Recency, Frequency, and Monetary values
- **KMeans Clustering**: Automated segmentation into 4 groups:
  - 🌟 **High-Value**: Frequent buyers with high spending
  - ✅ **Regular**: Consistent purchase patterns  
  - 📅 **Occasional**: Infrequent buyers
  - ⚠️ **At-Risk**: Customers who haven't purchased recently

### 2. Product Recommendations
- **Collaborative Filtering**: Item-based similarity using cosine distance
- Get top 5 similar products for any selected item
- Real-time recommendations through interactive interface

### 3. Interactive Dashboard
- Built with **Streamlit** for seamless user experience
- Two main modules:
  - Product Recommendation System
  - Customer Segmentation Predictor

## 🛠️ Tech Stack

- **Python 3.9+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (KMeans, StandardScaler, Cosine Similarity)
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Community Cloud

## 📊 Methodology

### RFM Feature Engineering
```python
Recency = Days since last purchase
Frequency = Number of unique transactions
Monetary = Total spend amount
```

### Model Training
1. Data preprocessing and cleaning
2. RFM computation for each customer
3. Feature standardization using StandardScaler
4. KMeans clustering (k=4) for customer segmentation
5. Customer-Product matrix construction
6. Cosine similarity computation for item-based CF

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/alwinappu/shopper-spectrum-segmentation.git
cd shopper-spectrum-segmentation
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

## 📁 Project Structure

```
shopper-spectrum-segmentation/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── rfm_scaler.pkl           # Trained StandardScaler
├── rfm_kmeans.pkl           # Trained KMeans model
├── cluster_labels.pkl       # Cluster-to-segment mapping
├── item_similarity.pkl      # Item similarity matrix
└── product_list.pkl         # Product catalog
```

## 💡 Usage

### Product Recommendations
1. Navigate to "📦 Product Recommendation" tab
2. Select a product from the dropdown
3. Click "Get Recommendations"
4. View top 5 similar products with similarity scores

### Customer Segmentation
1. Navigate to "👥 Customer Segmentation" tab
2. Input RFM values:
   - Recency (days)
   - Frequency (number of purchases)
   - Monetary (total spend)
3. Click "Predict Segment"
4. View predicted customer segment with actionable insights

## 📈 Model Performance

- **Clustering**: Silhouette Score optimized for k=4 clusters
- **Recommendation**: Cosine similarity-based collaborative filtering
- **Scalability**: Handles 500+ customers and 495+ unique products

## 🎓 Key Learnings

- RFM analysis for customer behavior understanding
- Unsupervised learning with KMeans clustering
- Collaborative filtering for recommendation systems
- End-to-end ML deployment with Streamlit

## 📝 Future Enhancements

- [ ] Add time-series forecasting for customer lifetime value
- [ ] Implement hybrid recommendation (content + collaborative)
- [ ] A/B testing framework for model evaluation
- [ ] Real-time data pipeline integration
- [ ] Advanced visualizations with Plotly

## 👨‍💻 Author

**Alwin Appu**  
[GitHub](https://github.com/alwinappu) | [LinkedIn](https://linkedin.com/in/alwinappu)

## 📄 License

MIT License - feel free to use this project for learning and development!

## 🙏 Acknowledgments

- Dataset: Synthetic e-commerce transactions (2022-2023)
- Inspired by real-world retail analytics challenges
- Built as part of ML portfolio projects

---

⭐ **Star this repo** if you found it helpful!

🔗 **Live Demo**: Coming soon on Streamlit Community Cloud
