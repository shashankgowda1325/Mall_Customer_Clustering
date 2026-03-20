import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Mall Customer Clustering App")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.write(data.head())
  
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

    k = st.slider("Select number of clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
  
    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='rainbow')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               s=200, c='black', label='Centroids')
    
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.legend()

    st.pyplot(fig)

    st.success("Clustering Completed!")
