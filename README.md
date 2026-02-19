# Tourism_labmentix
A Streamlit-based tourism recommendation system using collaborative filtering (SVD) to suggest personalized tourist attractions from user rating data.
# 🌍 Tourism Recommendation System (Streamlit App)

This project implements a **Tourism Experience Analytics system** that provides **personalized attraction recommendations** using **Collaborative Filtering (SVD)**.  
The application is built using **Python, Scikit-learn, and Streamlit**, and is executed locally using **VS Code**.

---

## 📌 Project Overview

Tourism platforms aim to enhance user experience by analyzing historical travel data and user preferences.  
This project focuses on building a **recommendation system** that suggests tourist attractions based on past user ratings and interactions.

The system uses **matrix factorization (Singular Value Decomposition – SVD)** to uncover latent user and attraction features and generate personalized recommendations.

---

## 🎯 Objectives

- Build a **user–attraction recommendation system**
- Apply **Collaborative Filtering using SVD**
- Develop an **interactive Streamlit application**
- Provide a stable and interpretable ML solution suitable for academic evaluation

---

## 🧠 Methodology

### 1. Data Preparation
- Loaded tourism transaction and attraction datasets from Excel files
- Created a **User–Item Rating Matrix**
- Handled missing values by filling unrated interactions with zeros

### 2. Recommendation Model
- Applied **Truncated Singular Value Decomposition (SVD)**
- Reduced dimensionality to capture latent patterns
- Reconstructed predicted preference scores using matrix multiplication

### 3. Recommendation Logic
- Excluded attractions already visited by the user
- Generated **Top-N recommendations** based on predicted scores

### 4. Streamlit Application
- Built a local Streamlit app for user interaction
- Model training is triggered via a button to ensure application stability
- Recommendations are displayed in tabular format

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- VS Code

---

## 📂 Project Structure

