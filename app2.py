import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Tourism Recommendation System",
    layout="centered"
)

st.title("🌍 Tourism Recommendation System")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    transactions = pd.read_excel("data/Transaction.xlsx")
    items = pd.read_excel("data/Item.xlsx")
    return transactions, items

transactions, items = load_data()
st.success("Data loaded successfully")

# -------------------------------------------------
# Create User–Item Matrix
# -------------------------------------------------
user_item_matrix = transactions.pivot_table(
    index="UserId",
    columns="AttractionId",
    values="Rating"
).fillna(0)

st.write("User–Item Matrix Shape:", user_item_matrix.shape)

# -------------------------------------------------
# Apply SVD (Cached as Resource)
# -------------------------------------------------
@st.cache_resource
def apply_svd(user_item_matrix, n_components=10):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(user_item_matrix)
    item_latent = svd.components_.T

    predicted_ratings = np.dot(user_latent, item_latent.T)

    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    return predicted_df

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommend_attractions(user_id, user_item_matrix, predicted_df, top_n=5):

    if user_id not in user_item_matrix.index:
        return None

    visited = user_item_matrix.loc[user_id]
    visited = visited[visited > 0].index.tolist()

    scores = predicted_df.loc[user_id]
    scores = scores.drop(index=visited, errors="ignore")

    return scores.sort_values(ascending=False).head(top_n)

# -------------------------------------------------
# Model Training Section
# -------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Train Recommendation Model")

if "predicted_df" not in st.session_state:
    st.info("Model not trained yet. Click the button below.")

if st.button("Train SVD Model", key="train_button"):
    with st.spinner("Training model, please wait..."):
        st.session_state["predicted_df"] = apply_svd(user_item_matrix)
    st.success("Model trained successfully!")

# -------------------------------------------------
# Recommendation UI
# -------------------------------------------------
# =====================================
# 🎯 RECOMMENDATION UI SECTION
# =====================================
st.markdown("---")
st.subheader("🎯 Get Recommendations")

# SINGLE number_input (ONLY ONE in whole app)
user_id = st.number_input(
    label="Enter User ID",
    min_value=int(user_item_matrix.index.min()),
    max_value=int(user_item_matrix.index.max()),
    step=1,
    key="user_id_recommendation"
)

# Recommendation button
if st.button("Recommend Attractions", key="recommend_button"):
    
    if "predicted_df" not in st.session_state:
        st.warning("⚠️ Please train the SVD model first.")
    
    else:
        predicted_df = st.session_state["predicted_df"]

        recommendations = recommend_attractions(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            predicted_df=predicted_df,
            top_n=5
        )

        if recommendations is None or recommendations.empty:
            st.warning("No recommendations available for this user.")
        
        else:
            rec_df = recommendations.reset_index()
            rec_df.columns = ["AttractionId", "Predicted Score"]

            rec_df = rec_df.merge(
                items[["AttractionId", "Attraction"]],
                on="AttractionId",
                how="left"
            )

            st.success("Top Recommended Attractions")
            st.dataframe(rec_df)
