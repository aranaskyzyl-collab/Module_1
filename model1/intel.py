# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=("AIzaSyB9iaO4DNzzHvHc6v_xXE00fmUozbn8sSw"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Helper function to get dataset path
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "fast-prototyping-of-genai-apps-with-streamlit", "data", "customer_reviews.csv")
    return csv_path

# Function to get sentiment using Gemini
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        prompt = f"""
        Classify the sentiment of the following review as exactly one word: Positive, Negative, or Neutral.
        
        Review: {text}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"

# Streamlit UI
# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Hi I am William Aranas</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Labad jud ako ulo ani Ma'am</h2>", unsafe_allow_html=True)


# Layout two buttons side by side
col1, col2 = st.columns(2)

# Custom button styles
button_css = """
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        color: white;
        background: linear-gradient(90deg, #4B0082, #6A5ACD);
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #6A5ACD, #4B0082);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    </style>
"""
st.markdown(button_css, unsafe_allow_html=True)

# Left column - Load Dataset
with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(10)
            st.success("✅ Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Dataset not found. Please check the file path.")

# Right column - Analyze Sentiment
with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"].loc[:, "Sentiment"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
                    st.success("✅ Sentiment analysis completed!")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
        else:
            st.warning("⚠️ Please ingest the dataset first.")

# Display dataset
if "df" in st.session_state:
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    # Visualization
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': 'red', 'Neutral': 'lightgray', 'Positive': 'green'}

        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in existing_sentiments if s in sentiment_colors}

        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=filtered_order, ordered=True)
        sentiment_counts = sentiment_counts.sort_values('Sentiment')

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {product}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
