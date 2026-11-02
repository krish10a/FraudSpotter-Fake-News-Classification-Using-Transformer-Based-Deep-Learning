"""
FraudSpotter - Streamlit Web Interface (Placeholder Scaffold)
Connects to the FastAPI backend or local ONNX runtime for interactive explainable classification.
"""

import requests
import streamlit as st

st.set_page_config(
    page_title="FraudSpotter | Fake News Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudSpotter: Explainable Fake News Detection")
st.markdown(
    "Welcome to **FraudSpotter**. This interface allows you to analyze news articles, "
    "view predicted veracity (Fake vs. Real), confidence calibration, and token-level explanations."
)

api_url = st.sidebar.text_input("Backend API URL", value="http://localhost:8000/api/v1")

title_input = st.text_input("Article Title", placeholder="Enter article headline...")
text_input = st.text_area(
    "Article Text Content",
    placeholder="Paste news body text here...",
    height=250,
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Analyze Veracity", type="primary"):
        if not text_input.strip():
            st.warning("Please provide news article content to analyze.")
        else:
            with st.spinner("Analyzing text and computing explainability attributions..."):
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"title": title_input, "text": text_input},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Prediction: {data['label']}")
                        st.progress(data["confidence"])
                        st.write(f"**Confidence:** {data['confidence'] * 100:.2f}%")
                        st.write(f"**Model:** {data['model_used']}")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to FraudSpotter backend: {e}")

with col2:
    st.subheader("Explanation & Attribution")
    st.info("Token attribution and explainability visualizations will be displayed here.")
