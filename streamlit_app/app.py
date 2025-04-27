import streamlit as st
import requests

st.title("Sentiment Analysis App (Flask + Streamlit)")
st.write("Enter one or more reviews (one per line):")

# Add model selection option
model_type = st.radio("Select Model Type:", ["Traditional ML", "LSTM Neural Network"])

user_input = st.text_area("Your review(s)", height=200)

if st.button("Predict Sentiment"):
    reviews = [line for line in user_input.split('\n') if line.strip()]
    if reviews:
        # Set endpoint based on model selection
        endpoint = "http://localhost:5000/predict" if model_type == "Traditional ML" else "http://localhost:5000/predict_lstm"
        
        # Send request to appropriate endpoint
        response = requests.post(
            endpoint,
            json={"texts": reviews}
        )
        
        if response.status_code == 200:
            preds = response.json()["predictions"]
            
            # Display model type being used
            st.subheader(f"Results using {model_type} model:")
            
            # Check if confidence scores available (LSTM only)
            if "confidence" in response.json() and model_type == "LSTM Neural Network":
                confidence_scores = response.json()["confidence"]
                for review, pred, conf in zip(reviews, preds, confidence_scores):
                    st.write(f"**Review:** {review}")
                    st.write(f"**Predicted Sentiment:** :blue[{pred}]")
                    st.write(f"**Confidence:** {conf:.2f}")
                    st.write("---")
            else:
                # Regular display for traditional model
                for review, pred in zip(reviews, preds):
                    st.write(f"**Review:** {review}")
                    st.write(f"**Predicted Sentiment:** :blue[{pred}]")
                    st.write("---")
        else:
            st.error(f"Error from Flask API: {response.text}")
    else:
        st.warning("Please enter at least one review.")

# Add explanation section
with st.expander("About the models"):
    st.write("""
    - **Traditional ML**: Uses TF-IDF vectorization with Logistic Regression for classification.
    - **LSTM Neural Network**: Uses deep learning to capture sequential patterns in text, potentially better at understanding context.
    """)
