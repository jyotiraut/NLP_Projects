import streamlit as st
import requests

st.title("Sentiment Analysis App (Flask + Streamlit)")
st.write("Enter one or more reviews (one per line):")

user_input = st.text_area("Your review(s)", height=200)

if st.button("Predict Sentiment"):
    reviews = [line for line in user_input.split('\n') if line.strip()]
    if reviews:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"texts": reviews}
        )
        if response.status_code == 200:
            preds = response.json()["predictions"]
            for review, pred in zip(reviews, preds):
                st.write(f"**Review:** {review}")
                st.write(f"**Predicted Sentiment:** :blue[{pred}]")
                st.write("---")
        else:
            st.error("Error from Flask API: " + response.text)
    else:
        st.warning("Please enter at least one review.")
