from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load('../models/sentiment_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('texts', [])
    
    # Transform input texts
    X = vectorizer.transform(texts)
    
    # Make predictions
    preds = model.predict(X)
    
    # Correct mapping: 0=Negative, 1=Neutral, 2=Positive
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    pred_labels = [sentiment_map[p] for p in preds]
    
    return jsonify({'predictions': pred_labels})


if __name__ == '__main__':
    app.run(debug=True)
