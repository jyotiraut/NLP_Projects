from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load traditional model and vectorizer
model = joblib.load('../models/sentiment_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Load LSTM model and tokenizer
lstm_model = load_model('../models/lstm_sentiment_model.h5')
tokenizer = joblib.load('../models/lstm_tokenizer.pkl')
max_length = 100  

app = Flask(__name__)

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
    pred_labels = [sentiment_map[int(p)] for p in preds]
    
    return jsonify({'predictions': pred_labels})

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    data = request.get_json()
    texts = data.get('texts', [])
    
    # LSTM preprocessing - convert to sequences and pad
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    # Make predictions with LSTM model
    predictions = lstm_model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1).tolist()
    
    # Use same sentiment mapping for consistency
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    pred_labels = [sentiment_map[int(p)] for p in predicted_classes]
    
    # Include confidence scores
    confidence_scores = [float(predictions[i][pred]) for i, pred in enumerate(predicted_classes)]
    
    return jsonify({
        'predictions': pred_labels,
        'confidence': confidence_scores,
        'model_type': 'LSTM'
    })

if __name__ == '__main__':
    app.run(debug=True)
