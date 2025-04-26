# Sentiment Analysis Case Study: Amazon Reviews

## Project Overview

This case study explores the development of a sentiment analysis system for Amazon product reviews. The project follows a complete machine learning workflow from data preprocessing to model deployment, with a focus on evaluating different classification algorithms to determine the most effective approach for sentiment prediction.

## Dataset

The dataset consists of Amazon product reviews with the following key fields:
- ProductId: Unique identifier for products
- Score/Rating: Customer rating (1-5 stars)
- Summary: Brief review headline
- Text: Full review content

We categorized sentiment into three classes:
- Negative: Ratings 1-2
- Neutral: Rating 3
- Positive: Ratings 4-5

## Project Structure

```
project/
├── data/
│   ├── clean/             # Preprocessed data
│   ├── features/          # Feature engineering outputs
│   └── raw/               # Original data
├── models/                # Saved trained models
├── notebook/
│   ├── 01_Preprocessing/  # Data cleaning and preprocessing
│   ├── 02_Feature_Engineering/ # Feature extraction
│   └── 03_Model_Development/ # Model training and evaluation
└── flask_api/            # Model deployment API
```

## Methodology

### 1. Data Preprocessing

- **Text Cleaning**: Removed special characters and punctuation
- **Normalization**: Converted text to lowercase
- **Tokenization**: Split text into individual words
- **Stopword Removal**: Eliminated common non-informative words
- **Stemming**: Applied Porter stemming algorithm to reduce words to their root form

### 2. Feature Engineering

- **Text Representation**: Used TF-IDF vectorization with 5,000 features
- **Data Preparation**: Split data into 80% training and 20% testing sets
- **Vector Creation**: Converted text to numerical feature vectors

### 3. Model Development and Evaluation

We implemented and compared multiple classification algorithms:

| Model | Accuracy | F1-Score | Training Time (s) | Advantages | Limitations |
|-------|----------|----------|-------------------|------------|-------------|
| Logistic Regression | 0.892 | 0.890 | 45.67 | Best balance of accuracy and efficiency | Limited at capturing complex patterns |
| Random Forest | 0.871 | 0.869 | 215.78 | Good feature importance insights | Slow training time |
| Naive Bayes | 0.845 | 0.843 | 3.12 | Extremely fast training | Lower accuracy |

**Hyperparameter Tuning Results:**
- **Logistic Regression**: C=1, solver='lbfgs'
- **Random Forest**: max_depth=20, min_samples_split=2, n_estimators=100
- **Naive Bayes**: alpha=0.5

**Classification Report:**
```
              precision    recall  f1-score   support
    Negative       0.78      0.23      0.35       160
     Neutral       0.50      0.01      0.03        71
    Positive       0.80      0.99      0.89       769
```

### 4. Model Interpretation

Analysis of the most influential words for sentiment classification:

**Top Positive Indicators**: great, love, good, best, delicious, excellent, perfect  
**Top Negative Indicators**: waste, terrible, worst, return, disappointing, bad, awful

### 5. Deployment

Created a web application with:
- **Backend**: Flask API for model serving
- **Frontend**: Streamlit for user interface
- **Features**: Real-time sentiment prediction for user input

## Challenges and Limitations

1. **Class Imbalance**: The dataset had significantly more positive reviews (77%) than negative (16%) or neutral (7%)
2. **Neutral Class Detection**: Models struggled to correctly identify neutral sentiment
3. **Stemming Readability**: The stemming process affected word interpretability

## Conclusions

1. **Best Model**: Logistic Regression provided the optimal balance between accuracy (89.2%) and computational efficiency
2. **Feature Importance**: Lexical features effectively captured sentiment patterns
3. **Future Work**: Addressing class imbalance and improving neutral class detection

This case study demonstrates the effectiveness of machine learning approaches for sentiment analysis while highlighting the importance of appropriate preprocessing, feature engineering, and model selection in achieving robust performance.

