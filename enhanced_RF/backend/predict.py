from joblib import load

# Load model and vectorizer
model = load('../model/sorf_model.pkl')
vectorizer = load('../model/vectorizer.pkl')

def predict_sms(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    
    # Get feature contributions
    contributions = model.get_feature_contributions(X)  # Assume you implemented this

    return {
        'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
        'contributions': contributions
    }
