import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.metrics import classification_report
from my_enhanced_rf import SuperOptimizedRandomForest  # Your enhanced RF class

# Load your dataset
data = pd.read_csv('sms_data.csv')  # Ensure it has 'text' and 'label' columns

# Preprocessing
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data['text'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train enhanced RF
model = SuperOptimizedRandomForest()
model.fit(X_train, y_train)

# Save model + vectorizer
dump(model, 'model/sorf_model.pkl')
dump(tfidf, 'model/vectorizer.pkl')

# Optional evaluation
pred = model.predict(X_test)
print(classification_report(y_test, pred))
