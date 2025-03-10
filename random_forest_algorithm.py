from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import chardet

# Detect Encoding
with open('dataset1.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Load Dataset
data = pd.read_csv('dataset1.csv', encoding=result['encoding'])
data = data[['LABEL', 'TEXT']]
data.columns = ['LABEL', 'TEXT']

# Preprocess Labels
le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])

# Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']

# --- Standard Random Forest ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# --- Evaluation ---
print("Standard Random Forest:")
print(classification_report(y_test, y_pred))