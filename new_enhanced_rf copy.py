from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import chardet 

# Detect encoding for training data
with open('spam.csv', 'rb') as f:
    result_train = chardet.detect(f.read())

# Detect encoding for testing data
with open('spamm.csv', 'rb') as f:
    result_test = chardet.detect(f.read())

# Load Training Dataset
train_data = pd.read_csv('spam.csv', encoding=result_train['encoding'])
train_data = train_data[['LABEL', 'TEXT']]
train_data.columns = ['LABEL', 'TEXT']

# Load Testing Dataset
test_data = pd.read_csv('spamm.csv', encoding=result_test['encoding'])
test_data = test_data[['LABEL', 'TEXT']]
test_data.columns = ['LABEL', 'TEXT']

# Preprocess Labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['LABEL'] = le.fit_transform(train_data['LABEL'])
test_data['LABEL'] = le.transform(test_data['LABEL'])

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['TEXT'])
y_train = train_data['LABEL']
X_test = vectorizer.transform(test_data['TEXT'])
y_test = test_data['LABEL']

# --- Standard Random Forest ---
rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_std.fit(X_train, y_train)
y_pred_std = rf_std.predict(X_test)

# --- Enhanced Random Forest with Spectral Co-Clustering ---
# Remove rows/columns with zero sums (as in your original code)
X_train = X_train.tocsc()
nonzero_row_indices_train = np.array(X_train.sum(axis=1)).flatten() > 0
nonzero_col_indices_train = np.array(X_train.sum(axis=0)).flatten() > 0
X_train = X_train[nonzero_row_indices_train, :]
X_train = X_train[:, nonzero_col_indices_train]
y_train = y_train[nonzero_row_indices_train]  # Apply filtering to y as well

# Apply Spectral Co-Clustering
model = SpectralCoclustering(n_clusters=5, random_state=42)
model.fit(X_train)
selected_features = model.get_indices(0)[1]
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Train Enhanced Random Forest
rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_enh.fit(X_train_reduced, y_train)
y_pred_enh = rf_enh.predict(X_test_reduced)

# --- Evaluation ---
print("Standard Random Forest:")
print(classification_report(y_test, y_pred_std))

print("\nEnhanced Random Forest with Spectral Co-Clustering:")
print(classification_report(y_test, y_pred_enh))