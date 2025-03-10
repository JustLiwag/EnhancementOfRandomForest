from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['LABEL', 'TEXT']]
data.columns = ['LABEL', 'TEXT']

# Preprocess Labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])

# Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']

# --- Standard Random Forest ---
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, test_size=0.2, random_state=42)
rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_std.fit(X_train_std, y_train_std)
y_pred_std = rf_std.predict(X_test_std)

# --- Enhanced Random Forest with Spectral Co-Clustering ---
# Remove rows/columns with zero sums (as in your original code)
X = X.tocsc()
nonzero_row_indices = np.array(X.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X.sum(axis=0)).flatten() > 0
X = X[nonzero_row_indices, :]
X = X[:, nonzero_col_indices]
y = y[nonzero_row_indices]  # Apply filtering to y as well

# Apply Spectral Co-Clustering
model = SpectralCoclustering(n_clusters=5, random_state=42)
model.fit(X)
selected_features = model.get_indices(0)[1]
X_reduced = X[:, selected_features]

# Train-Test Split
X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train Enhanced Random Forest
rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_enh.fit(X_train_enh, y_train_enh)
y_pred_enh = rf_enh.predict(X_test_enh)

# --- Evaluation ---
print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nEnhanced Random Forest with Spectral Co-Clustering:")
print(classification_report(y_test_enh, y_pred_enh))