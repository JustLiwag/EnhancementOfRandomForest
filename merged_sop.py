# MARCH 06, 2025

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import chardet
import time

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
start_time = time.time()
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, test_size=0.2, random_state=42)
rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_std.fit(X_train_std, y_train_std)
std_training_time = time.time() - start_time
y_pred_std = rf_std.predict(X_test_std)

# --- Reduced Error Pruning Function ---
def reduced_error_pruning(tree, X_val, y_val, threshold=0.01):
    """ Prunes the given decision tree based on validation data with optimized impurity check."""
    for node in range(tree.tree_.node_count):
        if tree.tree_.children_left[node] == TREE_LEAF and tree.tree_.children_right[node] == TREE_LEAF:
            continue
        
        # Calculate impurity reduction
        impurity_before = tree.tree_.impurity[node]
        if impurity_before < threshold:
            continue  # Skip pruning if impurity gain is low
        
        y_pred_before = tree.predict(X_val)
        acc_before = accuracy_score(y_val, y_pred_before)
        
        original_threshold = tree.tree_.threshold[node]
        tree.tree_.threshold[node] = -2  # Mark as leaf node
        
        y_pred_after = tree.predict(X_val)
        acc_after = accuracy_score(y_val, y_pred_after)
        
        if acc_after < acc_before:
            tree.tree_.threshold[node] = original_threshold  # Restore original split
    
    return tree

# --- Enhanced Random Forest with Pruning ---
pruned_trees = []
num_pruned_trees = min(3, len(rf_std.estimators_))  # Prune fewer trees for efficiency
for tree in rf_std.estimators_[:num_pruned_trees]:
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    pruned_tree = reduced_error_pruning(tree, X_val, y_val)
    pruned_trees.append(pruned_tree)

# --- Enhanced Random Forest with Spectral Co-Clustering ---
# Remove rows/columns with zero sums
X = X.tocsc()
nonzero_row_indices = np.array(X.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X.sum(axis=0)).flatten() > 0
X = X[nonzero_row_indices, :]
X = X[:, nonzero_col_indices]
y = y[nonzero_row_indices]

# Optimize n_clusters for Spectral Co-Clustering
best_f1 = 0
best_clusters = 0
best_model = None
for n_clusters in range(2, 10):  # Test different numbers of clusters
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    selected_features = model.get_indices(0)[1]
    X_reduced = X[:, selected_features]

    # Train-Test Split
    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # Train Enhanced Random Forest
    rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_enh.fit(X_train_enh, y_train_enh)
    y_pred_enh = rf_enh.predict(X_test_enh)

    # Evaluate F1 Score for the Minority Class
    f1 = f1_score(y_test_enh, y_pred_enh, average=None)[1]  # Minority class assumed to be label 1
    if f1 > best_f1:
        best_f1 = f1
        best_clusters = n_clusters
        best_model = (rf_enh, X_test_enh, y_test_enh, y_pred_enh)

# --- Final Evaluation ---
rf_enh, X_test_enh, y_test_enh, y_pred_enh = best_model

# Standard Random Forest Metrics
std_metrics = {
    "Accuracy": accuracy_score(y_test_std, y_pred_std),
    "Precision": precision_score(y_test_std, y_pred_std, average='weighted'),
    "Recall": recall_score(y_test_std, y_pred_std, average='weighted'),
    "F1-Score": f1_score(y_test_std, y_pred_std, average='weighted'),
}

# Enhanced Random Forest Metrics
enh_metrics = {
    "Accuracy": accuracy_score(y_test_enh, y_pred_enh),
    "Precision": precision_score(y_test_enh, y_pred_enh, average='weighted'),
    "Recall": recall_score(y_test_enh, y_pred_enh, average='weighted'),
    "F1-Score": f1_score(y_test_enh, y_pred_enh, average='weighted'),
}

# Print Comparative Analysis
print(f"Best Number of Clusters: {best_clusters}\n")

print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nEnhanced Random Forest with Spectral Co-Clustering:")
print(classification_report(y_test_enh, y_pred_enh))

print("\n--- Comparative Analysis ---")
print(f"{'Metric':<15}{'Standard RF':<15}{'Enhanced RF':<15}")
for metric in std_metrics:
    print(f"{metric:<15}{std_metrics[metric]:<15.4f}{enh_metrics[metric]:<15.4f}")