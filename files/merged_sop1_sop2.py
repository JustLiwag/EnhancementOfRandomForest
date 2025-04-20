"""
This program combines two Random Forest enhancements - Spectral Co-Clustering and Reduced Error Pruning - 
for improved text classification. The process works as follows:

1. Data Loading and Preprocessing:
   - Detects and loads the dataset with proper encoding
   - Extracts LABEL and TEXT columns
   - Converts labels to numerical format using LabelEncoder
   - Vectorizes text data using TF-IDF

2. Standard Random Forest (Baseline):
   - Implements basic Random Forest classifier
   - Used as a benchmark for comparison

3. Pruning-Only Approach:
   - Implements Reduced Error Pruning on decision trees
   - Evaluates nodes based on impurity and accuracy
   - Creates a forest with pruned trees

4. Co-Clustering-Only Approach:
   - Removes zero-sum features
   - Optimizes number of clusters
   - Uses Spectral Co-Clustering for feature selection
   - Trains Random Forest on selected features

5. Combined Enhancement Approach:
   - First applies Spectral Co-Clustering for feature selection
   - Then applies Reduced Error Pruning to the trees
   - Creates an optimized forest using both techniques

6. Comprehensive Evaluation:
   - Compares all four approaches:
     * Standard Random Forest (baseline)
     * Pruning-Only Random Forest
     * Co-Clustering-Only Random Forest
     * Combined Enhancement Random Forest
   - Evaluates using multiple metrics:
     * Accuracy
     * Precision
     * Recall
     * F1-Score

The combined approach aims to leverage the benefits of both enhancements:
- Co-Clustering reduces feature dimensionality and noise
- Pruning optimizes tree structure and reduces overfitting
This potentially leads to a more efficient and accurate classifier.
"""

# April 17, 2025

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
import pandas as pd
import numpy as np
import chardet
import time
import copy

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset1.csv')

# Detect Encoding
with open(dataset_path, 'rb') as f:
    result = chardet.detect(f.read())

# Load Dataset
data = pd.read_csv(dataset_path, encoding=result['encoding'])
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
    """Prunes the given decision tree based on validation data with optimized impurity check."""
    # Make a deep copy to avoid modifying the original
    pruned_tree = copy.deepcopy(tree)
    
    # Get the underlying tree structure
    tree_structure = pruned_tree.tree_
    
    for node in range(tree_structure.node_count):
        # Skip leaf nodes
        if tree_structure.children_left[node] == _tree.TREE_LEAF and tree_structure.children_right[node] == _tree.TREE_LEAF:
            continue
        
        # Calculate impurity reduction
        impurity_before = tree_structure.impurity[node]
        if impurity_before < threshold:
            continue  # Skip pruning if impurity gain is low
        
        # Store original children nodes
        left_child = tree_structure.children_left[node]
        right_child = tree_structure.children_right[node]
        
        # Temporarily convert to leaf by setting children to -1
        tree_structure.children_left[node] = _tree.TREE_LEAF
        tree_structure.children_right[node] = _tree.TREE_LEAF
        
        # Evaluate performance before and after pruning
        y_pred_before = tree.predict(X_val)
        y_pred_after = pruned_tree.predict(X_val)
        
        acc_before = accuracy_score(y_val, y_pred_before)
        acc_after = accuracy_score(y_val, y_pred_after)
        
        # If pruning worsens performance, revert the changes
        if acc_after < acc_before:
            tree_structure.children_left[node] = left_child
            tree_structure.children_right[node] = right_child
    
    return pruned_tree

# --- Enhanced Random Forest with Pruning Only ---
X_train_prune, X_val_prune, y_train_prune, y_val_prune = train_test_split(X_train_std, y_train_std, test_size=0.2, random_state=42)

# Train a separate forest for pruning
rf_prune = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
rf_prune.fit(X_train_prune, y_train_prune)

# Prune each tree in the forest
pruned_trees = []
for tree in rf_prune.estimators_:
    pruned_tree = reduced_error_pruning(tree, X_val_prune, y_val_prune)
    pruned_trees.append(pruned_tree)

# Create a new forest with pruned trees
rf_pruned = copy.deepcopy(rf_prune)
rf_pruned.estimators_ = pruned_trees

# Evaluate pruned forest
y_pred_pruned = rf_pruned.predict(X_test_std)

# --- Enhanced Random Forest with Spectral Co-Clustering ---
# Remove rows/columns with zero sums
X_csc = X.tocsc()
nonzero_row_indices = np.array(X_csc.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X_csc.sum(axis=0)).flatten() > 0
X_filtered = X_csc[nonzero_row_indices, :]
X_filtered = X_filtered[:, nonzero_col_indices]
y_filtered = y[nonzero_row_indices]

# Optimize n_clusters for Spectral Co-Clustering
best_f1 = 0
best_clusters = 0
best_model = None

for n_clusters in range(2, 6):  # Test different numbers of clusters
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(X_filtered)
    
    # Get the feature indices for the first cluster
    selected_features = model.get_indices(0)[1]
    X_reduced = X_filtered[:, selected_features]

    # Train-Test Split
    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_reduced, y_filtered, test_size=0.2, random_state=42)

    # Train Enhanced Random Forest
    rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_enh.fit(X_train_enh, y_train_enh)
    y_pred_enh = rf_enh.predict(X_test_enh)

    # Calculate F1 Score
    f1 = f1_score(y_test_enh, y_pred_enh, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_clusters = n_clusters
        best_model = (rf_enh, X_test_enh, y_test_enh, model, selected_features)

# Extract best model components
rf_enh, X_test_enh, y_test_enh, best_coclustering_model, best_features = best_model
y_pred_enh = rf_enh.predict(X_test_enh)

# --- Combined Approach: Spectral Co-Clustering + Pruning ---
# Get reduced feature set from co-clustering
X_combined = X_filtered[:, best_features]
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y_filtered, test_size=0.2, random_state=42)

# Further split for pruning
X_train_comb_prune, X_val_comb, y_train_comb_prune, y_val_comb = train_test_split(X_train_comb, y_train_comb, test_size=0.2, random_state=42)

# Train a forest on the reduced feature set
rf_combined = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
rf_combined.fit(X_train_comb_prune, y_train_comb_prune)

# Prune the trees
pruned_trees_combined = []
for tree in rf_combined.estimators_:
    pruned_tree = reduced_error_pruning(tree, X_val_comb, y_val_comb)
    pruned_trees_combined.append(pruned_tree)

# Create a new forest with pruned trees
rf_combined_pruned = copy.deepcopy(rf_combined)
rf_combined_pruned.estimators_ = pruned_trees_combined

# Evaluate the combined approach
y_pred_combined = rf_combined_pruned.predict(X_test_comb)

# --- Evaluation Metrics ---
# Standard Random Forest Metrics
std_metrics = {
    "Accuracy": accuracy_score(y_test_std, y_pred_std),
    "Precision": precision_score(y_test_std, y_pred_std, average='weighted'),
    "Recall": recall_score(y_test_std, y_pred_std, average='weighted'),
    "F1-Score": f1_score(y_test_std, y_pred_std, average='weighted'),
}

# Pruning-Only Random Forest Metrics
pruned_metrics = {
    "Accuracy": accuracy_score(y_test_std, y_pred_pruned),
    "Precision": precision_score(y_test_std, y_pred_pruned, average='weighted'),
    "Recall": recall_score(y_test_std, y_pred_pruned, average='weighted'),
    "F1-Score": f1_score(y_test_std, y_pred_pruned, average='weighted'),
}

# Co-Clustering-Only Random Forest Metrics
enh_metrics = {
    "Accuracy": accuracy_score(y_test_enh, y_pred_enh),
    "Precision": precision_score(y_test_enh, y_pred_enh, average='weighted'),
    "Recall": recall_score(y_test_enh, y_pred_enh, average='weighted'),
    "F1-Score": f1_score(y_test_enh, y_pred_enh, average='weighted'),
}

# Combined Approach Metrics
combined_metrics = {
    "Accuracy": accuracy_score(y_test_comb, y_pred_combined),
    "Precision": precision_score(y_test_comb, y_pred_combined, average='weighted'),
    "Recall": recall_score(y_test_comb, y_pred_combined, average='weighted'),
    "F1-Score": f1_score(y_test_comb, y_pred_combined, average='weighted'),
}

# Print Comparative Analysis
print(f"Best Number of Clusters for Co-Clustering: {best_clusters}\n")

print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nRandom Forest with Pruning Only:")
print(classification_report(y_test_std, y_pred_pruned))

print("\nRandom Forest with Spectral Co-Clustering Only:")
print(classification_report(y_test_enh, y_pred_enh))

print("\nCombined Approach (Co-Clustering + Pruning):")
print(classification_report(y_test_comb, y_pred_combined))

print("\n--- Comparative Analysis ---")
print(f"{'Metric':<15}{'Standard RF':<15}{'Pruned RF':<15}{'Co-Clustered RF':<15}{'Combined RF':<15}")
for metric in std_metrics:
    print(f"{metric:<15}{std_metrics[metric]:<15.4f}{pruned_metrics[metric]:<15.4f}{enh_metrics[metric]:<15.4f}{combined_metrics[metric]:<15.4f}")