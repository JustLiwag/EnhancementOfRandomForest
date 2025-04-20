import numpy as np
import pandas as pd
import time
import copy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
import chardet
import seaborn as sns
import matplotlib.pyplot as plt

# Mount and load dataset
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/THESIS_DATASET')

with open('dataset1.csv', 'rb') as f:
    result = chardet.detect(f.read())
data = pd.read_csv('dataset1.csv', encoding=result['encoding'])
data = data[['LABEL', 'TEXT']]
data.columns = ['LABEL', 'TEXT']

# Preprocess
le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])

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
    pruned_tree = copy.deepcopy(tree)
    tree_structure = pruned_tree.tree_
    for node in range(tree_structure.node_count):
        if tree_structure.children_left[node] == _tree.TREE_LEAF and tree_structure.children_right[node] == _tree.TREE_LEAF:
            continue
        impurity_before = tree_structure.impurity[node]
        if impurity_before < threshold:
            continue
        left_child = tree_structure.children_left[node]
        right_child = tree_structure.children_right[node]
        tree_structure.children_left[node] = _tree.TREE_LEAF
        tree_structure.children_right[node] = _tree.TREE_LEAF
        y_pred_before = tree.predict(X_val)
        y_pred_after = pruned_tree.predict(X_val)
        acc_before = accuracy_score(y_val, y_pred_before)
        acc_after = accuracy_score(y_val, y_pred_after)
        if acc_after < acc_before:
            tree_structure.children_left[node] = left_child
            tree_structure.children_right[node] = right_child
    return pruned_tree

# --- Combined Approach: Spectral Co-Clustering + Pruning ---
X_csc = X.tocsc()
nonzero_row_indices = np.array(X_csc.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X_csc.sum(axis=0)).flatten() > 0
X_filtered = X_csc[nonzero_row_indices, :]
X_filtered = X_filtered[:, nonzero_col_indices]
y_filtered = y[nonzero_row_indices]

best_f1 = 0
best_model = None
for n_clusters in range(2, 6):
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(X_filtered)
    selected_features = model.get_indices(0)[1]
    X_reduced = X_filtered[:, selected_features]
    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_reduced, y_filtered, test_size=0.2, random_state=42)
    rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_enh.fit(X_train_enh, y_train_enh)
    y_pred_enh = rf_enh.predict(X_test_enh)
    f1 = f1_score(y_test_enh, y_pred_enh, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_clusters = n_clusters
        best_model = (selected_features, X_filtered, y_filtered)

# Train final Combined RF with pruning
selected_features, X_filtered, y_filtered = best_model
X_combined = X_filtered[:, selected_features]
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y_filtered, test_size=0.2, random_state=42)
X_train_comb_prune, X_val_comb, y_train_comb_prune, y_val_comb = train_test_split(X_train_comb, y_train_comb, test_size=0.2, random_state=42)

rf_combined = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
rf_combined.fit(X_train_comb_prune, y_train_comb_prune)

pruned_trees_combined = []
for tree in rf_combined.estimators_:
    pruned_tree = reduced_error_pruning(tree, X_val_comb, y_val_comb)
    pruned_trees_combined.append(pruned_tree)

rf_combined_pruned = copy.deepcopy(rf_combined)
rf_combined_pruned.estimators_ = pruned_trees_combined
y_pred_combined = rf_combined_pruned.predict(X_test_comb)

# --- Evaluation Metrics ---
metrics = {
    "Standard RF": {
        "Accuracy": accuracy_score(y_test_std, y_pred_std),
        "Precision": precision_score(y_test_std, y_pred_std, average='weighted'),
        "Recall": recall_score(y_test_std, y_pred_std, average='weighted'),
        "F1-Score": f1_score(y_test_std, y_pred_std, average='weighted'),
    },
    "Combined RF": {
        "Accuracy": accuracy_score(y_test_comb, y_pred_combined),
        "Precision": precision_score(y_test_comb, y_pred_combined, average='weighted'),
        "Recall": recall_score(y_test_comb, y_pred_combined, average='weighted'),
        "F1-Score": f1_score(y_test_comb, y_pred_combined, average='weighted'),
    }
}

# --- Print Results ---
print(f"Best Number of Clusters for Co-Clustering: {best_clusters}\n")

print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nCombined Approach (Co-Clustering + Pruning):")
print(classification_report(y_test_comb, y_pred_combined))

print("\n--- Comparative Analysis ---")
print(f"{'Metric':<15}{'Standard RF':<15}{'Combined RF':<15}")
for metric in metrics["Standard RF"]:
    print(f"{metric:<15}", end="")
    for model in metrics:
        print(f"{metrics[model][metric]:<15.4f}", end="")
    print()
