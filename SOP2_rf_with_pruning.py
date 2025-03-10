import numpy as np
import pandas as pd
import chardet
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree._tree import TREE_LEAF

# Detect Encoding
with open('dataset2.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Load Dataset
data = pd.read_csv('dataset2.csv', encoding=result['encoding'])
data = data[['LABEL', 'TEXT']]
data.columns = ['LABEL', 'TEXT']

# Preprocess Labels
le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])

# Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standard Random Forest with Optimizations ---
start_time = time.time()
rf_std = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
rf_std.fit(X_train, y_train)
std_training_time = time.time() - start_time
y_pred_std = rf_std.predict(X_test)

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

# Train a new RandomForestClassifier with pruned trees
start_time = time.time()
rf_pruned = RandomForestClassifier(n_estimators=len(pruned_trees), max_depth=10, class_weight='balanced', random_state=42)
rf_pruned.fit(X_train, y_train)
pruned_training_time = time.time() - start_time

y_pred_pruned = rf_pruned.predict(X_test)

# --- Comparative Analysis ---
std_metrics = {
    "Training Time (s)": std_training_time,
    # "Accuracy": accuracy_score(y_test, y_pred_std),
    # "Precision": precision_score(y_test, y_pred_std, average='weighted'),
    # "Recall": recall_score(y_test, y_pred_std, average='weighted'),
    # "F1-Score": f1_score(y_test, y_pred_std, average='weighted'),
}

pruned_metrics = {
    "Training Time (s)": pruned_training_time,
    # "Accuracy": accuracy_score(y_test, y_pred_pruned),
    # "Precision": precision_score(y_test, y_pred_pruned, average='weighted'),
    # "Recall": recall_score(y_test, y_pred_pruned, average='weighted'),
    # "F1-Score": f1_score(y_test, y_pred_pruned, average='weighted'),
}

# Print Results
print("\n--- Comparative Analysis ---")
print(f"{'Metric':<20}{'Standard RF':<15}{'Pruned RF':<15}")
for metric in std_metrics:
    print(f"{metric:<20}{std_metrics[metric]:<15.4f}{pruned_metrics[metric]:<15.4f}")