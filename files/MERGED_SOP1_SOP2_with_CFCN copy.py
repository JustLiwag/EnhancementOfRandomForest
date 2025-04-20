"""
This program combines three Random Forest enhancements:
1. Spectral Co-Clustering
2. Reduced Error Pruning
3. Contextual Feature Contribution Network (CFCN)

The process works as follows:

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

6. CFCN Integration:
   - Implements Contextual Feature Contribution Network
   - Analyzes feature contributions
   - Visualizes top contributing features
   - Enhances interpretability of the model

7. Comprehensive Evaluation:
   - Compares all approaches:
     * Standard Random Forest (baseline)
     * Pruning-Only Random Forest
     * Co-Clustering-Only Random Forest
     * Combined Enhancement Random Forest
     * CFCN-Enhanced Random Forest
   - Evaluates using multiple metrics:
     * Accuracy
     * Precision
     * Recall
     * F1-Score
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
import pandas as pd
import chardet
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt

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

# --- Enhanced Random Forest with Pruning Only ---
X_train_prune, X_val_prune, y_train_prune, y_val_prune = train_test_split(X_train_std, y_train_std, test_size=0.2, random_state=42)
rf_prune = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
rf_prune.fit(X_train_prune, y_train_prune)

pruned_trees = []
for tree in rf_prune.estimators_:
    pruned_tree = reduced_error_pruning(tree, X_val_prune, y_val_prune)
    pruned_trees.append(pruned_tree)

rf_pruned = copy.deepcopy(rf_prune)
rf_pruned.estimators_ = pruned_trees
y_pred_pruned = rf_pruned.predict(X_test_std)

# --- Enhanced Random Forest with Spectral Co-Clustering ---
X_csc = X.tocsc()
nonzero_row_indices = np.array(X_csc.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X_csc.sum(axis=0)).flatten() > 0
X_filtered = X_csc[nonzero_row_indices, :]
X_filtered = X_filtered[:, nonzero_col_indices]
y_filtered = y[nonzero_row_indices]

best_f1 = 0
best_clusters = 0
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
        best_model = (rf_enh, X_test_enh, y_test_enh, model, selected_features)

rf_enh, X_test_enh, y_test_enh, best_coclustering_model, best_features = best_model
y_pred_enh = rf_enh.predict(X_test_enh)

# --- Combined Approach: Spectral Co-Clustering + Pruning ---
X_combined = X_filtered[:, best_features]
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

# --- CFCN Implementation ---
class CFCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(CFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, len(np.unique(y)))

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        contribution_scores = torch.sigmoid(self.fc2(hidden))
        output = self.fc3(contribution_scores * x)
        return output, contribution_scores

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Initialize and Train CFCN
input_dim = X_train_tensor.shape[1]
cfc_model = CFCN(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cfc_model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    outputs, _ = cfc_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluate CFCN
cfc_model.eval()
with torch.no_grad():
    y_pred_probs, feature_contributions = cfc_model(X_test_tensor)
    y_pred_cfcn = torch.argmax(y_pred_probs, axis=1).numpy()

# Analyze Feature Contributions
avg_contributions = torch.mean(feature_contributions, axis=0).detach().numpy()
feature_names = vectorizer.get_feature_names_out()
contribution_df = pd.DataFrame({'Feature': feature_names, 'Contribution': avg_contributions})
contribution_df = contribution_df.sort_values(by='Contribution', ascending=False).head(10)

# --- Evaluation Metrics ---
metrics = {
    "Standard RF": {
        "Accuracy": accuracy_score(y_test_std, y_pred_std),
        "Precision": precision_score(y_test_std, y_pred_std, average='weighted'),
        "Recall": recall_score(y_test_std, y_pred_std, average='weighted'),
        "F1-Score": f1_score(y_test_std, y_pred_std, average='weighted'),
    },
    "Pruned RF": {
        "Accuracy": accuracy_score(y_test_std, y_pred_pruned),
        "Precision": precision_score(y_test_std, y_pred_pruned, average='weighted'),
        "Recall": recall_score(y_test_std, y_pred_pruned, average='weighted'),
        "F1-Score": f1_score(y_test_std, y_pred_pruned, average='weighted'),
    },
    "Co-Clustered RF": {
        "Accuracy": accuracy_score(y_test_enh, y_pred_enh),
        "Precision": precision_score(y_test_enh, y_pred_enh, average='weighted'),
        "Recall": recall_score(y_test_enh, y_pred_enh, average='weighted'),
        "F1-Score": f1_score(y_test_enh, y_pred_enh, average='weighted'),
    },
    "Combined RF": {
        "Accuracy": accuracy_score(y_test_comb, y_pred_combined),
        "Precision": precision_score(y_test_comb, y_pred_combined, average='weighted'),
        "Recall": recall_score(y_test_comb, y_pred_combined, average='weighted'),
        "F1-Score": f1_score(y_test_comb, y_pred_combined, average='weighted'),
    },
    "CFCN": {
        "Accuracy": accuracy_score(y_test_tensor, y_pred_cfcn),
        "Precision": precision_score(y_test_tensor, y_pred_cfcn, average='weighted'),
        "Recall": recall_score(y_test_tensor, y_pred_cfcn, average='weighted'),
        "F1-Score": f1_score(y_test_tensor, y_pred_cfcn, average='weighted'),
    }
}

# Print Results
print(f"Best Number of Clusters for Co-Clustering: {best_clusters}\n")

print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nRandom Forest with Pruning Only:")
print(classification_report(y_test_std, y_pred_pruned))

print("\nRandom Forest with Spectral Co-Clustering Only:")
print(classification_report(y_test_enh, y_pred_enh))

print("\nCombined Approach (Co-Clustering + Pruning):")
print(classification_report(y_test_comb, y_pred_combined))

print("\nCFCN Results:")
print(classification_report(y_test_tensor, y_pred_cfcn))

print("\nTop 10 Contributing Features from CFCN:")
print(contribution_df)

print("\n--- Comparative Analysis ---")
print(f"{'Metric':<15}{'Standard RF':<15}{'Pruned RF':<15}{'Co-Clustered RF':<15}{'Combined RF':<15}{'CFCN':<15}")
for metric in metrics["Standard RF"]:
    print(f"{metric:<15}", end="")
    for model in metrics:
        print(f"{metrics[model][metric]:<15.4f}", end="")
    print()

# Visualize Feature Contributions
plt.figure(figsize=(10, 6))
sns.barplot(x='Contribution', y='Feature', data=contribution_df)
plt.title('Top 10 Features by Contribution - CFCN')
plt.tight_layout()
plt.savefig('cfcn_feature_contribution.png')
plt.show() 