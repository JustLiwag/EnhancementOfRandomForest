# MARCH 06, 2025
# SOP2: Random Forest with Reduced Error Pruning

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import chardet
import time
from joblib import Parallel, delayed

class PrunedRandomForestClassifier:
    """
    Random Forest Classifier with Reduced Error Pruning for individual trees.
    This implementation trains decision trees and then prunes them using a validation set,
    helping to reduce overfitting and improve performance.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 class_weight=None, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = None
        
    def _prune_tree(self, tree, X_val, y_val):
        """Prune a decision tree using reduced error pruning with validation set"""
        # Recursive function to prune nodes
        def _prune_node(node_id):
            # If this is a leaf node, return
            if tree.tree_.children_left[node_id] == -1:
                return
            
            # Prune the left and right subtrees first (bottom-up pruning)
            _prune_node(tree.tree_.children_left[node_id])
            _prune_node(tree.tree_.children_right[node_id])
            
            # If both children are leaf nodes, consider pruning this node
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]
            
            if (tree.tree_.children_left[left_child] == -1 and 
                tree.tree_.children_right[left_child] == -1 and
                tree.tree_.children_left[right_child] == -1 and
                tree.tree_.children_right[right_child] == -1):
                
                # Store the original values
                original_left = tree.tree_.children_left[node_id]
                original_right = tree.tree_.children_right[node_id]
                
                # Temporarily make this node a leaf by setting children to -1
                tree.tree_.children_left[node_id] = -1
                tree.tree_.children_right[node_id] = -1
                
                # Check if pruning improves accuracy on validation set
                accuracy_after_pruning = accuracy_score(y_val, tree.predict(X_val))
                
                # Restore the original structure (if pruning doesn't help)
                if accuracy_after_pruning < self.original_accuracy:
                    tree.tree_.children_left[node_id] = original_left
                    tree.tree_.children_right[node_id] = original_right
        
        # Calculate original accuracy before pruning
        self.original_accuracy = accuracy_score(y_val, tree.predict(X_val))
        
        # Start pruning from the root node
        _prune_node(0)
        
        return tree
    
    def fit(self, X, y):
        """
        Build a forest of pruned trees
        """
        self.classes_ = np.unique(y)
        
        # Split data into training and validation sets for pruning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        
        # Train trees in parallel
        def _fit_and_prune(i):
            # Use a random seed based on the estimator index
            seed = None if self.random_state is None else self.random_state + i
            
            # Train a single decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                class_weight=self.class_weight,
                random_state=seed
            )
            tree.fit(X_train, y_train)
            
            # Prune the tree
            pruned_tree = self._prune_tree(tree, X_val, y_val)
            
            return pruned_tree
        
        # Create pruned trees in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_and_prune)(i) for i in range(self.n_estimators)
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class for X using the pruned forest
        """
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Take majority vote
        majority_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=len(self.classes_))),
            axis=0,
            arr=predictions
        )
        
        return majority_vote

    def predict_proba(self, X):
        """
        Predict class probabilities for X using the pruned forest
        """
        # Get predictions from all trees
        predictions = np.array([tree.predict_proba(X) for tree in self.estimators_])
        
        # Average probabilities
        return np.mean(predictions, axis=0)


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

# --- 1. Standard Random Forest ---
print("Training Standard Random Forest...")
start_time = time.time()
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, test_size=0.2, random_state=42)
rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_std.fit(X_train_std, y_train_std)
y_pred_std = rf_std.predict(X_test_std)
std_time = time.time() - start_time
print(f"Standard RF training completed in {std_time:.2f} seconds")

# --- 2. Enhanced Random Forest with Spectral Co-Clustering (from SOP1) ---
print("\nTraining Enhanced Random Forest with Spectral Co-Clustering...")
start_time = time.time()

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
for n_clusters in range(2, 6):  # Testing fewer clusters for faster results
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

    # Evaluate F1 Score for the weighted average
    f1 = f1_score(y_test_enh, y_pred_enh, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_clusters = n_clusters
        best_model = (rf_enh, X_train_enh, X_test_enh, y_train_enh, y_test_enh, y_pred_enh)

rf_enh, X_train_enh, X_test_enh, y_train_enh, y_test_enh, y_pred_enh = best_model
enh_time = time.time() - start_time
print(f"Enhanced RF training completed in {enh_time:.2f} seconds")

# --- 3. Pruned Random Forest ---
print("\nTraining Pruned Random Forest...")
start_time = time.time()
# Use the same train/test split as the standard RF
X_train_pruned, X_test_pruned, y_train_pruned, y_test_pruned = X_train_std, X_test_std, y_train_std, y_test_std

# Train Pruned Random Forest
rf_pruned = PrunedRandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_pruned.fit(X_train_pruned, y_train_pruned)
y_pred_pruned = rf_pruned.predict(X_test_pruned)
pruned_time = time.time() - start_time
print(f"Pruned RF training completed in {pruned_time:.2f} seconds")

# --- Comparative Analysis ---
# Standard Random Forest Metrics
std_metrics = {
    "Accuracy": accuracy_score(y_test_std, y_pred_std),
    "Precision": precision_score(y_test_std, y_pred_std, average='weighted'),
    "Recall": recall_score(y_test_std, y_pred_std, average='weighted'),
    "F1-Score": f1_score(y_test_std, y_pred_std, average='weighted'),
    "Training Time": std_time
}

# Enhanced Random Forest Metrics
enh_metrics = {
    "Accuracy": accuracy_score(y_test_enh, y_pred_enh),
    "Precision": precision_score(y_test_enh, y_pred_enh, average='weighted'),
    "Recall": recall_score(y_test_enh, y_pred_enh, average='weighted'),
    "F1-Score": f1_score(y_test_enh, y_pred_enh, average='weighted'),
    "Training Time": enh_time
}

# Pruned Random Forest Metrics
pruned_metrics = {
    "Accuracy": accuracy_score(y_test_pruned, y_pred_pruned),
    "Precision": precision_score(y_test_pruned, y_pred_pruned, average='weighted'),
    "Recall": recall_score(y_test_pruned, y_pred_pruned, average='weighted'),
    "F1-Score": f1_score(y_test_pruned, y_pred_pruned, average='weighted'),
    "Training Time": pruned_time
}

# Print Results
print("\n--- Detailed Classification Reports ---")
print("Standard Random Forest:")
print(classification_report(y_test_std, y_pred_std))

print("\nEnhanced Random Forest with Spectral Co-Clustering:")
print(f"Best Number of Clusters: {best_clusters}")
print(classification_report(y_test_enh, y_pred_enh))

print("\nPruned Random Forest:")
print(classification_report(y_test_pruned, y_pred_pruned))

print("\n--- Comparative Analysis ---")
print(f"{'Metric':<15}{'Standard RF':<15}{'Enhanced RF':<15}{'Pruned RF':<15}")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    print(f"{metric:<15}{std_metrics[metric]:<15.4f}{enh_metrics[metric]:<15.4f}{pruned_metrics[metric]:<15.4f}")

print(f"\n{'Training Time':<15}{std_metrics['Training Time']:<15.2f}{enh_metrics['Training Time']:<15.2f}{pruned_metrics['Training Time']:<15.2f} seconds")

# Improvement percentages
print("\n--- Performance Improvements ---")
print("Pruned RF vs Standard RF:")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    improvement = ((pruned_metrics[metric] - std_metrics[metric]) / std_metrics[metric]) * 100
    print(f"{metric} improvement: {improvement:.2f}%")

time_improvement = ((std_metrics['Training Time'] - pruned_metrics['Training Time']) / std_metrics['Training Time']) * 100
print(f"Training time improvement: {time_improvement:.2f}%")

print("\nPruned RF vs Enhanced RF:")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    improvement = ((pruned_metrics[metric] - enh_metrics[metric]) / enh_metrics[metric]) * 100
    print(f"{metric} improvement: {improvement:.2f}%")

time_improvement = ((enh_metrics['Training Time'] - pruned_metrics['Training Time']) / enh_metrics['Training Time']) * 100
print(f"Training time improvement: {time_improvement:.2f}%")
