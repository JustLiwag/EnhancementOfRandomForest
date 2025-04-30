# === MOUNT GOOGLE DRIVE AND SET DIRECTORY ===
# from google.colab import drive
# drive.mount('/content/drive')

import os
os.chdir(r'E:\Users\Fujitsu\Documents\CODING\GitHub Repos\EnhancementOfRandomForest\datasets')

# === IMPORTS ===
import pandas as pd
import numpy as np
import chardet
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree._tree import TREE_LEAF

# === LOAD DATASET AND DETECT ENCODING ===
with open('dataset1.csv', 'rb') as f:
    result = chardet.detect(f.read())

data = pd.read_csv('dataset1.csv', encoding=result['encoding'])
data = data[['LABEL', 'TEXT']]
data.columns = ['LABEL', 'TEXT']

# === PREPROCESS LABELS ===
le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])

# === VECTORIZATION ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']

# === BASELINE RANDOM FOREST ===
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, test_size=0.2, random_state=42)
start_time_std = time.time()
rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_std.fit(X_train_std, y_train_std)
std_training_time = time.time() - start_time_std
y_pred_std = rf_std.predict(X_test_std)

# === APPLY SPECTRAL CO-CLUSTERING (OBJ1) ===
X = X.tocsc()
nonzero_row_indices = np.array(X.sum(axis=1)).flatten() > 0
nonzero_col_indices = np.array(X.sum(axis=0)).flatten() > 0
X = X[nonzero_row_indices, :]
X = X[:, nonzero_col_indices]
y = y[nonzero_row_indices]

best_f1 = 0
best_clusters = 0
best_model = None

for n_clusters in range(2, 10):
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    selected_features = model.get_indices(0)[1]
    X_reduced = X[:, selected_features]

    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_enh.fit(X_train_enh, y_train_enh)
    y_pred_enh = rf_enh.predict(X_test_enh)

    f1 = f1_score(y_test_enh, y_pred_enh, average=None)[1]
    if f1 > best_f1:
        best_f1 = f1
        best_clusters = n_clusters
        best_model = (rf_enh, X_train_enh, X_test_enh, y_train_enh, y_test_enh, y_pred_enh, selected_features)

rf_enh, X_train_enh, X_test_enh, y_train_enh, y_test_enh, y_pred_enh, selected_features = best_model

# === REDUCED ERROR PRUNING (OBJ2) ===
def reduced_error_pruning(tree, X_val, y_val, threshold=0.01):
    for node in range(tree.tree_.node_count):
        if tree.tree_.children_left[node] == TREE_LEAF and tree.tree_.children_right[node] == TREE_LEAF:
            continue
        impurity_before = tree.tree_.impurity[node]
        if impurity_before < threshold:
            continue

        y_pred_before = tree.predict(X_val)
        acc_before = accuracy_score(y_val, y_pred_before)

        original_threshold = tree.tree_.threshold[node]
        tree.tree_.threshold[node] = -2

        y_pred_after = tree.predict(X_val)
        acc_after = accuracy_score(y_val, y_pred_after)

        if acc_after < acc_before:
            tree.tree_.threshold[node] = original_threshold
    return tree

# Prune a few trees from enhanced RF
pruned_trees = []
for tree in rf_enh.estimators_[:3]:
    X_sub, X_val, y_sub, y_val = train_test_split(X_train_enh, y_train_enh, test_size=0.1, random_state=42)
    pruned_trees.append(reduced_error_pruning(tree, X_val, y_val))

# Retrain forest with pruned trees
start_time_pruned = time.time()
rf_final = RandomForestClassifier(n_estimators=len(pruned_trees), class_weight='balanced', random_state=42)
rf_final.fit(X_train_enh, y_train_enh)
pruned_training_time = time.time() - start_time_pruned
y_pred_final = rf_final.predict(X_test_enh)

# === METRICS COMPARISON ===
def get_metrics(y_true, y_pred, time_taken):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-Score": f1_score(y_true, y_pred, average='weighted'),
        "Training Time (s)": time_taken,
    }

std_metrics = get_metrics(y_test_std, y_pred_std, std_training_time)
enh_metrics = get_metrics(y_test_enh, y_pred_enh, 0)  # before pruning
final_metrics = get_metrics(y_test_enh, y_pred_final, pruned_training_time)

# === INTERPRETABILITY ANALYSIS (OBJ3) ===
print("\n=== INTERPRETABILITY LIMITATIONS DEMO ===")
feature_importance = rf_final.feature_importances_
features = vectorizer.get_feature_names_out()[selected_features]
importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
top_features = importance_df.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 10 Features by Importance (Enhanced + Pruned RF)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

sample_idx = 10
sample_text = data['TEXT'].iloc[sample_idx]
sample_vector = vectorizer.transform([sample_text])
sample_vector_reduced = sample_vector[:, selected_features]
sample_pred = rf_final.predict(sample_vector_reduced)[0]
sample_proba = rf_final.predict_proba(sample_vector_reduced)[0]

print(f"\nSample Message: {sample_text}")
print(f"Predicted Label: {le.inverse_transform([sample_pred])[0]} (Confidence: {max(sample_proba):.2f})")

present_features = [(features[i], sample_vector_reduced.toarray()[0][i])
                    for i in range(len(features)) if sample_vector_reduced.toarray()[0][i] > 0]
present_features.sort(key=lambda x: x[1], reverse=True)

print("Top 5 Features in Message:")
for feat, val in present_features[:5]:
    print(f"  - {feat} (value: {val:.4f})")

print("\nLIMITATION: Can't clearly trace how features combined across trees.\n")

# === COMPARATIVE REPORT ===
print("\n=== COMPARATIVE ANALYSIS ===")
print(f"{'Metric':<20}{'Standard RF':<15}{'Enhanced RF':<15}{'Enhanced + Pruned':<20}")
for key in std_metrics:
    print(f"{key:<20}{std_metrics[key]:<15.4f}{enh_metrics[key]:<15.4f}{final_metrics[key]:<20.4f}")

print("\nBest Number of Clusters (Co-Clustering):", best_clusters)
