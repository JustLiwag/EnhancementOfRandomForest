import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import chardet
import os
import copy
import time
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import SpectralCoclustering
import torch.nn as nn
import joblib
import glob
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="SMS Fraud Detection",
    page_icon="🔍",
    layout="centered"
)

# Create directories for model storage
MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Title and description
st.title("SMS Fraud Detection")
st.markdown("This application uses an enhanced Random Forest algorithm to detect fraudulent SMS messages.")

# Function to reduce error pruning
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

# CFCN Model definition
class CFCN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(CFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        contribution_scores = torch.sigmoid(self.fc2(hidden))
        output = self.fc3(contribution_scores * x)
        return output, contribution_scores

# Function to get available datasets
@st.cache_data
def get_available_datasets():
    datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    dataset_files = glob.glob(os.path.join(datasets_path, "*.csv"))
    return [os.path.basename(f) for f in dataset_files]

# Function to load a dataset
@st.cache_data
def load_dataset(dataset_name):
    datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    file_path = os.path.join(datasets_path, dataset_name)
    
    # Detect Encoding
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    
    # Load Dataset
    try:
        df = pd.read_csv(file_path, encoding=result['encoding'])
        
        # Try to extract relevant columns
        if set(['LABEL', 'TEXT']).issubset(df.columns):
            df = df[['LABEL', 'TEXT']]
        elif set(['label', 'text']).issubset(df.columns):
            df = df[['label', 'text']]
            df.columns = ['LABEL', 'TEXT']
        elif set(['v1', 'v2']).issubset(df.columns):  # Common format for spam datasets
            df = df[['v1', 'v2']]
            df.columns = ['LABEL', 'TEXT']
        elif set(['Category', 'Message']).issubset(df.columns):
            df = df[['Category', 'Message']]
            df.columns = ['LABEL', 'TEXT']
        else:
            # Take the first two columns and assume they are label and text
            first_cols = df.iloc[:, :2]
            df = first_cols
            df.columns = ['LABEL', 'TEXT']
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Function to preprocess the dataset
@st.cache_data
def preprocess_dataset(data):
    # Standardize labels for binary classification (fraud/not fraud, spam/ham)
    label_set = set(data['LABEL'].str.lower())
    
    # Check if it's a spam/ham dataset
    if 'spam' in label_set and 'ham' in label_set:
        data['LABEL'] = data['LABEL'].str.lower().map({'spam': 'Fraud', 'ham': 'Not Fraud'})
    
    # Check for other common labels
    if '1' in label_set and '0' in label_set:
        data['LABEL'] = data['LABEL'].map({'1': 'Fraud', '0': 'Not Fraud'})
    
    if 'fraud' in label_set and 'not fraud' not in label_set:
        non_fraud_labels = [l for l in label_set if l != 'fraud']
        if non_fraud_labels:
            mapping = {'fraud': 'Fraud'}
            for l in non_fraud_labels:
                mapping[l] = 'Not Fraud'
            data['LABEL'] = data['LABEL'].str.lower().map(mapping)
    
    # Normalize to proper case
    data['LABEL'] = data['LABEL'].str.capitalize()
    
    # Make sure we have exactly two classes for binary classification
    if len(data['LABEL'].unique()) > 2:
        most_common = data['LABEL'].value_counts().index[0]
        data['LABEL'] = data['LABEL'].apply(lambda x: x if x == most_common else 'Other')
        data['LABEL'] = data['LABEL'].map({most_common: 'Fraud', 'Other': 'Not Fraud'})
    
    return data

# Function to train models
def train_models(data, model_type="all"):
    """
    Trains the specified model type on the dataset.
    Returns the trained models and preprocessing components.
    """
    # Preprocess Labels
    le = LabelEncoder()
    y = le.fit_transform(data['LABEL'])
    
    # Text Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['TEXT'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    training_results = {}
    
    # Standard Random Forest
    if model_type in ["all", "Standard"]:
        start_time = time.time()
        rf_std = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf_std.fit(X_train, y_train)
        y_pred_std = rf_std.predict(X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred_std)
        f1 = f1_score(y_test, y_pred_std, average='weighted')
        
        models["Standard"] = {
            "model": rf_std,
            "vectorizer": vectorizer,
            "label_encoder": le
        }
        
        training_results["Standard"] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": training_time
        }
    
    # Pruned Random Forest
    if model_type in ["all", "Pruned"]:
        start_time = time.time()
        X_train_prune, X_val_prune, y_train_prune, y_val_prune = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        rf_prune = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
        rf_prune.fit(X_train_prune, y_train_prune)
        
        pruned_trees = []
        for tree in rf_prune.estimators_:
            pruned_tree = reduced_error_pruning(tree, X_val_prune, y_val_prune)
            pruned_trees.append(pruned_tree)
        
        rf_pruned = copy.deepcopy(rf_prune)
        rf_pruned.estimators_ = pruned_trees
        
        y_pred_pruned = rf_pruned.predict(X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred_pruned)
        f1 = f1_score(y_test, y_pred_pruned, average='weighted')
        
        models["Pruned"] = {
            "model": rf_pruned,
            "vectorizer": vectorizer,
            "label_encoder": le
        }
        
        training_results["Pruned"] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": training_time
        }
    
    # Co-Clustered Random Forest
    if model_type in ["all", "Co-Clustered"]:
        start_time = time.time()
        X_csc = X.tocsc()
        
        # Remove zero rows and columns
        nonzero_row_indices = np.array(X_csc.sum(axis=1)).flatten() > 0
        nonzero_col_indices = np.array(X_csc.sum(axis=0)).flatten() > 0
        X_filtered = X_csc[nonzero_row_indices, :]
        X_filtered = X_filtered[:, nonzero_col_indices]
        y_filtered = y[nonzero_row_indices]
        
        # Determine number of clusters
        n_clusters = min(3, min(X_filtered.shape[0], X_filtered.shape[1]))
        
        # Apply co-clustering
        co_model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
        co_model.fit(X_filtered)
        
        # Select features from the first cluster
        selected_features = co_model.get_indices(0)[1]
        X_reduced = X_filtered[:, selected_features]
        
        # Train on reduced feature set
        X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
            X_reduced, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )
        
        rf_enh = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf_enh.fit(X_train_enh, y_train_enh)
        
        y_pred_enh = rf_enh.predict(X_test_enh)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_enh, y_pred_enh)
        f1 = f1_score(y_test_enh, y_pred_enh, average='weighted')
        
        models["Co-Clustered"] = {
            "model": rf_enh,
            "vectorizer": vectorizer,
            "label_encoder": le,
            "coclustering_model": co_model,
            "nonzero_col_indices": nonzero_col_indices,
            "selected_features": selected_features
        }
        
        training_results["Co-Clustered"] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": training_time
        }
    
    # Combined Random Forest (Co-Clustering + Pruning)
    if model_type in ["all", "Combined"]:
        start_time = time.time()
        X_csc = X.tocsc()
        
        # Remove zero rows and columns
        nonzero_row_indices = np.array(X_csc.sum(axis=1)).flatten() > 0
        nonzero_col_indices = np.array(X_csc.sum(axis=0)).flatten() > 0
        X_filtered = X_csc[nonzero_row_indices, :]
        X_filtered = X_filtered[:, nonzero_col_indices]
        y_filtered = y[nonzero_row_indices]
        
        # Determine number of clusters
        n_clusters = min(3, min(X_filtered.shape[0], X_filtered.shape[1]))
        
        # Apply co-clustering
        co_model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
        co_model.fit(X_filtered)
        
        # Select features from the first cluster
        selected_features = co_model.get_indices(0)[1]
        X_combined = X_filtered[:, selected_features]
        
        # Split data for pruning
        X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
            X_combined, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )
        
        X_train_comb_prune, X_val_comb, y_train_comb_prune, y_val_comb = train_test_split(
            X_train_comb, y_train_comb, test_size=0.2, random_state=42, stratify=y_train_comb
        )
        
        # Train and prune
        rf_combined = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=10)
        rf_combined.fit(X_train_comb_prune, y_train_comb_prune)
        
        pruned_trees_combined = []
        for tree in rf_combined.estimators_:
            pruned_tree = reduced_error_pruning(tree, X_val_comb, y_val_comb)
            pruned_trees_combined.append(pruned_tree)
        
        rf_combined_pruned = copy.deepcopy(rf_combined)
        rf_combined_pruned.estimators_ = pruned_trees_combined
        
        y_pred_combined = rf_combined_pruned.predict(X_test_comb)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_comb, y_pred_combined)
        f1 = f1_score(y_test_comb, y_pred_combined, average='weighted')
        
        models["Combined"] = {
            "model": rf_combined_pruned,
            "vectorizer": vectorizer,
            "label_encoder": le,
            "coclustering_model": co_model,
            "nonzero_col_indices": nonzero_col_indices,
            "selected_features": selected_features
        }
        
        training_results["Combined"] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": training_time
        }
    
    # CFCN Model
    if model_type in ["all", "CFCN"]:
        start_time = time.time()
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
        )
        
        input_dim = X_train_tensor.shape[1]
        num_classes = len(np.unique(y))
        cfc_model = CFCN(input_dim, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cfc_model.parameters(), lr=0.001)
        
        # Train for a few epochs
        for epoch in range(10):
            optimizer.zero_grad()
            outputs, _ = cfc_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        cfc_model.eval()
        with torch.no_grad():
            y_pred_probs, _ = cfc_model(X_test_tensor)
            y_pred_cfcn = torch.argmax(y_pred_probs, axis=1).numpy()
        
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_tensor, y_pred_cfcn)
        f1 = f1_score(y_test_tensor, y_pred_cfcn, average='weighted')
        
        models["CFCN"] = {
            "model": cfc_model,
            "vectorizer": vectorizer,
            "label_encoder": le
        }
        
        training_results["CFCN"] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": training_time
        }
    
    return models, training_results

# Function to save trained models
def save_models(models, dataset_name):
    """Saves trained models to disk"""
    dataset_base = os.path.splitext(dataset_name)[0]
    
    for model_type, model_data in models.items():
        model_dir = os.path.join(MODELS_DIR, dataset_base)
        os.makedirs(model_dir, exist_ok=True)
        
        if model_type != "CFCN":
            # Save sklearn models
            joblib.dump(model_data["model"], os.path.join(model_dir, f"{model_type}_model.joblib"))
            joblib.dump(model_data["vectorizer"], os.path.join(model_dir, f"{model_type}_vectorizer.joblib"))
            joblib.dump(model_data["label_encoder"], os.path.join(model_dir, f"{model_type}_label_encoder.joblib"))
            
            # Save additional components for co-clustering models
            if model_type in ["Co-Clustered", "Combined"]:
                joblib.dump(model_data["coclustering_model"], os.path.join(model_dir, f"{model_type}_coclustering.joblib"))
                joblib.dump(model_data["nonzero_col_indices"], os.path.join(model_dir, f"{model_type}_nonzero_cols.joblib"))
                joblib.dump(model_data["selected_features"], os.path.join(model_dir, f"{model_type}_selected_features.joblib"))
        else:
            # Save PyTorch model
            torch.save(model_data["model"].state_dict(), os.path.join(model_dir, f"{model_type}_model.pt"))
            joblib.dump(model_data["vectorizer"], os.path.join(model_dir, f"{model_type}_vectorizer.joblib"))
            joblib.dump(model_data["label_encoder"], os.path.join(model_dir, f"{model_type}_label_encoder.joblib"))

# Function to load trained models
def load_models(dataset_name, model_type):
    """Loads trained models from disk"""
    dataset_base = os.path.splitext(dataset_name)[0]
    model_dir = os.path.join(MODELS_DIR, dataset_base)
    
    if not os.path.exists(model_dir):
        return None
    
    model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
    vectorizer_path = os.path.join(model_dir, f"{model_type}_vectorizer.joblib")
    label_encoder_path = os.path.join(model_dir, f"{model_type}_label_encoder.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
        if model_type == "CFCN":
            # Check for PyTorch model
            model_path = os.path.join(model_dir, f"{model_type}_model.pt")
            if not os.path.exists(model_path):
                return None
        else:
            return None
    
    try:
        if model_type != "CFCN":
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            label_encoder = joblib.load(label_encoder_path)
            
            model_data = {
                "model": model,
                "vectorizer": vectorizer,
                "label_encoder": label_encoder
            }
            
            # Load additional components for co-clustering models
            if model_type in ["Co-Clustered", "Combined"]:
                coclustering_path = os.path.join(model_dir, f"{model_type}_coclustering.joblib")
                nonzero_cols_path = os.path.join(model_dir, f"{model_type}_nonzero_cols.joblib")
                selected_features_path = os.path.join(model_dir, f"{model_type}_selected_features.joblib")
                
                if all(os.path.exists(p) for p in [coclustering_path, nonzero_cols_path, selected_features_path]):
                    model_data["coclustering_model"] = joblib.load(coclustering_path)
                    model_data["nonzero_col_indices"] = joblib.load(nonzero_cols_path)
                    model_data["selected_features"] = joblib.load(selected_features_path)
            
            return model_data
        else:
            # Load PyTorch model
            vectorizer = joblib.load(vectorizer_path)
            label_encoder = joblib.load(label_encoder_path)
            
            # Get input dimension from vectorizer
            input_dim = len(vectorizer.get_feature_names_out())
            num_classes = len(label_encoder.classes_)
            
            # Initialize model
            model = CFCN(input_dim, num_classes)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            return {
                "model": model,
                "vectorizer": vectorizer,
                "label_encoder": label_encoder
            }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to predict
def predict_fraud(text, dataset_name, model_type):
    """
    Predicts if the given text is fraudulent using the specified model type.
    """
    # Load the model
    model_data = load_models(dataset_name, model_type)
    
    if model_data is None:
        st.error(f"Model not found for {dataset_name} - {model_type}. Please train the model first.")
        return None
    
    # Extract components
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    label_encoder = model_data["label_encoder"]
    
    # Process the input text
    new_text_vec = vectorizer.transform([text])
    
    # Make prediction based on model type
    if model_type in ["Standard", "Pruned"]:
        prediction = model.predict(new_text_vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    elif model_type in ["Co-Clustered", "Combined"]:
        # Apply co-clustering transformations
        new_text_vec = new_text_vec.tocsc()
        
        # Apply feature selection
        nonzero_col_indices = model_data["nonzero_col_indices"]
        selected_features = model_data["selected_features"]
        
        # Check if dimensions match and apply transformations
        if new_text_vec.shape[1] > len(nonzero_col_indices):
            new_text_vec = new_text_vec[:, nonzero_col_indices]
        
        if new_text_vec.shape[1] > len(selected_features):
            new_text_vec = new_text_vec[:, selected_features]
        
        prediction = model.predict(new_text_vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    elif model_type == "CFCN":
        # Convert to tensor
        new_text_tensor = torch.tensor(new_text_vec.toarray(), dtype=torch.float32)
        
        with torch.no_grad():
            y_pred_probs, _ = model(new_text_tensor)
            prediction = torch.argmax(y_pred_probs, axis=1).numpy()[0]
        
        predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    return predicted_label

# Sidebar for dataset selection and model training
st.sidebar.header("Dataset Selection")

available_datasets = get_available_datasets()
selected_dataset = st.sidebar.selectbox("Select Dataset", available_datasets)

st.sidebar.markdown("---")
st.sidebar.header("Model Training")

if st.sidebar.button("Train Models"):
    with st.spinner("Loading dataset..."):
        data = load_dataset(selected_dataset)
        
        if data is not None:
            # Display dataset info
            st.sidebar.write(f"Dataset loaded: {len(data)} records")
            
            # Preprocess data
            data = preprocess_dataset(data)
            
            # Train models
            with st.spinner(f"Training models on {selected_dataset}..."):
                models, results = train_models(data)
                
                # Save models
                save_models(models, selected_dataset)
                
                # Display results
                st.sidebar.success("Training complete!")
                for model_type, metrics in results.items():
                    st.sidebar.write(f"**{model_type}**")
                    st.sidebar.write(f"Accuracy: {metrics['accuracy']:.4f}")
                    st.sidebar.write(f"F1 Score: {metrics['f1_score']:.4f}")
                    st.sidebar.write(f"Training Time: {metrics['training_time']:.2f}s")
                    st.sidebar.write("---")

# Check if models are trained for the selected dataset
dataset_base = os.path.splitext(selected_dataset)[0]
model_dir = os.path.join(MODELS_DIR, dataset_base)
models_trained = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0

# Main app
st.subheader("SMS Classification")

if not models_trained:
    st.warning(f"No trained models found for {selected_dataset}. Please train the models first.")

# Select model type
available_model_types = ["Standard", "Pruned", "Co-Clustered", "Combined"]
model_type = st.selectbox("Select Model Type", available_model_types)

# User input
user_input = st.text_area("Enter the SMS message to classify:", "")

if st.button("Classify"):
    if user_input:
        if models_trained:
            with st.spinner("Classifying..."):
                try:
                    # Start the classification process
                    start_time = time.time()
                    
                    result = predict_fraud(user_input, selected_dataset, model_type)
                    
                    end_time = time.time()
                    
                    if result:
                        # Display the result
                        if result == "Fraud":
                            st.error("🚨 Fraud Detected!")
                        else:
                            st.success("✅ Not Fraud")
                            
                        st.info(f"Classification completed in {end_time - start_time:.2f} seconds using {model_type} model")
                    else:
                        st.error("Classification failed. Please train the model first.")
                
                except Exception as e:
                    st.error(f"An error occurred during classification: {str(e)}")
                    st.exception(e)
        else:
            st.error("Please train the models first by clicking the 'Train Models' button in the sidebar.")
    else:
        st.warning("Please enter some text to classify.")

# Example messages for quick testing
with st.expander("Example Messages for Testing"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraudulent Messages")
        fraud_examples = [
            "Congratulations! You've won a $1000 gift card. To claim call now: +1-800-555-1234",
            "URGENT: Your bank account has been suspended. Call this number immediately to verify: 555-123-4567",
            "50% OFF! Limited time offer on luxury watches. Click here: www.fakewatches.com"
        ]
        
        for i, example in enumerate(fraud_examples):
            if st.button(f"Test Fraud Example {i+1}", key=f"fraud_{i}"):
                st.session_state.user_input = example
                st.rerun()
    
    with col2:
        st.subheader("Legitimate Messages")
        legit_examples = [
            "Hey Mark, are we still meeting for coffee tomorrow at 3pm?",
            "The meeting has been moved to 2pm. Please bring your presentation.",
            "Can you pick up milk on your way home? Thanks."
        ]
        
        for i, example in enumerate(legit_examples):
            if st.button(f"Test Legit Example {i+1}", key=f"legit_{i}"):
                st.session_state.user_input = example
                st.rerun()

# Information about the models
with st.expander("About the Models"):
    st.markdown("""
    ## Model Types
    - **Standard**: Basic Random Forest classifier
    - **Pruned**: Random Forest with Reduced Error Pruning
    - **Co-Clustered**: Random Forest with Spectral Co-Clustering for feature selection
    - **Combined**: Uses both Co-Clustering and Pruning techniques
    
    These enhanced Random Forest techniques improve classification accuracy and model interpretability.
    """)

# Footer
st.markdown("---")
st.markdown("SMS Fraud Detection System | Random Forest Enhancements") 