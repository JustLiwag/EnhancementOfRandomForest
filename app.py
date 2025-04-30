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
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.cluster import SpectralCoclustering
import torch.nn as nn
import joblib
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim

# Set page configuration
st.set_page_config(
    page_title="SMS Fraud Detection📨",
    page_icon="🔍",
    layout="centered"
)

# Create directories for model storage
MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Title and description
st.title("SMS Fraud Detection")
st.markdown("This application uses pre-trained Random Forest models to detect fraudulent SMS messages.")

# --- Functions for Reduced Error Pruning ---

# Function for reduced error pruning
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

# Function to get available pre-trained models
def get_available_models():
    model_dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return model_dirs

# Function to check and list all model types for a dataset
def get_model_types(dataset_name):
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    if not os.path.exists(model_dir):
        return []
    
    model_files = glob.glob(os.path.join(model_dir, "*_model.joblib"))
    model_types = [os.path.basename(f).replace('_model.joblib', '') for f in model_files]
    return model_types

# Function to load trained models
def load_models(dataset_name, model_type):
    """Loads trained models from disk"""
    dataset_base = os.path.splitext(dataset_name)[0] if '.' in dataset_name else dataset_name
    model_dir = os.path.join(MODELS_DIR, dataset_base)
    
    if not os.path.exists(model_dir):
        st.error(f"No pre-trained models found for {dataset_name}.")
        return None
    
    model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
    vectorizer_path = os.path.join(model_dir, f"{model_type}_vectorizer.joblib")
    label_encoder_path = os.path.join(model_dir, f"{model_type}_label_encoder.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
        # Try with old names for backward compatibility
        old_model_type_mapping = {
            "Standard RF": "Standard",
            "Enhanced RF": "Combined"
        }
        if model_type in old_model_type_mapping:
            old_model_type = old_model_type_mapping[model_type]
            model_path = os.path.join(model_dir, f"{old_model_type}_model.joblib")
            vectorizer_path = os.path.join(model_dir, f"{old_model_type}_vectorizer.joblib")
            label_encoder_path = os.path.join(model_dir, f"{old_model_type}_label_encoder.joblib")
            
            if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
                st.error(f"Missing model files for {model_type}.")
                return None
        else:
            st.error(f"Missing model files for {model_type}.")
            return None
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(label_encoder_path)
        
        model_data = {
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder
        }
        
        # Load additional components for Enhanced RF model
        if model_type == "Enhanced RF":
            old_model_type = "Combined"  # For backward compatibility
            
            # Check both new and old file names
            for component_type in ["coclustering", "nonzero_cols", "selected_features"]:
                new_path = os.path.join(model_dir, f"{model_type}_{component_type}.joblib")
                old_path = os.path.join(model_dir, f"{old_model_type}_{component_type}.joblib")
                
                if os.path.exists(new_path):
                    component_path = new_path
                elif os.path.exists(old_path):
                    component_path = old_path
                else:
                    continue
                
                component_key = {
                    "coclustering": "coclustering_model",
                    "nonzero_cols": "nonzero_col_indices",
                    "selected_features": "selected_features"
                }[component_type]
                
                model_data[component_key] = joblib.load(component_path)
        
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# CFCN Model Definition
class CFCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(CFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, 2)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        contribution_scores = torch.sigmoid(self.fc2(hidden))
        output = self.fc3(contribution_scores * x)
        return output, contribution_scores

# Function to get feature contributions
def get_feature_contributions(text, model_data):
    """Get feature contributions for the input text using CFCN"""
    vectorizer = model_data["vectorizer"]
    
    # Transform input text and get feature names
    text_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    input_dim = len(feature_names)
    
    # Get only the features present in the input text
    text_features = []
    for idx, val in enumerate(text_vec.toarray()[0]):
        if val > 0:
            text_features.append((feature_names[idx], val))
    
    # Initialize CFCN
    cfc_model = CFCN(input_dim)
    
    # Get predictions and contributions
    text_tensor = torch.tensor(text_vec.toarray(), dtype=torch.float32)
    cfc_model.eval()
    with torch.no_grad():
        _, feature_contributions = cfc_model(text_tensor)
    
    # Create contribution dataframe only for features present in the text
    contributions = []
    for feature, value in text_features:
        contribution_score = feature_contributions[0][vectorizer.vocabulary_[feature]].item()
        contributions.append({
            'Feature': feature,
            'Contribution': contribution_score,
            'TF-IDF Value': value
        })
    
    contribution_df = pd.DataFrame(contributions)
    contribution_df = contribution_df.sort_values('Contribution', ascending=False)
    
    return contribution_df

# Function to predict
def predict_fraud(text, dataset_name, model_type):
    """
    Predicts if the given text is fraudulent using the specified model type.
    """
    # Load the model
    model_data = load_models(dataset_name, model_type)
    
    if model_data is None:
        st.error(f"Model not found for {dataset_name} - {model_type}.")
        return None, None
    
    # Extract components
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    label_encoder = model_data["label_encoder"]
    
    # Process the input text
    new_text_vec = vectorizer.transform([text])
    
    # Make prediction based on model type
    if model_type == "Standard RF":
        prediction = model.predict(new_text_vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        feature_contributions = None
    
    elif model_type == "Enhanced RF":
        # Apply co-clustering transformations
        new_text_vec = new_text_vec.tocsc()
        
        # Apply feature selection if available
        nonzero_col_indices = model_data.get("nonzero_col_indices")
        selected_features = model_data.get("selected_features")
        
        if nonzero_col_indices is not None and selected_features is not None:
            if new_text_vec.shape[1] > len(nonzero_col_indices):
                new_text_vec = new_text_vec[:, nonzero_col_indices]
            
            if new_text_vec.shape[1] > len(selected_features):
                new_text_vec = new_text_vec[:, selected_features]
        
        prediction = model.predict(new_text_vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Get feature contributions for visualization (OBJ3)
        feature_contributions = get_feature_contributions(text, model_data)
    
    return predicted_label, feature_contributions

# Check for available pre-trained models
available_models = get_available_models()
if not available_models:
    st.warning("No pre-trained models found. Please ensure models are in the 'trained_models' directory.")
    st.stop()

# Automatically select the first available model set
selected_dataset = available_models[0]

# Main app
# st.subheader("SMS Classification")

# Get available model types for the selected dataset
model_types = get_model_types(selected_dataset)
available_model_types = ["Standard RF", "Enhanced RF"]
# Filter available model types to only show those that are available
filtered_model_types = [mt for mt in available_model_types if mt in model_types]

if not filtered_model_types:
    st.error(f"No model types available for the default model set ({selected_dataset})")
    st.stop()

model_type = st.selectbox("Select Model Type", 
                         filtered_model_types,
                         help="Standard RF: Basic Random Forest | Enhanced RF: Enhanced Random Forest with Co-Clustering and Pruning")

# User input
user_input = st.text_area("Enter the SMS message to classify:", "")

if st.button("Check"):
    if user_input:
        if model_type != "No models available":
            with st.spinner("Checking..."):
                try:
                    # Start the detection process timing
                    detection_start_time = time.time()
                    
                    result, feature_contributions = predict_fraud(user_input, selected_dataset, model_type)
                    
                    # End detection timing
                    detection_end_time = time.time()
                    detection_time = detection_end_time - detection_start_time
                    
                    if result:
                        # Display the result
                        if result == "Fraud":
                            st.error("🚨 Fraud Detected!")
                        else:
                            st.success("✅ Not Fraud")
                            
                        st.info(f"Detection completed in {detection_time:.2f} seconds using {model_type} model")
                        
                        # Show feature contributions for Enhanced RF (OBJ3)
                        if model_type == "Enhanced RF" and feature_contributions is not None:
                            st.subheader("Feature Contributions Analysis")
                            st.write("Most influential features from your message:")
                            
                            if not feature_contributions.empty:
                                # Create and display the visualization
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Contribution', y='Feature', 
                                          data=feature_contributions.head(10), ax=ax)
                                plt.title('Top Features by Contribution')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Display the raw data
                                st.write("Feature Contribution Details:")
                                st.dataframe(feature_contributions[['Feature', 'Contribution', 'TF-IDF Value']].head(10))
                            else:
                                st.warning("No significant features found in the input text.")
                    else:
                        st.error("Detection failed. Please ensure the selected model type is available.")
                
                except Exception as e:
                    st.error(f"An error occurred during classification: {str(e)}")
                    st.exception(e)
        else:
            st.error("No pre-trained models available. Please add model files to the 'trained_models' directory.")
    else:
        st.warning("Please enter some text to classify.")

# Information about the models
with st.expander("About the Models"):
    st.markdown("""
    ## Model Types
    - **Standard Random Forest**: Basic Random Forest classifier
    - **Enhanced Random Forest**: Advanced Random Forest combining Spectral Co-Clustering and Reduced Error Pruning
    
    ## Implementation Details
    These models were pre-trained using implementations from train_model_algorithm.py:
    - The Standard RF model uses a basic Random Forest implementation
    - The Enhanced RF model uses an advanced approach with co-clustering for feature selection and tree pruning
    """)

# Footer
st.markdown("---")
st.markdown("SMS Fraud Detection System | Random Forest Enhancements") 