# SMS Fraud Detection System

This application uses an enhanced Random Forest algorithm to detect fraudulent SMS messages. The system incorporates several advanced techniques:

1. Spectral Co-Clustering for feature selection
2. Reduced Error Pruning for decision trees
3. Contextual Feature Contribution Network (CFCN) for analyzing feature contributions

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have at least one dataset file (CSV format) in the `datasets` directory

## Dataset Format

The system expects datasets with the following characteristics:
- CSV files with at least two columns
- Columns typically named 'LABEL' and 'TEXT' (or similar)
- Binary classification labels (Fraud/Not Fraud, Spam/Ham, etc.)

Several datasets are included in the `datasets` directory for your convenience.

## Running the Application

Start the Streamlit application by running:
```
streamlit run app.py
```

This will open the web application in your default browser.

## Using the System

1. **Select a dataset** from the dropdown in the sidebar
2. **Train models** by clicking the "Train Models" button in the sidebar
3. **Select a model type** from the dropdown (Standard, Pruned, Co-Clustered, or Combined)
4. **Enter the SMS message** text in the text area
5. **Click "Classify"** to analyze the message
6. The system will display whether the message is fraudulent or not

## Model Types

- **Standard**: Basic Random Forest classifier
- **Pruned**: Random Forest with Reduced Error Pruning
- **Co-Clustered**: Random Forest with Spectral Co-Clustering for feature selection
- **Combined**: Uses both Co-Clustering and Pruning techniques

Each model offers different trade-offs between accuracy, efficiency, and interpretability.

## Model Training and Storage

The application automatically:
1. Trains the selected models on the chosen dataset
2. Saves the trained models to the `trained_models` directory
3. Loads the appropriate model when making predictions
4. Displays training metrics (accuracy, F1 score, and training time)

## Requirements

- Python 3.7 or higher
- Streamlit
- PyTorch
- scikit-learn
- pandas
- numpy
- joblib
- chardet (for dataset encoding detection) 