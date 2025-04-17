import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet

# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.chdir('/content/drive/My Drive/THESIS_DATASET')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# --- Evaluation ---
print("Standard Random Forest:")
print(classification_report(y_test, y_pred))

# --- CFCN Implementation ---
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

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
y_pred_probs, feature_contributions = cfc_model(X_test_tensor)
y_pred_tensor = torch.argmax(y_pred_probs, axis=1)
print("\nContextual Feature Contribution Network (CFCN) Results:")
print(classification_report(y_test_tensor, y_pred_tensor))

# Analyze Feature Contributions
avg_contributions = torch.mean(feature_contributions, axis=0).detach().numpy()
feature_names = vectorizer.get_feature_names_out()
contribution_df = pd.DataFrame({'Feature': feature_names, 'Contribution': avg_contributions})
contribution_df = contribution_df.sort_values(by='Contribution', ascending=False).head(10)
print("\nTop 10 Contributing Features from CFCN:")
print(contribution_df)

# Visualize Contribution
plt.figure(figsize=(10, 6))
sns.barplot(x='Contribution', y='Feature', data=contribution_df)
plt.title('Top 10 Features by Contribution - CFCN')
plt.tight_layout()
plt.savefig('cfcn_feature_contribution.png')
plt.show()
