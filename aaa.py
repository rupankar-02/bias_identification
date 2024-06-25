

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertModel
import torch
from joblib import dump


# Load the dataset
data = pd.read_csv(r'BIAS_DATASET.csv')
print(data.info())
print("Missing values per column:")
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Check for and drop duplicate rows
data = data.drop_duplicates()

# Visualize the distribution of the labels
label_counts = data['LABEL'].value_counts()
print(label_counts)
colors = ['orange', 'green', 'red', 'blue']
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Categories in the LABEL Column')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Redownload the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_and_extract_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply the BERT embedding function to each text entry
print("Extracting BERT embeddings...")
data['embeddings'] = data['TEXT'].apply(tokenize_and_extract_embeddings)

# Convert embeddings and labels to numpy arrays
X = np.vstack(data['embeddings'].values)
y = data['LABEL'].values

# Apply SMOTE to balance the dataset
print("Applying SMOTE...")
smote = SMOTE(sampling_strategy={'non_bias': 350, 'country_bias': 364, 'gender_bias': 337, 'religion_bias': 301}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualize the distribution of the labels after SMOTE
resampled_label_counts = pd.Series(y_resampled).value_counts()
print(resampled_label_counts)
plt.figure(figsize=(8, 6))
resampled_label_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Categories in the LABEL Column After SMOTE')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Train-test split
print("Performing train-test split...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}

from joblib import dump
# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Results for {name}:")
    dump(clf, f'{name}.pkl')
    print(classification_report(y_test, y_pred))
    print("\n" + "="*60 + "\n")