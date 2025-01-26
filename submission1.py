# SAMPLING ASSIGNMENT
# Author: Preet_Chaudhary_3C75_103303748
# Date: 26 January 2025

import pandas as pd

# Load the dataset
dataset = pd.read_csv("Creditcard_data.csv")
print(dataset.head(), dataset.info(), dataset.describe())

# Analyze the distribution of the target class
target_distribution = dataset['Class'].value_counts(normalize=True) * 100
print("Class Distribution:\n", target_distribution)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Separate features and labels
features = dataset.drop(columns=['Class'])
labels = dataset['Class']

# Apply SMOTE for balancing the dataset
smote_sampler = SMOTE(random_state=42)
balanced_features, balanced_labels = smote_sampler.fit_resample(features, labels)

# Validate the new class distribution
balanced_distribution = balanced_labels.value_counts(normalize=True) * 100
print("Balanced Class Distribution:\n", balanced_distribution)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    balanced_features, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
)

# Sampling strategies
def random_sampling(features, labels, n_samples):
    indices = np.random.choice(range(len(features)), size=n_samples, replace=False)
    return features.iloc[indices], labels.iloc[indices]

def stratified_sampling(features, labels, n_samples):
    return train_test_split(features, labels, train_size=n_samples, random_state=42, stratify=labels)

# Define sample sizes
sample_counts = [100, 200, 300, 400, 500]

# Define models to evaluate
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Initialize a dictionary to store results
evaluation_results = {}

# Iterate through each sample size
for idx, sample_count in enumerate(sample_counts):
    evaluation_results[f"Sample Size {idx + 1}"] = {}

    # Apply random sampling
    sampled_features, sampled_labels = random_sampling(X_train, y_train, sample_count)

    # Train and evaluate each model
    for classifier_name, classifier in classifiers.items():
        # Train the classifier
        classifier.fit(sampled_features, sampled_labels)

        # Predict on the test set
        predictions = classifier.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, predictions)
        evaluation_results[f"Sample Size {idx + 1}"][classifier_name] = accuracy

# Convert results to a DataFrame for better visualization
evaluation_df = pd.DataFrame(evaluation_results)
print("Evaluation Results:\n", evaluation_df)
