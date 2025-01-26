Overview:-

Credit card fraud detection is a critical task where imbalanced datasets can hinder the performance of machine learning models. This project involves:
1. Balancing the dataset using Synthetic MinorityOversampling Technique (SMOTE).
2. Creating five samples of different sizes using various sampling techniques.
3. Training five machine learning models on these samples.
4. Comparing model performance and identifying the best sampling technique for each model.


Dataset:-

The dataset used is Creditcard_data.csv(which has been provided), containing:
1. Features (V1 to V28): Principal components of transaction data.
2. Time: Seconds elapsed between transactions.
3. Amount: Transaction amount.
4. Class: Fraud label (0 for legitimate, 1 for fraud).

Class Imbalance:
The dataset is highly imbalanced with-
1. Class 0: 98.83% of transactions (legitimate).
2. Class 1: 1.17% of transactions (fraudulent).


Sampling Techniques used are:
1. Random Sampling: Randomly selects a subset of data.
2. Stratified Sampling: Ensures class proportions are maintained.
3. Systematic Sampling: Selects every nth sample.
4. Cluster Sampling: Divides the dataset into clusters and selects entire clusters.
5. Oversampling: Increases minority class instances further.

Machine Learning Models used are:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Gradient Boosting


Observations:-

1. Logistic Regression performs best with Stratified Sampling and Oversampling.
2. Decision Tree works best with Stratified Sampling.
3. Random forest gives best result with Oversampling.
4. SVM performs best with Oversampling.
5. Gradient Boosting worked best with Stratified Sampling.