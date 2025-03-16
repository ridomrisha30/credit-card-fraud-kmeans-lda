# Credit Card Fraud Detection Using K-Means & LDA

##  Project Overview

This project detects fraudulent credit card transactions using K-Means Clustering and Fisher’s Linear Discriminant Analysis (LDA). We leverage unsupervised learning to identify anomalies and enhance fraud classification with supervised learning techniques.

## Dataset

* Source: Real-world credit card transaction dataset (creditcard.csv)
* Key Features Used: Amount, Time, Class
* Preprocessing: Applied Min-Max Scaling for better clustering and classification.

## Methodology

1. Data Preprocessing
* Selected Amount and Time as key features.
* Scaled features using Min-Max Scaling.

2. K-Means Clustering (Unsupervised Learning)
* Applied Elbow Method to determine the optimal number of clusters.
* Used K-Means with n_clusters=2 to separate fraud and non-fraud transactions.
* Visualized clusters and centroids.

3. Fisher’s LDA for Classification (Supervised Learning)
* Split dataset into train (80%) and test (20%).
* Trained LDA model on transaction data.
* Evaluated model performance with accuracy score.
* Visualized LDA decision boundary.

  ## Results

* Effective fraud detection using combined clustering and classification.
* Clear decision boundaries for fraud and non-fraud transactions.
* Model Accuracy: Reported using Fisher’s LDA.

  
## Technologies Used

* Python
* Pandas, NumPy (Data Processing)
* Scikit-Learn (K-Means, LDA, Accuracy Evaluation)
* Matplotlib, Seaborn (Visualization)

  ## Future Improvements

* Incorporate additional features for better fraud detection.
* Experiment with other anomaly detection methods.
* Deploy as a real-time fraud detection system.

  
