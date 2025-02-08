1. Project Overview

The Sentiment Recognition project aims to classify text data into sentiment categories using machine learning techniques. The project follows a structured flow to ensure clarity and efficiency.

2. Directory structure 

sentiment_recognition/
│── data/
│   ├── sentiment_data.csv  # Raw dataset
│── models/
│   ├── sentiment_model.pkl  # Trained sentiment model
│   ├── vectorizer.pkl  # TF-IDF vectorizer
│── results/
│   ├── confusion_matrix.png  # Model performance visualization
│   ├── clustered_data.csv  # Sentiment clusters
│── scripts/
│   ├── train.py  # Model training
│   ├── test.py  # Model evaluation
│   ├── visualize.py  # Result visualization
│   ├── clustering.py  # Sentiment clustering
│   ├── final.py  # Full pipeline execution



3. The project follows these steps:

Data Collection: Load and preprocess sentiment data.

Feature Extraction: Convert text data into numerical vectors using TF-IDF.

Model Training: Train a Naive Bayes classifier.

Model Evaluation: Test accuracy and generate a confusion matrix.

Clustering: Use K-Means to analyze sentiment groups.

Visualization & Results: Save evaluation results and clustered data.


4. Dependancies
pip install pandas numpy matplotlib seaborn scikit-learn joblib

5. Data Description

sentiment_data.csv (Raw data)

text: The text content for sentiment analysis.

sentiment: The corresponding sentiment label (e.g., Positive, Negative, Neutral).


6.  Scripts Breakdown

train.py - Loads data, preprocesses it, and trains a Naive Bayes model.

test.py - Evaluates the trained model on a test dataset and calculates accuracy.

visualize.py - Generates and saves a confusion matrix for performance assessment.

clustering.py - Uses K-Means to group similar sentiments.

final.py - Runs the entire workflow from data loading to clustering.




7. Execution : 
python scripts/final.py


8. Outputs

Model & Vectorizer: Stored in models/

Evaluation & Clustering Results: Stored in results/






