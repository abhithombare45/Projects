### train.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load Data
df = pd.read_csv("data/sentiment_data.csv")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("Training completed.")

### test.py
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load Model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
df = pd.read_csv("data/sentiment_data.csv")
X = vectorizer.transform(df['text'])
y = df['sentiment']

# Predictions
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))

### visualize.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib

# Load Data & Model
df = pd.read_csv("data/sentiment_data.csv")
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
X = vectorizer.transform(df['text'])
y = df['sentiment']
y_pred = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("results/confusion_matrix.png")
plt.show()

### clustering.py
from sklearn.cluster import KMeans
import pandas as pd
import joblib

# Load Data
vectorizer = joblib.load("models/vectorizer.pkl")
df = pd.read_csv("data/sentiment_data.csv")
X = vectorizer.transform(df['text'])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
df.to_csv("results/clustered_data.csv", index=False)
print("Clustering completed.")

### final.py
import os
os.system("python train.py")
os.system("python test.py")
os.system("python visualize.py")
os.system("python clustering.py")
print("Pipeline execution completed.")


