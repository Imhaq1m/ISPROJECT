import string
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')


# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Clean up line breaks in email text
df['text'] = df['text'].str.replace('\r\n', ' ')

# Initialize stemmer and list of English stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize (split into words)
    words = text.split()
    # Remove stopwords and apply stemming
    cleaned_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    # Join words back into a sentence
    return ' '.join(cleaned_words)

# Apply preprocessing to all emails
print("Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Create bag-of-words features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text']).toarray()  # Features
y = df['label_num']  # Labels (0 = ham, 1 = spam)

# Split into training and test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
print("Training model...")
model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Detailed performance report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Feature Importances
# Get feature names from CountVectorizer
feature_names = vectorizer.get_feature_names_out()

# Get feature importances from Random Forest
importances = model.feature_importances_

# Sort them by importance
indices = np.argsort(importances)[::-1]

# Plot top N features
top_n = 20
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices[:top_n]], y=feature_names[indices[:top_n]], palette="viridis")
plt.title(f'Top {top_n} Most Important Words')
plt.xlabel('Importance')
plt.ylabel('Words')
plt.tight_layout()
plt.show()