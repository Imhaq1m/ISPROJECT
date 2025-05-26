
# =============================
# 1. Import Required Libraries
# =============================
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# =============================
# 2. Load and Prepare Dataset
# =============================
df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].str.replace('\r\n', ' ')  # Clean line breaks

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]  # Remove stopwords and stem
    return ' '.join(words)

# Apply preprocessing to all emails
df['cleaned_text'] = df['text'].apply(preprocess_text)

# =============================
# 3. Vectorize Text Data
# =============================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label_num']  # Target labels (0 = ham, 1 = spam)

# =============================
# 4. Train the Model on Full Dataset
# =============================
model = RandomForestClassifier(n_jobs=-1)
model.fit(X, y)  # Train on full data

# =============================
# 5. Predict All Emails
# =============================
all_predictions = model.predict(X)  # Predict all emails

# Add predictions back to DataFrame
df['predicted_label_num'] = all_predictions
df['predicted_label'] = df['predicted_label_num'].map({0: 'ham', 1: 'spam'})

# =============================
# 6. Show Some Results
# =============================
# View a few random predictions
print("Predictions for sample emails:")
print(df[['label', 'predicted_label', 'text']].sample(10))

# Optional: Save updated DataFrame with predictions to CSV
df.to_csv('spam_ham_predictions.csv', index=False)
print("\nAll predictions saved to 'spam_ham_predictions.csv'")
#------------------------------------------------------------------------------------------#