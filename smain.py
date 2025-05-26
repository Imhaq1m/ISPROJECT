import string
import math
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
    cleaned_words = [stemmer.stem(word)
                     for word in words if word not in stopwords_set]
    # Join words back into a sentence
    return ' '.join(cleaned_words)


# Apply preprocessing to all emails
print("Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Create bag-of-words features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])  # Features
y = df['label_num']  # Labels (0 = ham, 1 = spam)

# Split into training and test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    # Increase max_iter for convergence
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
}

# List to store model performance
performance = []

for name, model in models.items():
    # print(f"Evaluating: {name}")
    print(f"Evaluating: {name} with cross-validation")

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # Record cross-validated accuracy
    accuracy_cv = cv_scores.mean()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save to performance list
    performance.append({
        'Model': name,
        'Accuracy': accuracy,
        'CV Accuracy': accuracy_cv,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    print(f"Evaluating {name} completed")

# Convert to DataFrame
df_performance = pd.DataFrame(performance)

# Set up the plot
metrics = ['CV Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = df_performance['Model']

n_metrics = len(metrics)
x = np.arange(len(model_names))  # label locations for models
width = 0.2  # width of each bar
group_width = n_metrics * width  # total width of a metric group
x_positions = x * (group_width + 0.5)  # add spacing between groups

fig, ax = plt.subplots(figsize=(16, 8))

# Create bars for each metric
rects = []
for i, metric in enumerate(metrics):
    rects.append(
        ax.bar(x_positions + i*width,
               df_performance[metric], width, label=metric)
    )

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, and F1 Score')
ax.set_xticks(x_positions + (group_width / 2))
ax.set_xticklabels(model_names)
ax.legend()

# Rotate x-labels for readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(0.6, 1.0)  # Zoom in on high scores
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Number of models
n_models = len(models)
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

# Set up the figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()  # Flatten to easily loop through

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f'{name} CM')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

# Hide any extra subplots
for idx in range(len(models), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.suptitle('Confusion Matrices for All Models', y=1.02)
plt.show()


# Plot Feature Importances (Only for Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:top_n]],
            y=feature_names[indices[:top_n]], palette="viridis")
plt.title(f'Top {top_n} Most Important Words (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Words')
plt.tight_layout()
plt.show()
