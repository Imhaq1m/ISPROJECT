# Import standard libraries
import string
import math
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Import specific tools from sklearn and nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset (spam vs ham emails)
df = pd.read_csv('spam_ham_dataset.csv')

# Clean up line breaks (\r\n) in email text
df['text'] = df['text'].str.replace('\r\n', ' ')

# Initialize tools for text preprocessing
stemmer = PorterStemmer()  # Used to shorten words like "running" → "run"
# List of common English words to remove
stopwords_set = set(stopwords.words('english'))


# Function to clean and simplify text
def preprocess_text(text):
    # Convert all characters to lowercase
    text = text.lower()
    # Remove punctuation marks (like commas, periods)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split text into individual words
    words = text.split()
    # Remove unimportant words and shorten remaining ones
    cleaned_words = [stemmer.stem(word)
                     for word in words if word not in stopwords_set]
    # Join cleaned words back into a sentence
    return ' '.join(cleaned_words)


# Apply preprocessing to all emails
print("Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Convert cleaned text into numerical features using Bag-of-Words
vectorizer = CountVectorizer()
# X = input features (word counts)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label_num']  # y = output labels (0 = ham, 1 = spam)

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define models we want to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
}

# List to store performance metrics of each model
performance = []

# Loop through each model and evaluate its performance
for name, model in models.items():
    print(f"Evaluating: {name} with cross-validation")

    # Use cross-validation to estimate general performance
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy_cv = cv_scores.mean()  # Average accuracy across folds

    # Train the model on training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save metrics for later use
    performance.append({
        'Model': name,
        'Accuracy': accuracy,
        'CV Accuracy': accuracy_cv,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print(f"Evaluating {name} completed")

# Convert performance list to DataFrame for easier plotting
df_performance = pd.DataFrame(performance)

# Set up bar chart comparing model performance
metrics = ['CV Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = df_performance['Model']

# Set positions for bars
x = np.arange(len(model_names))  # One position per model
width = 0.2  # Width of each bar
group_width = len(metrics) * width  # Total width of all metrics for one model
x_positions = x * (group_width + 0.5)  # Add space between models

# Create figure and axis for plot
fig, ax = plt.subplots(figsize=(16, 8))

# Draw bars for each metric
rects = []
for i, metric in enumerate(metrics):
    rects.append(
        ax.bar(x_positions + i*width,
               df_performance[metric], width, label=metric)
    )

# Add labels and titles
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, and F1 Score')
ax.set_xticks(x_positions + group_width / 2)
ax.set_xticklabels(model_names)
ax.legend()

# Rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(0.6, 1.0)  # Show scores from 0.6 to 1.0
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for clarity

# Add value labels on top of bars
for bar_group in rects:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset from bar top
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Display the final bar chart
plt.show()


# Plot confusion matrices for all models
n_models = len(models)
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

# Set up subplots (grid of plots)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()  # Flatten array for easy looping

# Loop through models and plot confusion matrix
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot heatmap for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f'{name} CM')  # Title for subplot
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

# Hide extra subplots if number of models < total subplots
for idx in range(len(models), len(axes)):
    fig.delaxes(axes[idx])

# Final layout adjustments
plt.tight_layout()
plt.suptitle('Confusion Matrices for All Models', y=1.02)
plt.show()


# Plot feature importances (Only for Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Sort feature names by importance
indices = np.argsort(importances)[::-1]  # Descending order
top_n = 20  # Top 20 most important words

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:top_n]],
            y=feature_names[indices[:top_n]], palette="viridis")
plt.title(f'Top {top_n} Most Important Words (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Words')
plt.tight_layout()
plt.show()
