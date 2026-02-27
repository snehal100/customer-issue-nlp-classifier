
# Customer Sentiment & Issue Classification Project

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


# 1.Sentiment Distribution

def sentiment_distribution(df):
    if 'sentiment' in df.columns:
        print("\nSentiment Distribution (%):")
        print(df['sentiment'].value_counts(normalize=True).round(3) * 100)

        plt.figure()
        df['sentiment'].value_counts().plot(
            kind='bar',
            color=['green','orange','red','gray']
        )
        plt.title('Pre-labeled Sentiment Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()



# 2.Customer Rating Distribution

def customer_rating_distribution(df):
    if 'customer_rating' in df.columns:
        print("\nCustomer Rating Distribution:")
        print(df['customer_rating'].value_counts().sort_index())

        plt.figure()
        df['customer_rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('Customer Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


# 3.Complaint Registered Analysis

def complaint_registered(df):
    print("\nComplaint Registered (%):")
    print(df['complaint_registered'].value_counts(normalize=True).round(3) * 100)

    df['text_length'] = df['review_text'].astype(str).apply(len)

    print("\nAverage Review Length:",
          df['text_length'].mean().round(1),
          "characters")

    plt.figure()
    df['text_length'].hist(bins=60,
                           color='lightblue',
                           edgecolor='black')
    plt.title('Review Text Length Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.show()



# 4.Assign Issue Type

def assign_issue_type(text):

    if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
        return 'Other/General'

    text = text.lower()

    if any(k in text for k in ['late', 'delay', 'not delivered']):
        return 'Delivery Delay'

    if any(k in text for k in ['rude', 'bad behavior', 'unprofessional']):
        return 'Delivery Person Behavior'

    if any(k in text for k in ['damaged', 'broken', 'wrong item']):
        return 'Package Condition Issue'

    if any(k in text for k in ['tracking', 'app issue', 'payment failed']):
        return 'Technical/Tracking Issue'

    if any(k in text for k in ['great', 'excellent', 'happy']):
        return 'Positive Feedback'

    return 'Other/General'



# 5.Issue Type Distribution Graph

def plot_issue_distribution(df):

    print("\nIssue Type Distribution:")
    print(df['issue_type'].value_counts())

    plt.figure()
    sns.countplot(
        y='issue_type',
        data=df,
        order=df['issue_type'].value_counts().index,
        palette='Set2'
    )
    plt.title('Distribution of Assigned Issue Types')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Issue Type')
    plt.tight_layout()
    plt.show()



# 6.Clean Text

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text.strip()



# 7.Train & Evaluate Model (Confusion Matrix = 5th Graph)

def train_model(df):

    X = df['clean_text']
    y = df['issue_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
