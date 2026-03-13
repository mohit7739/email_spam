"""
train.py — Run this ONCE to train the spam model on your dataset.

Usage:
    python train.py

Output:
    spam_model.pkl   — trained Naive Bayes classifier
    vectorizer.pkl   — fitted TF-IDF vectorizer
"""

import os
import sys
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Paths (flat structure: train.py is inside backend/) ───────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, '..', 'data', 'spam.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'spam_model.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')


# ── Text cleaner (inline — no import needed) ──────────────────────────────────
def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    print(f"📂 Loading dataset from: {path}")

    try:
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')

        # Kaggle format: v1=label, v2=text
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].copy()
            df.columns = ['label', 'text']
        elif 'label' in df.columns and 'text' in df.columns:
            df = df[['label', 'text']].copy()
        else:
            raise ValueError("CSV must have columns: 'v1','v2' OR 'label','text'")

        df.dropna(subset=['label', 'text'], inplace=True)
        df['label'] = df['label'].str.lower().str.strip()

        print(f"✅ Loaded {len(df)} samples")
        print(f"   Spam: {(df['label'] == 'spam').sum()}")
        print(f"   Ham:  {(df['label'] == 'ham').sum()}")
        return df

    except FileNotFoundError:
        print(f"\n❌ Dataset not found at: {path}")
        print("   Place spam.csv inside the data/ folder")
        sys.exit(1)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    print("\n🔧 Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Split → Train: {len(X_train)}, Test: {len(X_test)}")

    print("🔢 Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  stop_words='english', min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("🤖 Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred   = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTOR_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n💾 spam_model.pkl saved")
    print(f"💾 vectorizer.pkl saved")
    print("🚀 Done! Now run: python app.py")


if __name__ == '__main__':
    df = load_dataset(DATA_PATH)
    train(df)