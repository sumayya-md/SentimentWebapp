import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure stopwords are available
nltk.download('stopwords')

# 🔹 Load your labeled dataset
df = pd.read_csv("reddit_data_labeled.csv")
df = df.dropna(subset=["sentiment"])

# Clean labels
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
mapping = {
    "negative": "negative",
    "neagtive": "negative",   # fix typo
    "neg": "negative",
    "positive": "positive",
    "pos": "positive",
    "neutral": "neutral",
}
df["sentiment"] = df["sentiment"].map(mapping).fillna("neutral")

print("📊 Original class distribution:\n", df["sentiment"].value_counts())

# -------------------------------
# Balance dataset via upsampling
# -------------------------------
df_majority = df[df.sentiment == "negative"]
df_minority_pos = df[df.sentiment == "positive"]
df_minority_neu = df[df.sentiment == "neutral"]

df_pos_upsampled = resample(df_minority_pos, replace=True,
                            n_samples=len(df_majority), random_state=42)
df_neu_upsampled = resample(df_minority_neu, replace=True,
                            n_samples=len(df_majority), random_state=42)

df_balanced = pd.concat([df_majority, df_pos_upsampled, df_neu_upsampled])
print("📊 Balanced class distribution:\n", df_balanced["sentiment"].value_counts())

# -------------------------------
# Feature preparation
# -------------------------------
df_balanced["combined_text"] = df_balanced["Title"].fillna("") + " " + df_balanced["Text"].fillna("")
X = df_balanced["combined_text"]
y = df_balanced["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words=stopwords.words("english"),
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Logistic Regression
# -------------------------------
log_model = LogisticRegression(max_iter=300, class_weight="balanced")
log_model.fit(X_train_vec, y_train)

y_pred_log = log_model.predict(X_test_vec)
print("\n🔹 Logistic Regression Performance 🔹")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Confusion matrix
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_log, labels=["negative", "neutral", "positive"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# 🔹 Save confusion matrix as PNG
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as confusion_matrix.png")

plt.show()


# -------------------------------
# Naive Bayes (for comparison)
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

print("\n🔹 Naive Bayes Performance 🔹")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -------------------------------
# Cross-validation
# -------------------------------
print("\n📊 Cross-Validation (5-fold) Results")
log_cv = cross_val_score(log_model, vectorizer.transform(X), y, cv=5, scoring="accuracy")
nb_cv = cross_val_score(nb_model, vectorizer.transform(X), y, cv=5, scoring="accuracy")

print("Logistic Regression CV Accuracy:", log_cv.mean())
print("Naive Bayes CV Accuracy:", nb_cv.mean())

# -------------------------------
# Save best model
# -------------------------------
if log_cv.mean() >= nb_cv.mean():
    joblib.dump(log_model, "sentiment_model.pkl")
    print("\n✅ Saved Logistic Regression as best model.")
else:
    joblib.dump(nb_model, "sentiment_model.pkl")
    print("\n✅ Saved Naive Bayes as best model.")

joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Vectorizer saved successfully!")
