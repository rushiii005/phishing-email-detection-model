from pathlib import Path
import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WORDS = {"urgent","verify","verification","password","login","account","suspended","locked","confirm","claim","prize","payment","security","refund","credential","immediately","expire","expired","recovery","bank","wallet"}

def make_features(df):
    x = df.copy()
    x["full_text"] = (x["subject"].fillna("") + " " + x["body"].fillna("") + " " + x["urls"].fillna("")).str.strip()
    x["url_count"] = x["urls"].fillna("").map(lambda s: len(URL_RE.findall(s)))
    x["suspicious_url"] = x["urls"].fillna("").str.contains(r"(login|verify|secure|account|billing|confirm|claim|reset|bank|wallet)", case=False, regex=True).astype(int)
    x["keyword_count"] = x["full_text"].str.lower().map(lambda s: sum(1 for w in WORDS if w in s))
    x["urgent_language"] = x["full_text"].str.lower().str.contains(r"\burgent\b|\bimmediately\b|\bfinal warning\b|\btoday\b", regex=True).astype(int)
    x["http_url"] = x["urls"].fillna("").str.startswith("http://").astype(int)
    return x

def make_model():
    prep = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True), "full_text"),
        ("num", "passthrough", ["url_count","suspicious_url","keyword_count","urgent_language","http_url"])
    ])
    return Pipeline([("features", prep), ("classifier", LogisticRegression(max_iter=1500, random_state=42))])

def main():
    df = pd.read_csv(BASE/"emails.csv")
    X = make_features(df)
    y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = make_model()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, labels=["safe","phishing"], target_names=["Safe","Phishing"], digits=3, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=["safe","phishing"])
    joblib.dump(model, BASE/"phishing_model.joblib")
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(cm, display_labels=["Safe","Phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Phishing Email Detection")
    fig.tight_layout()
    fig.savefig(BASE/"confusion_matrix.png", dpi=150)
    plt.close(fig)
    (BASE/"metrics.txt").write_text(f"Dataset size: {len(df)}\nTraining samples: {len(Xtr)}\nTest samples: {len(Xte)}\nAccuracy: {acc:.3f}\n\n{report}", encoding="utf-8")
    print("Training completed successfully.")
    print(f"Dataset size: {len(df)}")
    print(f"Training samples: {len(Xtr)}")
    print(f"Test samples: {len(Xte)}")
    print(f"Accuracy: {acc:.3f}\n")
    print(report)
    print("Generated: phishing_model.joblib, metrics.txt, confusion_matrix.png")

if __name__ == "__main__":
    main()
