from pathlib import Path
import argparse
import pandas as pd
import joblib
from train_model import make_features

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--urls", default="")
    args = p.parse_args()
    model = joblib.load(BASE/"phishing_model.joblib")
    sample = pd.DataFrame([{"subject":args.subject,"body":args.body,"urls":args.urls}])
    pred = model.predict(make_features(sample))[0]
    probs = model.predict_proba(make_features(sample))[0]
    classes = list(model.named_steps["classifier"].classes_)
    conf = probs[classes.index(pred)]
    print("Phishing Email Detection")
    print("-"*34)
    print(f"Prediction : {pred.title()}")
    print(f"Confidence : {conf:.1%}")

if __name__ == "__main__":
    main()
