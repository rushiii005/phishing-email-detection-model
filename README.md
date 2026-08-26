# Phishing Email Detection Model

A Scikit-learn mini project that classifies emails as **Phishing** or **Safe**.

## Features
- Labeled phishing/safe email dataset
- TF-IDF text features from subject, body, and URLs
- Extra URL/security features
- Logistic Regression classifier
- Accuracy and classification report
- Confusion matrix
- Command-line prediction for new emails

## Run
```bash
pip install -r requirements.txt
python train_model.py
```

Then test a new email:
```bash
python predict.py --subject "Urgent account verification" --body "Your account will be suspended. Verify your login immediately." --urls "https://secure-account-check.example/verify"
```

## Dataset
The included CSV is a small synthetic educational dataset so the project is self-contained and reproducible. It is suitable for demonstrating the internship task, but its accuracy should not be treated as production-level phishing detection.

## Files
`emails.csv` - dataset  
`train_model.py` - training/evaluation  
`predict.py` - prediction  
`requirements.txt` - dependencies  
`README.md` - documentation
