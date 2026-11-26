# Phishing Email Detection Using Machine Learning

A Python + Scikit-learn project that classifies pasted email text as **Phishing** or **Safe**.
It combines TF-IDF text features with simple URL/urgency indicators and provides a Streamlit web interface.

## Important dataset note
`dataset/sample_emails.csv` is a small **illustrative demo dataset created for this runnable starter project**. It is not a representative real-world phishing corpus. The included evaluation metrics are only a software demonstration and must not be described as validated real-world accuracy. For a serious internship evaluation, replace or supplement it with a properly licensed, representative, labelled email dataset, remove duplicates, check class balance, and evaluate on a time-separated or independently sourced test set.

## Requirements
- Python 3.10 or newer
- VS Code (optional)
- Internet connection for installing Python packages

## Run on Windows
Open a terminal in this project folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Run on macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

The app normally opens at `http://localhost:8501`.

## Dataset format
Replace `dataset/sample_emails.csv` with your own labelled CSV using these exact columns:

| email_text | label |
|---|---|
| Email subject and body text | Phishing |
| Email subject and body text | Safe |

Labels must be exactly `Phishing` or `Safe`. Keep personal information, real credentials, and confidential messages out of the dataset.

## Project structure
```text
Phishing_Email_Detection/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── sample_emails.csv
├── models/
│   ├── phishing_model.joblib  # created after training
│   └── metrics.json           # created after training
└── src/
    ├── __init__.py
    ├── preprocessing.py
    └── train_model.py
```

## Methodology
1. Load and validate labelled email text.
2. Split into stratified training and test sets.
3. Convert text into TF-IDF unigram/bigram features.
4. Extract simple indicators: URL count, URL character count, IP-address URL indicator, URL-shortener indicator, and urgency-related word count.
5. Train a class-balanced Logistic Regression classifier.
6. Evaluate with accuracy, phishing precision, phishing recall, F1-score, and confusion matrix.
7. Save the trained model with Joblib and load it in Streamlit.

## Limitations and responsible use
- This is a text classifier, not a secure email gateway or malware scanner.
- It does not inspect attachments, authenticate senders, resolve URLs, or browse links.
- Attackers can evade text-based detection; legitimate emails can also be flagged.
- Model confidence is not a calibrated probability of safety.
- Never enter passwords, OTPs, private emails, or confidential company data into the demo.
- Do not click a link merely because the model says an email is Safe.

## Suggested improvements
- Train and validate on a large, licensed public dataset.
- Add sender-domain and email-header analysis.
- Add robust URL parsing and domain reputation checks using safe, approved services.
- Use cross-validation, threshold tuning, and an independent test dataset.
- Add model versioning, experiment tracking, and explainability.
