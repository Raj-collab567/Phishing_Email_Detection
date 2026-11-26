# PROJECT REPORT
## Phishing Email Detection Using Machine Learning

### 1. Abstract
Phishing is a social-engineering technique in which an attacker attempts to trick a recipient into disclosing sensitive information or taking an unsafe action. This project demonstrates a machine-learning approach to screening email text and classifying it as Phishing or Safe. The implementation uses TF-IDF text representation, a small set of URL and urgency indicators, and a Logistic Regression classifier. A Streamlit interface allows a user to paste email content and view the model's prediction and evaluation metrics.

### 2. Problem Statement
Users may receive deceptive emails that imitate trusted organizations or create urgency. The aim is to build a basic tool that analyzes email text and highlights messages that resemble phishing examples.

### 3. Objectives
- Prepare labelled email text for machine-learning experiments.
- Convert text into numerical TF-IDF features.
- Extract simple URL and urgency-related indicators.
- Train a binary Logistic Regression classifier.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Provide a simple interface for testing email text.

### 4. Tools and Technologies
Python, Pandas, NumPy, Scikit-learn, TF-IDF, Logistic Regression, Joblib, and Streamlit.

### 5. Methodology
The labelled CSV is checked for required columns and valid labels. Data is divided into stratified training and test subsets. TF-IDF represents words and two-word phrases numerically. Additional indicators count URLs, measure URL text length, detect IP-address URLs and common URL shorteners, and count selected urgency-related terms. These features are combined and passed to a class-balanced Logistic Regression model. The trained pipeline is saved and used by the Streamlit application.

### 6. System Modules
1. Dataset module: reads labelled email examples.
2. Preprocessing and feature extraction module: creates text and URL indicators.
3. Training module: splits data and trains the classifier.
4. Evaluation module: reports classification metrics and confusion matrix.
5. User interface module: accepts email text and displays prediction scores.

### 7. Expected Output
The application displays one of two labels, Phishing or Safe, with the model's class scores. It also displays evaluation metrics generated from the held-out test subset.

### 8. Limitations
The included CSV is an illustrative demo dataset, not a representative real-world corpus. Therefore, its metrics are not evidence of production performance. The model only examines pasted text and simple indicators; it does not verify sender identity, inspect attachments, or establish whether a destination website is malicious. A Safe result is not a guarantee.

### 9. Future Scope
The system can be improved by using a large, licensed and representative dataset, analyzing email headers and sender domains, adding safe URL reputation checks, evaluating on independent datasets, and monitoring false positives and false negatives.

### 10. Conclusion
This project demonstrates an end-to-end machine-learning workflow for basic phishing-email text screening, from labelled data and feature extraction to model evaluation and a user-facing web interface. It is a learning prototype and requires stronger data and independent validation before any operational use.

### Submission checklist
- [ ] Replace the illustrative dataset with an approved real dataset if required by the internship.
- [ ] Run training and record the actual output metrics.
- [ ] Test the Streamlit application with sample messages.
- [ ] Add your name, internship details, institute/company details, and screenshots where required.
- [ ] Be prepared to explain TF-IDF, Logistic Regression, precision, recall, F1-score, and false positives/negatives.
