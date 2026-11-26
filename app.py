from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Phishing Email Detection")
st.caption("Machine-learning based email text screening using TF-IDF and URL indicators.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("Model file not found. Train the model first using: python src/train_model.py")
    st.stop()

left, right = st.columns([2, 1])
with left:
    email_text = st.text_area(
        "Paste email content",
        height=240,
        placeholder="Paste the email subject and body here. Do not paste real passwords, OTPs, or confidential information."
    )
    check = st.button("Analyze Email", type="primary", use_container_width=True)

with right:
    st.subheader("How it works")
    st.write("The model combines TF-IDF text features with simple URL and urgency indicators.")
    st.info("This is a screening aid, not a guarantee. Never open suspicious links based only on this result.")

if check:
    if not email_text.strip():
        st.warning("Please paste email content before analysis.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100

        if prediction == "Phishing":
            st.error(f"⚠️ Likely Phishing — model confidence: {confidence:.1f}%")
            st.write("Do not click links, download attachments, or share credentials. Verify the sender through an independent official channel.")
        else:
            st.success(f"✅ Classified as Safe — model confidence: {confidence:.1f}%")
            st.write("Still verify the sender and links. A 'Safe' prediction does not guarantee that an email is legitimate.")

        st.subheader("Model scores")
        st.dataframe(pd.DataFrame({
            "Class": classes,
            "Model score": [f"{p*100:.1f}%" for p in probabilities]
        }), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Evaluation on the held-out demo test split")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "F1 (Phishing)"],
                               ["accuracy", "precision_phishing", "recall_phishing", "f1_phishing"]):
        col.metric(label, f"{metrics[key]*100:.1f}%")
    st.caption("These scores are from the included illustrative sample dataset only. They should not be presented as real-world performance.")
    st.write("Confusion matrix (rows = actual, columns = predicted; order: Phishing, Safe)")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                              index=["Actual Phishing", "Actual Safe"],
                              columns=["Predicted Phishing", "Predicted Safe"]),
                 use_container_width=True)
else:
    st.info("Metrics will appear after training the model.")
