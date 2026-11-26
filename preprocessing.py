import re
import numpy as np
from sklearn.preprocessing import FunctionTransformer

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)

def extract_url_features(texts):
    """Extract simple, explainable URL/text indicators for each email."""
    features = []
    for value in texts:
        text = str(value or "")
        urls = URL_RE.findall(text)
        url_count = len(urls)
        url_chars = sum(len(u) for u in urls)
        has_ip = int(bool(re.search(r"https?://(?:\d{1,3}\.){3}\d{1,3}", text, re.I)))
        shortened = int(bool(re.search(r"bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly", text, re.I)))
        urgency_words = len(re.findall(
            r"\b(urgent|immediately|verify|suspended|expire|locked|act now|password|claim)\b",
            text, re.I
        ))
        features.append([url_count, url_chars, has_ip, shortened, urgency_words])
    return np.asarray(features, dtype=float)
