import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
   text = str(text).lower()
   text = re.sub(r"http\S+|www\S+", "", text)
   text = text.translate(str.maketrans("", "", string.punctuation))
   text = re.sub(r"\d+", "", text)
   text = re.sub(r"\s+", " ", text).strip()
   return text


def preprocess_dataset(df):
   df["clean_text"] = df["text"].apply(clean_text)
   return df


def extract_features(df):
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(df["clean_text"])
   y = df["label"]
   return X, y, vectorizer