import re
import string


def clean_text(text):
   text = str(text).lower()
   text = re.sub(r"http\S+|www\S+", "", text)
   text = text.translate(str.maketrans("", "", string.punctuation))
   text = re.sub(r"\d+", "", text)
   text = re.sub(r"\s+", " ", text).strip()
   return text


