import pickle

def save_model(model, vectorizer):
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)

   with open("vectorizer.pkl", "wb") as f:
       pickle.dump(vectorizer, f)

   print("Model and vectorizer saved successfully.")

def load_model():
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)

   with open("vectorizer.pkl", "rb") as f:
       vectorizer = pickle.load(f)

   return model, vectorizer

def predict_sentiment(text, model, vectorizer, clean_text_function):
   cleaned_text = clean_text_function(text)
   text_vector = vectorizer.transform([cleaned_text])
   prediction = model.predict(text_vector)[0]
   return prediction