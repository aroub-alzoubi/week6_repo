from flask import Flask, render_template, request
import pickle
from text_processing import clean_text

app = Flask(__name__)


with open("model.pkl", "rb") as f:
   model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
   vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
   prediction = None

   if request.method == "POST":
       text = request.form["text"]

       cleaned = clean_text(text)
       vector = vectorizer.transform([cleaned])
       prediction = model.predict(vector)[0]

   return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
   app.run(debug=True)


   