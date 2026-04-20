from data_loader import load_dataset, explore_dataset
from text_processing import preprocess_dataset, extract_features, clean_text
from model import split_data, train_model
from evaluation import evaluate_model
from deployment import save_model, load_model, predict_sentiment

def main():
   print("Program started...")
   df = load_dataset()
   explore_dataset(df)

   df = preprocess_dataset(df)

   X, y, vectorizer = extract_features(df)

   X_train, X_test, y_train, y_test = split_data(X, y)

   model = train_model(X_train, y_train)

   evaluate_model(model, X_test, y_test)

   save_model(model, vectorizer)

   loaded_model, loaded_vectorizer = load_model()
  
   sample_text = "I really love this game"
   prediction = predict_sentiment(sample_text, loaded_model, loaded_vectorizer, clean_text)

   print("\nSample text:")
   print(sample_text)

   print("\nPredicted sentiment:")
   print(prediction)

   print("\nProgram finished successfully.")

if __name__ == "__main__":
   main()