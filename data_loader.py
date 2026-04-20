import pandas as pd

def load_dataset():
   df = pd.read_csv(
       "twitter_training.csv",
       header=None,
       names=["id", "entity", "label", "text"]
   )
   df = df.dropna(subset=["text"])
   df = df.head(2000)
   return df

def explore_dataset(df):
   print("First 5 rows:")
   print(df.head())

   print("\nDataset shape:")
   print(df.shape)

   print("\nMissing values:")
   print(df.isnull().sum())

   print("\nClass distribution:")
   print(df["label"].value_counts())