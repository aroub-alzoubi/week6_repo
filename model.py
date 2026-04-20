from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def split_data(X, y):
   return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   return model