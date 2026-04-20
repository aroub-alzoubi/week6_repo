import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)

   print("\nAccuracy:")
   print(accuracy_score(y_test, y_pred))

   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))

   cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
   disp.plot()
   plt.savefig("confusion_matrix.png")