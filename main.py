from utils import *

if __name__ == "__main__:
  X, y = load_data('creditcard.csv')
  X_train, X_test, y_train, y_test = preprocess_data(X, y)
  models, scores = train_models(X_train, y_train)
  print(scores)
  evaluate_models(models, X_test, y_test)
