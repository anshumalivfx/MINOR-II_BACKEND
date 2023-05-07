from logistic import LogisticRegression
import joblib

# Load the model
model = joblib.load("LogisticRegression.joblib")

# Make a prediction for a new sample
X_new = [[0.6]]
y_pred = model.predict(X_new)
y_pred = model.predict_with_confidence(X_new)
print(y_pred)