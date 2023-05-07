from flask import Flask, request, jsonify
import numpy as np
import joblib
from logistic import LogisticRegression


app = Flask(__name__)

# Load the model
model = joblib.load("LogisticRegression.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    # Convert it to a numpy array
    X = np.array([[data['Eh']]])
    print(X)
    X = X.reshape(-1, 1)
    # Make the prediction
    y_pred = model.predict(X)
    
    print(y_pred)
    # Get the confidence level
    confidence = model.predict_with_confidence(X)
    print(confidence[1])
    # Return the prediction and confidence level as JSON
    response = {'prediction': int(y_pred[0]), 'confidence': float(confidence[1])}
    return jsonify(response)



app.run(debug=True)