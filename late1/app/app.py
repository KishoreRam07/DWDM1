from flask import Flask, render_template, request, redirect, url_for
import joblib
import json
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load your trained Random Forest model
model = joblib.load('model/random_forest_classifier.pkl')

# Load example dataset for accuracy calculation
data = load_iris()  # Replace with your actual dataset
X = data.data
y = data.target

# Sample data to visualize in the dashboard (for demonstration)
predictions = {
    'Late': 0,
    'On Time': 0
}

# Example feature names, replace these with your actual feature names
feature_names = ['Delivery Distance', 'Delivery Time', 'Delivery Urgency', 'Delivery Location']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    
    input_data = [[float(feature1), float(feature2), float(feature3), float(feature4)]]
    
    # Make predictions
    prediction = model.predict(input_data)[0]
    prediction_label = "On Time" if prediction == 0 else "Late"

    # Store prediction result
    predictions[prediction_label] += 1

    # Calculate accuracy based on the dataset (This is just a placeholder logic for demonstration)
    accuracy = model.score(X, y) * 100  # Get accuracy as a percentage

    # Pass prediction and accuracy to the dashboard
    return render_template('dashboard.html', prediction=prediction_label, accuracy=accuracy, graphJSON=get_prediction_graph(), importance_graphJSON=get_importance_graph())

def get_prediction_graph():
    data = [
        go.Bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            marker=dict(color=['blue', 'red'])
        )
    ]
    return json.dumps(data, cls=PlotlyJSONEncoder)

def get_importance_graph():
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order

    importance_data = [
        go.Bar(
            x=[feature_names[i] for i in indices],
            y=importances[indices],
            marker=dict(color='green')
        )
    ]
    return json.dumps(importance_data, cls=PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True)
