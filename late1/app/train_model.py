# app/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Replace with your actual dataset loading code
from sklearn.metrics import accuracy_score  # Import to calculate accuracy
import joblib
import os

# Create model directory if it doesn't exist
model_directory = 'model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Load your dataset (replace with your actual dataset)
data = load_iris()  # Example dataset, replace with your dataset
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

# Save the trained model and accuracy
joblib.dump(rf_classifier, os.path.join(model_directory, 'random_forest_classifier.pkl'))
joblib.dump(accuracy, os.path.join(model_directory, 'model_accuracy.pkl'))  # Save accuracy separately

print(f"Model trained and saved as 'model/random_forest_classifier.pkl'. Accuracy: {accuracy:.2f} percent")
