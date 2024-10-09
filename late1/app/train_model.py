import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# Handle missing values in 'Order Zipcode' and 'Product Description'
data['Order Zipcode'].fillna(data['Order Zipcode'].mode()[0], inplace=True)
data['Product Description'].fillna("Unknown", inplace=True)

# Define features and target
feature_columns = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Category Id',
    'Customer Zipcode',
    'Order Item Quantity',
    'Order Item Product Price'
]
target_column = 'Late_delivery_risk'

X = data[feature_columns]
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), feature_columns),
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

with open('accuracy.txt', 'w') as f:
    f.write(f'{accuracy:.2f}')

joblib.dump(pipeline, 'model.joblib')
