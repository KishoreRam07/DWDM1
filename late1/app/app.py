from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.inspection import permutation_importance

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        days_shipping_real = float(request.form['days_shipping_real'])
        days_shipment_scheduled = float(request.form['days_shipment_scheduled'])
        benefit_per_order = float(request.form['benefit_per_order'])
        sales_per_customer = float(request.form['sales_per_customer'])
        category_id = int(request.form['category_id'])
        customer_zipcode = int(request.form['customer_zipcode'])
        order_item_quantity = int(request.form['order_item_quantity'])
        order_item_product_price = float(request.form['order_item_product_price'])

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
            'Days for shipping (real)': [days_shipping_real],
            'Days for shipment (scheduled)': [days_shipment_scheduled],
            'Benefit per order': [benefit_per_order],
            'Sales per customer': [sales_per_customer],
            'Category Id': [category_id],
            'Customer Zipcode': [customer_zipcode],
            'Order Item Quantity': [order_item_quantity],
            'Order Item Product Price': [order_item_product_price]
        })

        prediction = model.predict(input_data)

        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            synthetic_data = pd.concat([input_data] * 10, ignore_index=True)
            for col in synthetic_data.columns:
                synthetic_data[col] += np.random.normal(0, 0.01, size=synthetic_data.shape[0])
            result = permutation_importance(model, synthetic_data, model.predict(synthetic_data), n_repeats=10, random_state=0)
            feature_importances = result.importances_mean

        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        importance_df['Importance'] /= importance_df['Importance'].sum()

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Normalized Feature Importance')
        plt.title('Feature Importances Impacting Prediction')
        plt.grid(axis='x')

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        shap_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        with open('accuracy.txt', 'r') as f:
            accuracy = f.read().strip()

        return render_template('dashboard.html', 
                               prediction=prediction[0], 
                               accuracy=accuracy,  # Use the real-time accuracy
                               shap_plot_url=shap_plot_url,
                               reasons=importance_df['Importance'].tolist(),
                               feature_names=importance_df['Feature'].tolist())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
