from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
application = Flask(__name__)

# Shortcut to the Flask application
app = application

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        print(pred_df)  # Check the data before prediction
        print("Before Prediction")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Make the prediction
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Return result to the home page template
        return render_template('home.html', results=results[0])


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
