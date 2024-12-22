import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the saved model and scaler
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])



@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price based on input data from form."""
    try:
        # Get the input values from the form
        data = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]

        # Scale the input data
        final_input = scalar.transform(np.array(data).reshape(1, -1))

        # Make the prediction
        output = regmodel.predict(final_input)[0]

        # Render the template with the prediction
        return render_template('home.html', prediction_text=f"The predicted house price is: ${output:.2f}")
    except Exception as e:
        # Handle any errors during prediction
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

