import numpy as np
from flask import Flask, render_template, request
import pickle
import os

img = os.path.join('static', 'Image')

app = Flask(__name__)

# Load your pre-trained decision tree model
model = pickle.load(open('./training/ML-occupancy-rates.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Temperature = float(request.form['temperature'])
        Humidity = float(request.form['humidity'])
        Light = float(request.form['light'])
        CO2 = float(request.form['co2'])
        HumidityRatio = float(request.form['humidityRatio'])
        

        print("Received data:")
        print("Temperature:", Temperature)
        print("Humidity:", Humidity)
        print("Light:", Light)
        print("CO2:", CO2)
        print("HumidityRatio:", HumidityRatio)
        

        # Create a NumPy array with the input data
        total = np.array([[Temperature, Humidity, Light, CO2, HumidityRatio]])

        # Use the pre-trained model to predict occupancy
        y_test = model.predict(total)

        print("Model prediction:", y_test)

        # Process the prediction result
        if y_test[0] == 0:
            ans = "It is not Occupied"
        else:
            ans = "It is Occupied"

        return render_template("home.html", showcase=ans)

    except Exception as e:
        return render_template("home.html", showcase="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
