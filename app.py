import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model_obesitas.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', **locals())

@app.route("/predict", methods=["POST"])
def predict():
    # Mengambil nilai dari form menggunakan atribut 'name'
    Age = int(request.form.get('Age'))
    Consumption_of_Fast_Food = float(request.form.get('Consumption_of_Fast_Food'))
    Frequency_of_Consuming_Vegetables = float(request.form.get('Frequency_of_Consuming_Vegetables'))
    Number_of_Main_Meals_Daily = float(request.form.get('Number_of_Main_Meals_Daily'))
    Physical_Excercise = float(request.form.get('Physical_Excercise'))
    
    # Menyusun fitur untuk prediksi
    features = np.array([Age, Consumption_of_Fast_Food, Frequency_of_Consuming_Vegetables, 
                         Number_of_Main_Meals_Daily, Physical_Excercise]).reshape(1, -1)
    
    # Melakukan prediksi menggunakan model
    prediction = model.predict(features)[0]

    if prediction == 1:
        prediction_text = "Berat Badan Kurang"
    elif prediction == 2:
        prediction_text = "Berat Badan Normal"
    elif prediction == 3:
        prediction_text = "Kelebihan Berat Badan"
    else:
        prediction_text = "Obesitas"
    
    # Render halaman hasil prediksi (prediksi.html)
    return render_template('prediksi.html', prediction_text=prediction_text)


if __name__=="__main__":
    app.run(debug=True)