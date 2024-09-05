from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("iris_model.pkl")  # IRIS veri seti için model dosyanızın adı

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    # IRIS veri setinde sınıf isimlerini belirliyoruz
    classes = ['setosa', 'versicolor', 'virginica']
    result = classes[int(prediction[0])]
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
