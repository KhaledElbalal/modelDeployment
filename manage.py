#!pip install flask
#!pip install sklearn
#!pip install pandas
#!pip install numpy
# Load needed libraries
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
# Evaluation and model selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Data Set
df = pd.read_csv("dataset.csv")
logReg = LogisticRegression(C=1.623776739188721, max_iter=10000, solver='sag')
sc = StandardScaler()
# Change 0 values to their mean
columns = ['BloodPressure', 'BMI', 'Glucose',
           'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']
for column in columns:
    mean = df[column][df[column] > 0].mean()
    df[column].replace(0, np.floor(mean), inplace=True)
# Data Optimiztion
df['Pregnancies'][df.Pregnancies > 0] = 1
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].apply(
    lambda x: 1/(1+np.exp(-x)))
df[columns] = sc.fit_transform(df[columns])
# Split training data
Outcome = df.Outcome
newDf = df.drop('Outcome', axis=1)
trainingData_x, testData_x, trainingData_y, testData_y = train_test_split(
    newDf, Outcome, test_size=0.3, random_state=123)
# Use logisitic regression to fit training data
logReg.fit(trainingData_x, trainingData_y)
# Flask server
app = Flask(__name__)
print(logReg.score(testData_x, testData_y))


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    try:
        Preg = request.form['Pregnancy']
        Gluc = request.form['Glucose']
        BP = request.form['BloodPressure']
        ST = request.form['SkinThickness']
        Insul = request.form['Insulin']
        BMI = request.form['BMI']
        DPF = request.form['DiabetesPedigreeFunction']
        DPF = 1/(1+np.exp(-DPF))
        Age = request.form['Age']
        arr = [Gluc, BP, ST, Insul, BMI, DPF]
        arr = sc.fit_transform(arr)
        arr.append(Age)
        arr.insert(Preg, 0)
        print(arr)
        json = jsonify({"Prediction": str(logReg.predict([arr])[0])})
        return json
    except Exception as ex:
        print(ex)
        return jsonify({"Prediction": "An error has occured"})


app.run()
