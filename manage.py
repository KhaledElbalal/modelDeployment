#!pip install flask
#!pip install sklearn
#!pip install pandas
#!pip install numpy
#!pip install scipy
# Load needed libraries
import pandas as pd
import numpy as np
from scipy import stats
from flask import Flask, jsonify, render_template, request
import json
from waitress import serve
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
# Remove Outliers
# Outlier removal
df = pd.DataFrame(stats.trimboth(df, proportiontocut=0.02), columns=[
                  'Pregnancies', 'BloodPressure', 'BMI', 'Glucose', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
# Data Optimiztion
df['Pregnancies'][df.Pregnancies > 0] = 1
auxData = df.drop('Outcome', axis=1)
df[columns] = sc.fit_transform(df[columns])
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].apply(
    lambda x: 1/(1+np.exp(-x)))
# Split training data
Outcome = df.Outcome
newDf = df.drop('Outcome', axis=1)
trainingData_x, testData_x, trainingData_y, testData_y = train_test_split(
    newDf, Outcome, test_size=0.33, random_state=123)
# Use logisitic regression to fit training data
logReg.fit(trainingData_x, trainingData_y)
# Flask server
app = Flask(__name__)
print("Model accuracy:", logReg.score(testData_x, testData_y))


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    try:
        arr = []
        # Fill Array
        for requests in request.form:
            if(requests != 'Pregnancy' or requests != 'Age'):
                arr.append([str(requests), float(request.form[requests])])
        test = json.dumps(arr)
        # Convert it to data frame
        dataN = pd.read_json(test).T
        headers = dataN.iloc[0]
        dataN = pd.DataFrame(dataN.values[1:], columns=headers)
        # Clean Input Data
        dataN['DiabetesPedigreeFunction'] = dataN['DiabetesPedigreeFunction'].apply(
            lambda x: 1/(1+np.exp(-x)))
        dataN['Pregnancies'][dataN.Pregnancies > 0] = 1
        newAuxData = auxData.append(dataN)
        newAuxData[columns] = sc.fit_transform(newAuxData[columns])
        print("test")
        dataN = newAuxData.iloc[-1:]
        return render_template('diabetes.html', positive=logReg.predict(dataN)[0])
    except Exception as ex:
        print(ex)
        return render_template('error.html')


serve(
    app,
    host="0.0.0.0",
    port=5000,
    url_scheme='https'
)
