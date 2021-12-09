
"""
Created on Nov 23, 2021

@author: Ramniwash
"""

from flask import Flask, jsonify, request
from sklearn import *
from tensorflow import keras
import pandas as pd
import numpy as np
import json
import joblib


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
   
import flask

app = Flask(__name__)

main_cols = joblib.load("columns.pkl")

def standardize_data(dta):
    scaler = joblib.load("std_scaler.pkl")
    X_transformed = scaler.transform(dta)
    return X_transformed


def clean_data(df):
    le = LabelEncoder()
    df = pd.get_dummies(data = df, columns=["Branch"], drop_first = False)
    df = pd.get_dummies(data = df, columns=["City"], drop_first = False)
    df = pd.get_dummies(data = df, columns=["Customer type"], drop_first = False)
    df.Gender = le.fit_transform(df.Gender)
    df = pd.get_dummies(data = df, columns=["Product line"], drop_first = False)
    df = pd.get_dummies(data = df, columns=["Payment"], drop_first = False)
    df = df.sort_index(axis=1)
    return df


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    df_input = pd.DataFrame.from_records([form_data], )
    df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)

    sample_df = pd.DataFrame(columns=main_cols)
    print(main_cols)
    clean_df = clean_data(df_input)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)
    std_df = standardize_data(main_df)
    clf = joblib.load('fynnlabs_project1_model.sav')
    pred = clf.predict(std_df)

    x = pred

    print(x)
    
    return flask.render_template('index.html', predicted_value="Rating: {}".format(x))
    # return jsonify({'prediction': str(x)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)