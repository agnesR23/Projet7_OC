#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:39:29 2023

@author: agnes
"""

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import json


import xgboost as xgb




##################################################################################################
#Load data
#Fichier à prédire qui a subi un préprocessing comme le fichier d'entraînement
filename = "DATA/Source/df_predict_ft_reduit.csv"
df_predict_ft_reduit = pd.read_csv(filename)

#Fichier pour application du modèle XGBoost
path = "DATA/Source/X_scal.npy"
X_scal = np.load(path)



##################################################################################################
    
#Présentation du DF à prédire
print('Dataframe de données à prédire avec 19 features sélectionnées précédemment et réduit :')
print('df shape : ', df_predict_ft_reduit.shape)
df_predict_ft_reduit




##################################################################################################
#Load model, imputer and standardscaler

model = xgb.XGBClassifier()
model.load_model("MODELS/xgb_model.json")
print('Type du fichier :', type(model))



##################################################################################################
#Application du modèle xgboost avec seuil calculé précédemment et détermination de la probabilité
y_predict_prob = model.predict_proba(X_scal)[:,1]
#Seuil métier optimal : 0.535
y_predict = (model.predict_proba(X_scal)[:,1] >= 0.535).astype(int)

##################################################################################################
#Ajout de la probabilité et de la prédiction au DF et colonne réponse
df = df_predict_ft_reduit.copy()
df['predict_proba'] = y_predict_prob
df['predict'] = y_predict
df['Demande_credit'] = np.nan
df.loc[df['predict'] == 0, 'Demande_credit'] = "Accordée"
df.loc[df['predict'] == 1, 'Demande_credit'] = "Refusée"

#Vérifications 
print("Vérification que le modèle prédit les targets dans les mêmes proportions que dans le notebook")
print(df['predict'].value_counts(normalize=True))
print("Résultats dans le notebook :")
print("0    0.681198")
print("1    0.318802")

#Sauvegarde de df pour streamlit
path = 'DATA/Autre/df.csv'
df.to_csv(path, index=False)

##################################################################################################


app = Flask(__name__)



@app.route('/')
def index():
    return jsonify({'hello': 'world'})
    
    
@app.route("/reponse", methods=['POST'])
def reponse():
    # récupérer l'identifiant
    id = request.json['Identifiant']
    id = int(id)
    result = df.loc[df["SK_ID_CURR"] == id, 'Demande_credit']
    proba_str = str(df.loc[df['SK_ID_CURR'] == id, 'predict_proba'].values[0])
    if result.values == "Refusée":
        return jsonify({"Réponse" : "Non", "Proba_client" : proba_str})
    if result.values == "Accordée":
        return jsonify({"Réponse" : "Oui", "Proba_client" : proba_str})
    else:
        return jsonify({"Réponse" : "Erreur"})

@app.route('/data_customer/', methods=['POST'])
def data_customer():
    id = request.json['Identifiant']
    id = int(id)
    # Get the personal data customer (pd.Series)
    X_cust = df.loc[df["SK_ID_CURR"] == id, :]
    #Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust.to_json())
    # Return the cleaned data
    return jsonify({'status': 'ok',
                    'data': X_cust_json,
                    })

  



if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host="0.0.0.0", port=8080, debug=True) #en local




