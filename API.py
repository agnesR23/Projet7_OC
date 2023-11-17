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




import pickle

import xgboost as xgb




##################################################################################################
#Load data
#Fichier à prédire qui a subi un préprocessing comme le fichier d'entraînement

filename = "DATA/Source/df_prec.csv"
df_prec = pd.read_csv(filename)



##################################################################################################
    
#Présentation du DF à prédire
print('Dataframe de données à prédire avec 19 features sélectionnées précédemment :')
print('df_prec shape : ', df_prec.shape)
df_prec

##################################################################################################


#Load 18 features sélectionnées précédemment sans 'SK_ID_CURR'
feat = 'DATA/Autre/arr_list_feat_pred18.npy'
arr_list_feat_pred18 = np.load(feat)  
list_feat_pred18 = list(arr_list_feat_pred18) 
print(list_feat_pred18)

#Load 19 features avec 'SK_ID_CURR'
feat = 'DATA/Autre/arr_list_feat_pred19.npy'
arr_list_feat_pred19 = np.load(feat)  
list_feat_pred19 = list(arr_list_feat_pred19)
print(list_feat_pred19)


#Suppression feature SK_ID_CURR pour le modèle (sélection sans SK_ID_CURR)
X = df_prec[list_feat_pred18]
print('Dataframe sans SK_ID_CURR :')
print(X)



##################################################################################################
#Load model, imputer and standardscaler

model = xgb.XGBClassifier()
model.load_model("MODELS/xgb_model.json")
print('Type du fichier :', type(model))



path = "MODELS/imputer.sav"
f = open(path, 'rb')
imputer = pickle.load(f)
print('Type du fichier :', type(imputer))
f.close()

path = "MODELS/scaler.sav"
f = open(path, 'rb')
scaler = pickle.load(f)
print('Type du fichier :', type(scaler))
f.close()


##################################################################################################
#Application de l'imputer sauvegardé et du standardscaler
X_imp = imputer.transform(X)
X_scal = scaler.transform(X_imp)

#Sauvegarde de X_scal qui servira pour shap dans streamlit
path = 'DATA/Autre/X_scal.npy'
np.save(path, X_scal)


##################################################################################################
#Application du modèle xgboost avec seuil calculé précédemment et détermination de la probabilité
y_predict_prob = model.predict_proba(X_scal)[:,1]
#Seuil métier optimal : 0.535
y_predict = (model.predict_proba(X_scal)[:,1] >= 0.535).astype(int)

##################################################################################################
#Ajout de la probabilité et de la prédiction au DF et colonne réponse
df = df_prec.copy()
df['predict_proba'] = y_predict_prob
df['predict'] = y_predict
df['Demande_credit'] = np.nan
df.loc[df['predict'] == 0, 'Demande_credit'] = "Accordée"
df.loc[df['predict'] == 1, 'Demande_credit'] = "Refusée"

#Vérifications 
print("Vérification que le modèle prédit les targets dans les mêmes proportions que dans le notebook")
print(df['predict'].value_counts(normalize=True))
print("Résultats dans le notebook :")
print("0    0.681171")
print("1    0.318829")

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
    app.run(host="0.0.0.0", port=8080, debug=True)




