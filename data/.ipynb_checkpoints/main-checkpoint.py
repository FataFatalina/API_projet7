
import joblib
from joblib import dump, load
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel # classe qui constomise les entrées de fassapi
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import sys
from fastapi.middleware.cors import CORSMiddleware
sys.setrecursionlimit(1000000)

app = FastAPI() # initialise l'aplication 

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pour récupeer les clients données d'entrée d'API

class Client(BaseModel):
    index: int
    
    

svc_joblib_model = joblib.load('logreg_model_joblib')

# import the data 

X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')
X_train=pd.read_csv('X_train.csv')
y_train=pd.read_csv('y_train.csv')

app_x_test=pd.read_csv('app_x_test.csv')

x_test=X_test.iloc[0:100]
y_y_test=y_test[0:100]['TARGET']
x_train=X_train.iloc[0:300]
y_y_train=y_train[0:300]['TARGET']




@app.post('/predict') # l'URL de l'appi 
def predict_score(sample_client:Client):
    data = sample_client.dict()
    index = int(data['index']) # récuperer l'index
    client= X_test.iloc[[index]] # pour récuperer les données du  client
    preds=svc_joblib_model.predict_proba(client)
    seuil =0.2
    if preds[0][0]< seuil:
        prediction = 'Client solvable'
        probabilite = preds[0][0] #la probabilité réél du client à être solvable
    else:
        prediction='Client non solvable'
        probabilite = preds[0][1]
    #print(prediction)
    return {'prediction': prediction, 'probabilite': probabilite}

@app.get('/get_data') #Le tableau de données
def get_data():
    cleaned_data = app_x_test.iloc[: , 1:]
    return {"data": cleaned_data.head()} #Renvoyer le tableau

@app.get('/get_client_indexes') #Avoir les indexes des clients
def get_clients_indexes():
    clients_indexes = []
    for i in range(len(app_x_test)):
        clients_indexes.append(i)
    return {"data": clients_indexes} #Renvoyer le tableau

@app.get('/feature_importance')
def feature_importance():
    feature_names = X_train.columns
    perm_importance = permutation_importance(svc_joblib_model, X_test, y_test)
    mean_perm_importances = np.array(sorted(perm_importance.importances_mean, reverse=True))[0:20].tolist()
    sorted_idx =np.array(mean_perm_importances).argsort().tolist()
    features_sorted_= feature_names[sorted_idx].tolist()
 
    
    return {"mean_perm_importances": mean_perm_importances, "features_sorted_": features_sorted_}