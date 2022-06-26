
import joblib
from joblib import dump, load
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel # classe qui constomise les entrées de fassapi


app = FastAPI() # initialise l'aplication 

# pour récupeer les clients données d'entrée d'API

class Client(BaseModel):
    index: int
    
    

svc_joblib_model = joblib.load('logreg_model_joblib')

# import the data 

X_test=pd.read_csv('app_x_test.csv')


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


