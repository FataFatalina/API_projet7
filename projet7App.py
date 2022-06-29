
import joblib
from joblib import dump, load
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel # classe qui constomise les entrées de fassapi
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import sys
from lime import lime_tabular
sys.setrecursionlimit(1000000)

app = FastAPI() # initialise l'aplication 

# pour récupeer les clients données d'entrée d'API
class Client(BaseModel):
    index: int
    
    

svc_joblib_model = joblib.load('./data/logreg_model_joblib')

# import the data 

X_test=pd.read_csv('./data/X_test.csv')
y_test=pd.read_csv('./data/y_test.csv')
X_train=pd.read_csv('./data/X_train.csv')
y_train=pd.read_csv('./data/y_train.csv')

app_x_test=pd.read_csv('./data/app_x_test.csv')

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

@app.post('/local_features_importance')
def local_features_importance(sample_client:Client):
    explainer = lime_tabular.LimeTabularExplainer(training_data =X_train.values,
                                                feature_names=X_train.columns.values,
                                                class_names =['Non_defaulter_0','Defaulter_1'],
                                                mode="classification",
                                                verbose=False,
                                                random_state=10)
    client_index = sample_client.dict()
    num=int(client_index['index'])
    test_sample=x_test.iloc[num,:]
    lim_exp =explainer.explain_instance(data_row=test_sample, predict_fn=svc_joblib_model.predict_proba, num_features=117)
    
    weights = []
    counter = 0
    #Iterate over feature matrix
    for x in X_test.values[0:100]:
        if counter == num:
            #Get explanation
            exp = lim_exp
            exp_list = exp.as_map()[1]
            exp_list = sorted(exp_list, key=lambda x: x[0])
            exp_weight = [x[1] for x in exp_list]
            #Get weights
            weights.append(exp_weight)
            #Create DataFrame
            lime_dt = pd.DataFrame(data=weights,columns=X_test.columns)
            var_names = []
            var_values = []
            for key, value in lime_dt.iteritems():
                var_names.append(key)
                var_values.append(value[0])
            
            return {"var_names": var_names[0:20], "var_values": var_values[0:20]}

        counter += 1

    return "client not found"

