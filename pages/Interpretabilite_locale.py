import streamlit as st
import pandas as pd
import lime
import requests
import json
from types import SimpleNamespace

# online
url_api = 'https://applicationprojet7.herokuapp.com'

# local
# url_api = 'http://127.0.0.1:8000'

 # Display the title
st.title('Credit application dashboard')
st.header("Adèle Souleymanova - Data Science project 7- Openclassrooms")
# Requete pour avoir la liste des indexes de tous les client (que nous allons afficher dans le menu à gauche)
client_indexes_response = requests.get(url=str(url_api)+str("/get_client_indexes")).json()

# Menu pour les clients 
identifiant_client= st.sidebar.selectbox("Choisir ID d'un client", client_indexes_response["data"][0:100])

st.markdown(str('Identifiant client: ') + str(identifiant_client)) 

client={"index": int(identifiant_client)}

# envoyer l'identifiant du client à l'api 
response = requests.post(url=str(url_api)+str("/local_features_importance"), json=client).json()
if response == "client not found":
    st.markdown(response)
else:
    var_names = response['var_names']
    var_values = response['var_values']
    resultData = pd.DataFrame(var_values, var_names)
    st.bar_chart(resultData)


# lime_dt = response['lime_dt']
# st.bar_chart(lime_dt)



