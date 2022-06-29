import streamlit as st
import pandas as pd
import lime
import requests
import json
from types import SimpleNamespace

# online
url_api = 'https://applicationprojet7.herokuapp.com/feature_importance'

# local
# url_api = 'http://127.0.0.1:8000/feature_importance'

 # Display the title
st.title('Credit application dashboard')
st.header("Adèle Souleymanova - Data Science project 7- Openclassrooms")

# Menu pour les clients 


# envoyer l'identifiant du client à l'api 
response = requests.get(url=url_api).json()
perm_importance_mean = response['mean_perm_importances']
features_idx = response['features_sorted_']


# st.markdown(perm_importance_mean)
resultData = pd.DataFrame(perm_importance_mean, features_idx)

st.bar_chart(resultData)


