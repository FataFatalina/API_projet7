import streamlit as st
import pandas as pd
import lime
import requests
import json
from types import SimpleNamespace

# online
url_api = 'https://applicationprojet7.herokuapp.com/get_data'

# local
# url_api = 'http://127.0.0.1:8000/get_data'

 # Display the title
st.title('Credit application dashboard')
st.header("Adèle Souleymanova - Data Science project 7- Openclassrooms")

# Menu pour les clients 


# envoyer l'identifiant du client à l'api 
reponse = requests.get(url=url_api).json()
responseData = reponse['data']
st.dataframe(responseData)


