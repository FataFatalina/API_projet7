import streamlit as st
import pandas as pd
import lime
import requests

url_api = 'https://applicationprojet7.herokuapp.com/predict'
data= pd.read_csv('app_x_test.csv')

 # Display the title
st.title('Credit application dashboard')
st.header("Adèle Souleymanova - Data Science project 7- Openclassrooms")

# Menu pour les clients 
identifiant_client= st.sidebar.selectbox("Choisir ID d'un client", list(data.index))

st.markdown(str(identifiant_client)) 

client={"index": int(identifiant_client)}

# envoyer l'identifiant du client à l'api 
reponse = requests.post(url=url_api, json=client).json()
st.markdown(reponse)