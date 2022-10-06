import streamlit as st 
import pandas as pd
import numpy as np 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write('''

# App Simple pour la prévision des fleurs d'Iris
Cette application  prédit la catégorie des fleurs d'Iris

''')

st.sidebar.header("Les parametres d'entrée")

def user_input():
    sepal_lenght=st.sidebar.slider('La longeur de Sepal',4.3,7.9,5.3)
    sepal_with =st.sidebar.slider('La largeur de Sepal',2.0,4.4,3.3)
    petal_lenght=st.sidebar.slider('La longeur de Petal',1.0,6.9,2.3)
    petal_with=st.sidebar.slider('La largeur de Petal',0.1,2.5,1.3)
    data = {
        'sepal_lenght' : sepal_lenght,
        'sepal_with': sepal_with,
        'petal_lenght': petal_lenght,
        'petal_with': petal_with
    }
    fleur_parametres = pd.DataFrame(data,index=[0])
    return fleur_parametres 

df =user_input()


st.subheader('On veut trouver la catégorie de cette fleur ')
st.write(df)


iris = datasets.load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data,iris.target)
 
prediction = clf.predict(df)

st.subheader("La catégorie de la fleur d'iris est :")

st.write(iris.target_names[prediction])