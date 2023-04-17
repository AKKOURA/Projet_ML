from flask import Flask, Response, render_template, request
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
from threading import Thread
from plotly.graph_objs import *
import json
import plotly
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
from plotly.subplots import make_subplots



app = Flask(__name__)

# Charger vos données dans un dataframe
df = pd.read_csv("data/dermatology_csv.csv")
# Créer le dictionnaire de correspondance
correspondance_classes = {
    1: "Psoriasis",
    2: "Seboreic dermatitis",
    3: "lichen planus",
    4: "Pityriasis rosea",
    5: "Cronic dermatitis",
    6: "pityriasis rubra pilaris"
}


# Créez la fonction pour générer le graphe 1
def generate_graph1():
   # Charger vos données dans un dataframe
    data = request.get_json()
    maladies = data.get('maladies')
    df = pd.read_csv("data/dermatology_csv.csv")
    df = df[df['class'].isin(map(int, maladies))]
    type = df["class"].replace(correspondance_classes)
    # Données du graphe
    data = [Bar(x=type.value_counts().index,
                y=type.value_counts().values)]

    # Mise en forme du graphe
    layout = Layout(title="Le nombre de personnes par le type de la maladie ( class)",
                    xaxis=dict(title="Type de maladie"),
                    yaxis=dict(title="Nombre de personnes"),
                    annotations=[
                       dict(x=0.5,
                            y=1.1,
                            xref='paper',
                            yref='paper',
                            text="Ce graphe montre le nombre de patients par type de maladie (ou classe) présente dans le dataset. Cela permet d'observer la répartition des patients selon le type de maladie.",
                            showarrow=False)
                   ])

    # Création de la figure
    fig = go.Figure(data=data, layout=layout)

    # Conversion de la figure en JSON
    fig_json = fig.to_json()

    return fig_json

def generate_graph2():
   # Charger vos données dans un dataframe
    data = request.get_json()
    maladies = data.get('maladies')
    df = pd.read_csv("data/dermatology_csv.csv")
    df = df[df['class'].isin(map(int, maladies))]
    df = df.replace({'class': correspondance_classes})

    # Calculer le nombre de personnes pour chaque combinaison "classe, âge"
    count_df = df.groupby(['class', 'age']).size().reset_index(name='count')

    # Créer le graphique en utilisant le dataframe avec les comptes
    fig = px.scatter(count_df, x='class', y='age', color='class', size='count', hover_name='class', hover_data={'age': True, 'count': True})

    # Personnaliser le graphique
    fig.update_layout(title_text="Le nombre de personnes ayant chaque maladie pour chaque âge",
                      xaxis_title_text="Maladie",
                      yaxis_title_text="Âge",
                      hoverlabel=dict(namelength=0))

    # Convertir la figure en JSON
    fig_json = fig.to_json()

    return fig_json


def generate_graph3():
    data = request.get_json()
    maladies = data.get('maladies')
    df = pd.read_csv("data/dermatology_csv.csv")
    df = df[df['class'].isin(map(int, maladies))]

    
    # Obtenir les symptômes ayant une valeur de 3 pour chaque maladie
    symptomes = []
    colors = []
    for i, maladie in enumerate(maladies):
        symptomes_maladie = df[df['class'] == int(maladie)].iloc[:, :-2]
        symptomes_maladie = symptomes_maladie[symptomes_maladie.eq(3)].stack().reset_index()
        symptomes_maladie.columns = ['index', 'symptome', 'value']
        symptomes_maladie = symptomes_maladie.drop(['index', 'value'], axis=1)
        symptomes_maladie = symptomes_maladie.drop_duplicates()
        symptomes_maladie['color'] = f'rgb({i*50}, {i*70}, {i*90})'
        symptomes.append(symptomes_maladie['symptome'])
        colors.append(symptomes_maladie['color'][0])

    # Créer les données pour chaque maladie
    fig = make_subplots(rows=len(maladies), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for i, maladie in enumerate(maladies):
        top_symptomes = df[df['class'] == int(maladie)].iloc[:, :-2][symptomes[i]].sum().sort_values(ascending=False)
        data = [Bar(x=top_symptomes.index, y=top_symptomes.values, name=f"{correspondance_classes[int(maladie)]}", marker_color=colors[i])]
        fig.add_traces(data, rows=[i+1], cols=[1])
        
    # Mise en forme du graphe
    fig.update_layout(title=f"Symptômes avec une forte présence pour chaque maladie",
                      xaxis=dict(title="Symptômes"),
                      yaxis=dict(title="Fréquence"),
                      height=600,
                      showlegend=True,
                      annotations=[
                          dict(x=0.5,
                               y=1.1,
                               xref='paper',
                               yref='paper',
                               text="Ce graphe montre les symptômes ayant une forte présence pour chaque maladie sélectionnée.",
                               showarrow=False)
                      ])
    
    # Conversion de la figure en JSON
    fig_json = fig.to_json()

    return fig_json

 # def generate_graph4():
def generate_graph4():
    # Charger vos données dans un dataframe
    data = request.get_json()
    maladies = data.get('maladies')
    df = pd.read_csv("data/dermatology_csv.csv")
    df = df[df['class'].isin(map(int, maladies))]

    # Trouver le nombre de cas avec antécédents familiaux
    hereditary_cases = df[df['family_history'] == 1].shape[0]
    # Trouver le nombre de cas sans antécédents familiaux
    non_hereditary_cases = df[df['family_history'] == 0].shape[0]

    # Calculer les pourcentages d'hérédité et de non-hérédité
    total_cases = hereditary_cases + non_hereditary_cases
    hereditary_percentage = hereditary_cases / total_cases * 100
    non_hereditary_percentage =  100 - hereditary_percentage

    # Données du graphe
    labels = ['Héréditaire', 'Non-héréditaire']
    values = [hereditary_percentage, non_hereditary_percentage]
    colors = ['rgb(255, 127, 14)', 'rgb(31, 119, 180)']
    data = [Pie(labels=labels, values=values, marker=dict(colors=colors))]

    # Mise en forme du graphe
    layout = Layout(title="Pourcentage de personne qui ont eu la maladie de manière héréditaire",
                    annotations=[
                       dict(x=0.5,
                            y=1.1,
                            xref='paper',
                            yref='paper',
                            text="Ce graphique en camembert montre le pourcentage de personne ayant eu la maladie sélectionnée de manière héréditaire ou non. Cela permet d'observer la proportion de cas de maladies qui sont hériditaires parmi les patients étudiés.",
                            showarrow=False)
                   ])

    # Création de la figure
    fig = go.Figure(data=data, layout=layout)

    # Conversion de la figure en JSON
    fig_json = fig.to_json()

    return fig_json

def generate_graph5():
    # Charger vos données dans un dataframe
    data = request.get_json()
    maladies = data.get('maladies')
    df = pd.read_csv("data/dermatology_csv.csv")
    df = df[df['class'].isin(map(int, maladies))]

    # Trouver les pourcentages d'hérédité des maladies sélectionnées
    hereditary_counts = df[df['family_history'] == 1]['class'].value_counts(normalize=True).sort_values(ascending=False) * 100

    # Remplacer les codes de classes par leur correspondance
    hereditary_counts.index = hereditary_counts.index.map(correspondance_classes.get)

    # Données du graphe
    data = [go.Pie(labels=hereditary_counts.index, values=hereditary_counts.values)]

    # Mise en forme du graphe
    layout = go.Layout(title="comparaison du pourcentage d'hérédité des maladies sélectionnées",
                       annotations=[
                           dict(x=0.5,
                                y=1.1,
                                xref='paper',
                                yref='paper',
                                text="Ce graphe montre le pourcentage d'hérédité des maladies sélectionnées. Cela permet d'observer la prédisposition génétique des patients à ces maladies.",
                                showarrow=False)
                       ])

    # Création de la figure
    fig = go.Figure(data=data, layout=layout)

    # Conversion de la figure en JSON
    fig_json = fig.to_json()

    return fig_json


@app.route('/graph1', methods=['POST'])
def graph1():
  fig_json = generate_graph1()
  return fig_json

# Créez la route pour afficher le graphe 2 
@app.route('/graph2', methods=['POST'])
def graph2():
  fig_json = generate_graph2()
  return fig_json

# Créez la route pour afficher le graphe 3
@app.route('/graph3', methods=['POST'])
def graph3():
  fig = generate_graph3()
  return fig

# Créez la route pour afficher le graphe 3
@app.route('/graph4', methods=['POST'])
def graph4():
  fig = generate_graph4()
  return fig

# Créez la route pour afficher le graphe 5
@app.route('/graph5', methods=['POST'])
def graph5():
  fig = generate_graph5()
  return fig

# Créez la route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)