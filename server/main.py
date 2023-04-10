from flask import Flask, Response, render_template
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
    # Données du graphe
    data = []
    for i in range(1, 7):
        trace = Histogram(x=df[df['class']==i]['age'], nbinsx=20, name=correspondance_classes[i].format(i), opacity=0.5)
        data.append(trace)

    # Mise en forme du graphe
    layout = Layout(title="Répartition des âges par le type de la maladie (class)",
                    xaxis=dict(title="Âge"),
                    yaxis=dict(title="Nombre de patients"),
                    barmode='overlay',
                    annotations=[
                       dict(x=0.5,
                            y=1.1,
                            xref='paper',
                            yref='paper',
                            text="Ce graphe montre comment les patients sont répartis selon leur âge pour chaque type de maladie de peaux (ou classe) présente dans le dataset.",
                            showarrow=False)
                   ])
    

    # Création de la figure
    fig = Figure(data=data, layout=layout)

    # Conversion de la figure en JSON
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig_json

def generate_graph3():
  # Données du graphe
  data = [Bar(x=df['family_history'].unique(),
              y=df.groupby('family_history')['class'].count().values)]

  # Mise en forme du graphe
  layout = Layout(title="Le nombre de personnes avec antécédents familiaux par classe",
                  xaxis=dict(title="Antécédents familiaux"),
                  yaxis=dict(title="Nombre de personnes"),
                  annotations=[
                       dict(x=0.5,
                            y=1.1,
                            xref='paper',
                            yref='paper',
                            text="Ce graphe montre le nombre de personnes ayant des antécédents familiaux pour chaque classe de maladie, permettant ainsi de déterminer les classes de maladies les plus héréditaires.",
                            showarrow=False)
                   ])

  # Création de la figure
  fig = Figure(data=data, layout=layout)

  # Conversion de la figure en JSON
  fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

  return fig_json

def generate_graph4():
  # Trouver les classes de maladies les plus héréditaires
  hereditary_diseases = df[df['family_history'] == 1]['class']
  counts = hereditary_diseases.value_counts().sort_values(ascending=False)

  # Remplacer les codes de classes par leur correspondance
  counts.index = counts.index.map(correspondance_classes.get)

  # Données du graphe
  data = [go.Bar(x=counts.index, y=counts.values)]

  # Mise en forme du graphe
  layout = go.Layout(title="Classes de maladies les plus héréditaires",
                      xaxis=dict(title="Classe de maladie"),
                      yaxis=dict(title="Nombre de patients"),
                      annotations=[
                       dict(x=0.5,
                            y=1.1,
                            xref='paper',
                            yref='paper',
                            text="Ce graphique présente les classes de maladies les plus héréditaires parmi les patients étudiés. Nous pouvons voir que Psoriasis est la maladie la plus hériditaire, avec un nombre de patients significativement plus élevé que les autres classes.",
                            showarrow=False)
                   ])

  # Création de la figure
  fig = go.Figure(data=data, layout=layout)

  # Conversion de la figure en JSON
  fig_json = fig.to_json()

  return fig_json
# Créez la route pour afficher le graphe 1
@app.route('/graph1')
def graph1():
  fig_json = generate_graph1()
  return fig_json

# Créez la route pour afficher le graphe 2 
@app.route('/graph2')
def graph2():
  fig_json = generate_graph2()
  return fig_json

# Créez la route pour afficher le graphe 3
@app.route('/graph3')
def graph3():
  fig = generate_graph3()
  return fig

# Créez la route pour afficher le graphe 3
@app.route('/graph4')
def graph4():
  fig = generate_graph4()
  return fig

# Créez la route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)