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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.subplots as sp
from sklearn.decomposition import PCA


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

# Fonction pour obtenir les couleurs des symptômes
def get_symptomes_colors(symptomes):
  symptomes_colors = {'0': 'rgb(220,220,220)', '1': 'rgb(178,223,138)', '-1': 'rgb(251,154,153)'}

  # Convertir les valeurs de couleurs hexadécimales en valeurs de couleurs RVB
  symptomes_colors = {k: tuple(map(int, v[4:-1].split(','))) for k, v in symptomes_colors.items()}

  # Appliquer les couleurs à chaque symptôme
  symptomes_colors = pd.DataFrame(symptomes).replace(symptomes_colors)
  return symptomes_colors

def generate_graph3():
  data = request.get_json()
  maladies = data.get('maladies')
  df = pd.read_csv("data/dermatology_csv.csv")
  df = df[df['class'].isin(map(int, maladies))]
  # Créer une matrice de corrélation pour les symptômes
  corr_matrix = df.iloc[:, :-2].corr()

  # Créer le graphe pour chaque maladie sélectionnée
  figs = []
  for maladie in maladies:
      maladie = int(maladie)
      # Sélectionner les symptômes pour la maladie
      symptomes_maladie = df[df['class'] == maladie].iloc[:, :-2]

      # Ajouter les couleurs pour les symptômes
      symptomes_maladie_colors = get_symptomes_colors(symptomes_maladie)
        
      # Créer la matrice de corrélation pour la maladie sélectionnée
      corr_matrix_maladie = symptomes_maladie_colors.corr()

      # Créer le graphe
      fig = go.Figure()

      # Ajouter les boules de couleur pour chaque symptôme
      for i, symptome in enumerate(symptomes_maladie.columns):
          for j, symptome2 in enumerate(symptomes_maladie.columns):
              fig.add_trace(go.Scatter(
                  x=[symptome],
                  y=[symptome2],
                  mode='markers',
                  marker=dict(size=50, color=symptomes_maladie_colors.iloc[i, j]),
                  showlegend=False
              ))

      # Mise en forme du graphe
      fig.update_layout(
          title=f"Corrélation des symptômes pour la maladie : {correspondance_classes[int(maladie)]}",
          xaxis=dict(title="Symptômes"),
          yaxis=dict(title="Symptômes"),
          height=600,
          margin=dict(l=50, r=50, b=100, t=100, pad=4),
          plot_bgcolor='rgba(0, 0, 0, 0)'
      )
      
      # Ajouter la figure à la liste des figures
      figs.append(fig)

  # Retourner la liste des figures en format JSON
  return [fig.to_json() for fig in figs]

  

    # # Sélectionner les colonnes de symptômes
    # cols_symptoms = df.drop(['age'], axis=1)

    # # Normaliser les données
    # scaler = StandardScaler()
    # X = scaler.fit_transform(cols_symptoms)

 
    # # Effectuer l'ACP pour réduire la dimensionnalité des données
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)

    # # Trouver le nombre de clusters
    # n_clusters = len(maladies)
    # # Faire le clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # kmeans.fit(X)
    # # Ajouter les données d'étiquette de cluster et créer le graphe
    # df['cluster'] = kmeans.labels_
    # fig = px.scatter(df, x=X_pca[:,0], y=X_pca[:,1], color='cluster', hover_data=cols_symptoms.columns)

    # # Personnaliser le graphe
    # fig.update_layout(title_text="Clustering des symptômes pour chaque maladie sélectionnée",
    #                   xaxis_title_text="Composante principale 1",
    #                   yaxis_title_text="Composante principale 2",
    #                   height=800, width=800)

    # # Convertir la figure en JSON
    # fig_json = fig.to_json()
    # return fig_json

def generate_graph6():
     # Récupérer les données de la requête JSON
    data = request.get_json()
    maladies = data.get('maladies') # Liste des maladies sélectionnées
    
    # Charger le jeu de données
    df = pd.read_csv("data/dermatology_csv.csv")
    
    # Filtrer les maladies sélectionnées
    df = df[df['class'].isin(map(int, maladies))]
    df = df.replace({'class': correspondance_classes})
    # Sélectionner les colonnes de symptômes
    cols_symptoms = df.drop(['age', 'family_history', 'class'], axis=1)

    # Normaliser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(cols_symptoms)

    # Effectuer l'ACP pour réduire la dimensionnalité des données
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Ajouter les données d'étiquette de maladie et créer le graphe
    df['class'] = df['class'].astype(str)
    fig = px.scatter(df, x=X_pca[:,0], y=X_pca[:,1], color='class', hover_data=cols_symptoms.columns)

    # Personnaliser le graphe
    fig.update_layout(title_text="ACP pour les maladies sélectionnées",
                      xaxis_title_text="Composante principale 1",
                      yaxis_title_text="Composante principale 2",
                      height=800, width=800)

    # Convertir la figure en JSON
    fig_json = fig.to_json()
    
    return fig_json

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

def generate_graph7():
  # Récupérer les données de la requête JSON
  data = request.get_json()
  maladies = data.get('maladies') # Liste des maladies sélectionnées

  # Obtenir les symptômes ayant une valeur de 3 pour chaque maladie
  symptomes = []
  colors = ['rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(255, 133, 27)', 'rgb(22, 96, 167)']
  for i, maladie in enumerate(maladies):
      symptomes_maladie = df[df['class'] == int(maladie)].iloc[:, :-2]
      symptomes_maladie = symptomes_maladie[symptomes_maladie.eq(3)].stack().reset_index()
      symptomes_maladie.columns = ['index', 'symptome', 'value']
      symptomes_maladie = symptomes_maladie.drop(['index', 'value'], axis=1)
      symptomes_maladie = symptomes_maladie.drop_duplicates()
      symptomes_maladie['color'] = colors[i]
      symptomes.append(symptomes_maladie['symptome'])
      colors.append(symptomes_maladie['color'][0])

  # Créer les données pour chaque maladie
  fig = make_subplots(rows=len(maladies), cols=1, shared_xaxes=True, vertical_spacing=0.05)
  for i, maladie in enumerate(maladies):
      top_symptomes = df[df['class'] == int(maladie)].iloc[:, :-2][symptomes[i]].sum().sort_values(ascending=False)
      top_symptomes_percent = (top_symptomes/top_symptomes.sum())*100
      data = [Bar(x=top_symptomes.index, y=top_symptomes.values, name=f"{correspondance_classes[int(maladie)]}", marker_color=colors[i], 
                  text=top_symptomes_percent.round(2).astype(str) + '%',
                  textposition='auto')]
      fig.add_traces(data, rows=[i+1], cols=[1])

  # Mise en forme du graphe
  fig.update_layout(title=f"Symptômes avec une forte présence pour chaque maladie",
                    yaxis=dict(title="Fréquence", range=[0, 300]),
                    height=600,
                    showlegend=True)

  # Conversion de la figure en JSON
  fig_json = fig.to_json()

  # Retourner le JSON
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

# # Créez la route pour afficher le graphe 3
# @app.route('/graph4', methods=['POST'])
# def graph4():
#   fig = generate_graph4()
#   return fig

# Créez la route pour afficher le graphe 5
@app.route('/graph5', methods=['POST'])
def graph5():
  fig = generate_graph5()
  return fig

@app.route('/graph6', methods=['POST'])
def graph6():
  fig = generate_graph6()
  return fig

@app.route('/graph7', methods=['POST'])
def graph7():
  fig = generate_graph7()
  return fig

# Créez la route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)