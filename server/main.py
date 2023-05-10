from flask import Flask, Response, render_template, request
from flask_bootstrap import Bootstrap
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from sklearn.decomposition import PCA
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
from werkzeug.utils import secure_filename
import os
import pickle
from PIL import Image
import dask.array as da
import dask.bag as db
from dask import delayed
import joblib


app = Flask(__name__)
bootstrap = Bootstrap(app)

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
                            text="Ce graphe montre le nombre de patients par type de maladie (ou classe) présente dans le dataset.",
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
        # Récupérer les données de la requête JSON
        data = request.get_json()
        maladies = data.get('maladies') # Liste des maladies sélectionnées

        # Charger le jeu de données
        df = pd.read_csv("data/dermatology_csv.csv")

        # Filtrer les maladies sélectionnées
        df = df[df['class'].isin(map(int, maladies))]
        df = df.replace({'class': correspondance_classes})

        # Sélectionner les colonnes de symptômes
        cols_symptoms = df.iloc[:, :-2]

        # Calculer la matrice de corrélation pour chaque maladie
        figs = []
        for maladie in maladies:
            maladie_df = cols_symptoms[df['class'] == maladie]
            corr_matrix = maladie_df.corr()

            # Créer le heatmap pour la matrice de corrélation
            fig_heatmap = px.imshow(corr_matrix, labels=dict(x="Symptômes", y="Symptômes"),
                                    x=maladie_df.columns, y=maladie_df.columns)
            fig_heatmap.update_layout(title_text=f"Corrélation des symptômes pour la maladie : {correspondance_classes[int(maladie)]}")

            # Normaliser les données
            scaler = StandardScaler()
            X = scaler.fit_transform(maladie_df)

            # Effectuer l'ACP pour réduire la dimensionnalité des données
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Ajouter les données d'étiquette de maladie et créer le graphe
            maladie_df['class'] = maladie_df['class'].astype(str)
            fig_scatter = px.scatter(maladie_df, x=X_pca[:,0], y=X_pca[:,1], color='class',
                                     hover_data=maladie_df.columns, title=f"ACP pour la maladie : {correspondance_classes[int(maladie)]}")

            # Personnaliser le graphe
            fig_scatter.update_layout(xaxis_title_text="Composante principale 1",
                                      yaxis_title_text="Composante principale 2",
                                      height=400, width=400)

            figs.append(fig_heatmap)
            figs.append(fig_scatter)

        # Convertir la figure en JSON
        figs_json = [fig.to_json() for fig in figs]

        return figs_json


  #-------------------------- SVM ------------------------------------------------------
  # Chemin vers le modèle entraîné
model_path = "model/modelSVM.pkl"
# Charger le modèle
pca, svm = joblib.load(model_path)

@app.route('/predict', methods=['GET', 'POST'])
def im():
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        if 'image' not in request.files:
            return render_template('Accueil.html', error='Aucun fichier téléchargé.')

        file = request.files['image']

        # Vérifier si le fichier a un nom valide
        if file.filename == '':
            return render_template('Accueil.html', error='Aucun fichier sélectionné.')

        # Vérifier si le fichier est une image
        if file and allowed_file(file.filename):
            # Enregistrer le fichier dans un dossier temporaire
            #mettre vos propre repertoire
            app.config['UPLOAD_FOLDER'] = '/Users/user'
            temp_path = os.path.join( app.config['UPLOAD_FOLDER'], file.filename)

            file.save(temp_path)
            # Charger l'image et effectuer la prédiction
            result = predict_image(temp_path)

            # Supprimer le fichier temporaire
            os.remove(temp_path)

            return render_template('prediction.html', result=result,image=file)

    return render_template('prediction.html')



def predict_image(image_path):
    # Charger l'image
    target_size = (64, 64)
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)

    # Aplatir l'image en vecteur de caractéristiques
            
    # Charger l'image et effectuer la prédiction
    img_flat = img.reshape(1, -1)
    img_norm = img_flat / 255.0

    X_test_pca = pca.transform(img_norm)

    # Effectuer la prédiction
    prediction = svm.predict(X_test_pca)

    # Renvoyer la prédiction (par exemple, en tant que chaîne de caractères)
    if prediction == 0:
         print("BENIN")
         return "Bénin"
       
    else:
        print("MALIN")
        return "Maligne"

    # fonction responsable pour la predicition des données numériques

def predict_from_form(form_data):
    # Charger les données du formulaire dans un DataFrame
    form_df = pd.DataFrame(form_data, index=[0])

    # Charger le modèle de décision tree entraîné
    decision_tree = joblib.load('model/modelDecisionTree.pkl')

    # Effectuer les prétraitements sur les données du formulaire (par exemple, encoder les catégories, normaliser les valeurs, etc.)
    # Assurez-vous d'appliquer les mêmes transformations que celles appliquées lors de l'entraînement du modèle

    # Effectuer la prédiction
    prediction = decision_tree.predict(form_df)

    # Convertir la prédiction en classe de maladie
    disease_classes = {
        1: 'Psoriasis',
        2: 'Seborrheic Dermatitis',
        3: 'Lichen Planus',
        4: 'Pityriasis Rosea',
        5: 'Chronic Dermatitis',
        6: 'Pityriasis Rubra Pilaris'
    }
    predicted_class = disease_classes[prediction]

    # Renvoyer la prédiction
    return predicted_class

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
                            text="Ce graphique en camembert montre le pourcentage de personne ayant eu la maladie sélectionnée de manière héréditaire ou non.",
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
                                text="Ce graphe montre le pourcentage d'hérédité des maladies sélectionnées.",
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

    # Charger le jeu de données
    df = pd.read_csv("data/dermatology_csv.csv")

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
        symptomes_maladie = df[df['class'] == int(maladie)][symptomes[i]].sum(axis=1)
        n_patients = symptomes_maladie.count()
        top_symptomes = symptomes_maladie[symptomes_maladie > 0].value_counts(normalize=True)
        top_symptomes_percent = top_symptomes * 100
        top_symptomes_percent = top_symptomes_percent.rename(lambda x: f"{x} ({top_symptomes[x]:.0f}/{n_patients})")
        data = [Bar(x=symptomes[i], y=top_symptomes_percent.values, name=f"{correspondance_classes[int(maladie)]}", marker_color=colors[i],
                    text=top_symptomes_percent.round(2).astype(str) + '%',
                    textposition='auto')]
        fig.add_traces(data, rows=[i+1], cols=[1])

    # Mise en forme du graphe
    fig.update_layout(title=f"Symptômes les plus fréquents pour chaque maladie",
                      xaxis=dict(title=""),
                      yaxis=dict(title="% de patients"),
                      height=600,
                      showlegend=True)

    # Conversion de la figure en JSON
    fig_json = fig.to_json()

    # Retourner le JSON
    return fig_json
    return render_template("prediction.html", symptomes=symptomes_maladie)

#Routage
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
    return render_template('Accueil.html')

@app.route('/prediction')
def form():
    return render_template('prediction.html')

@app.route('/statistique')
def analyse():
    return render_template('statistique.html')


# Définir les types de fichiers autorisés
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Vérifier si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Créer une route pour télécharger des images
@app.route('/upload', methods=['POST'])
def upload():
    # Vérifier si un fichier a été envoyé
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    # Récupérer le fichier envoyé
    file = request.files['image']

    # Vérifier si le nom de fichier est valide
    if file.filename == '':
        return 'No selected file', 400
    if not allowed_file(file.filename):
        return 'Invalid file type', 400

    # Enregistrer le fichier sur le serveur
    filename = secure_filename(file.filename)
    file.save('C:/Users/33766/Desktop/' + filename)

    # Retourner une réponse HTTP avec le nom de fichier
    return 'File uploaded successfully', 200

if __name__ == '__main__':
    app.run(debug=True)