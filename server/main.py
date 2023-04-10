from flask import Flask, Response, render_template
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
from threading import Thread

app = Flask(__name__)

# Charger vos données dans un dataframe
df = pd.read_csv("data/dermatology_csv.csv")

# Créez la fonction pour générer le graphe 1
def generate_graph1():
  fig = sns.countplot(x=df["class"])
  plt.title("Le nombre de personnes par classe")
  plt.xlabel("Type de maladie")
  buffer = BytesIO()
  fig.figure.savefig(buffer, format='png')
  buffer.seek(0)
  return buffer.getvalue()

def generate_graph2():
  # Créer le graphe
 # Les ages de patients pour chaque classe pour identifier si possible la tranche d’age de chaque maladie
  fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
  axs = axs.flatten()
  for i in range(1, 7):
      ax = axs[i-1]
      df[df['class']==i]['age'].plot(kind='hist', bins=range(0, 100, 5), color='green', ax=ax)
      ax.set_title('Classe {}'.format(i))
      ax.set_xlabel('Age')
      ax.set_ylabel('Nombre de patients')
  buffer = BytesIO()
  fig.savefig(buffer, format='png')
  buffer.seek(0)
  return buffer.getvalue()

# Créez la route pour afficher le graphe 1
@app.route('/graph1')
def graph1():
  buffer = generate_graph1()
  return Response(buffer, mimetype='image/png')

# Créez la route pour afficher le graphe 2 
@app.route('/graph2')
def graph2():
  buffer = generate_graph2()
  return Response(buffer, mimetype='image/png')

# Créez la route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)