import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Charger les données depuis le fichier CSV
data = pd.read_csv('server/data/dermatology_csv.csv')
data = data.dropna()

# Diviser les données en features (X) et labels (y)
X = data.drop('class', axis=1)
y = data['class']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de décision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = decision_tree.predict(X_test)

# Calculer l'accuracy du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Sauvegarder le modèle entraîné
model_path = "server/model/modelDecisionTree.pkl"
joblib.dump(decision_tree, model_path)
