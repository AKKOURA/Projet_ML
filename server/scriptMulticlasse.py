# Définir les chemins des dossiers de données
data_dir = 'server/data/tache_peau/'

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib


# Obtenir une liste des sous-dossiers dans le dossier data_dir
class_names = os.listdir(data_dir)
# Initialiser des tableaux numpy vides pour stocker les images et les étiquettes de classe
images = []
labels = []

# Définir la taille de l'image redimensionnée
img_size = (64, 64)

# Boucler sur les sous-dossiers de classe
for i, class_name in enumerate(class_names):
    # Obtenir le chemin complet vers le sous-dossier de classe
    class_path = os.path.join(data_dir, class_name)

    # Boucler sur les fichiers d'image dans le sous-dossier de classe
    for file_name in os.listdir(class_path):
        # Obtenir le chemin complet vers l'image
        image_path = os.path.join(class_path, file_name)
        # Charger l'image à l'aide de la bibliothèque Pillow
        image = Image.open(image_path)

        # Redimensionner l'image
        image = image.resize(img_size)

        # Convertir l'image en un tableau numpy et l'ajouter au tableau d'images
        images.append(np.array(image))

        # Ajouter l'étiquette de classe correspondante au tableau d'étiquettes
        labels.append(i)

# Convertir les tableaux d'images et d'étiquettes en tableaux numpy
images = np.array(images)
labels = np.array(labels)

# Séparer les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Aplatir les images en vecteurs de caractéristiques pour la classification
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0


#Déclaration du RandomForest
randomforest=RandomForestClassifier(n_estimators=40)
#Entrainement du RandomForest
randomforest.fit(X_train_norm , y_train)
#Calcul de la performance du RandomForest en cross-validation
score = cross_val_score(randomforest, X_test_norm , y_test,cv=10)
print ("Sur ce jeu de données, le taux de succès en classification moyen de :",score.mean())
#Calcul de la prédiction du Random Forest pour le jeu de données X
y_predit=randomforest.predict(X_test_norm )
# Sauvegarder le modèle entraîné
model_path = "server/model/modelMulticlasse.pkl"
joblib.dump(randomforest, model_path)