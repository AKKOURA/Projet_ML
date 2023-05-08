# train_model.py
import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Définir les chemins des dossiers de données
dir_train = 'server/data/train/'
dir_test = 'server/data/test/'

# Fonction pour charger les images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((64, 64))
        images.append(np.array(img))
    return images

# Charger les images à partir des deux dossiers
train_benign_images = load_images(os.path.join(dir_train, 'benign'))
train_malign_images = load_images(os.path.join(dir_train, 'malignant'))
test_benign_images = load_images(os.path.join(dir_test, 'benign'))
test_malign_images = load_images(os.path.join(dir_test, 'malignant'))

# Créer des labels pour les images
train_benign_labels = np.zeros(len(train_benign_images))
train_malign_labels = np.ones(len(train_malign_images))
test_benign_labels = np.zeros(len(test_benign_images))
test_malign_labels = np.ones(len(test_malign_images))

# Fusionner les images et les labels
X_train = np.concatenate((train_benign_images, train_malign_images))
y_train = np.concatenate((train_benign_labels, train_malign_labels))
X_test = np.concatenate((test_benign_images, test_malign_images))
y_test = np.concatenate((test_benign_labels, test_malign_labels))

# Aplatir les images en vecteurs de caractéristiques
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

# Créer un objet SVM et entraîner le modèle sur les données normalisées
svm = SVC(kernel='linear', C=1)
svm.fit(X_train_norm, y_train)

# Sauvegarder le modèle entraîné
model_path = "model/modelSVM.pkl"
joblib.dump(svm, model_path)