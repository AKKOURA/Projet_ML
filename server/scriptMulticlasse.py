# Définir les chemins des dossiers de données
data_dir = 'server/data/data_classe/'

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf


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



# Convertir les étiquettes en encodage catégorique
y_train_cat = to_categorical(y_train, num_classes=7)
y_test_cat = to_categorical(y_test, num_classes=7)

model = tf.keras.Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.22),
    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_norm, y_train_cat, epochs=5, validation_data=(X_test_norm, y_test_cat))
# Sauvegarder le modèle entraîné
model_path = "server/model/modelMulticlasse.pkl"
joblib.dump(model, model_path)