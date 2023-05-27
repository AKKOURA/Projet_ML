import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Définir les chemins des dossiers de données
dir_train = 'server/data/train/'
dir_test = 'server/data/test/'

# Fonction pour charger les images et les redimensionner
def load_images(folder, target_size):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize(target_size)
        images.append(np.array(img))
    return images

# Charger les images à partir des deux dossiers
target_size = (224, 224)  # Taille d'entrée attendue par le modèle
train_benign_images = load_images(os.path.join(dir_train, 'benign'), target_size)
train_malign_images = load_images(os.path.join(dir_train, 'malignant'), target_size)
test_benign_images = load_images(os.path.join(dir_test, 'benign'), target_size)
test_malign_images = load_images(os.path.join(dir_test, 'malignant'), target_size)

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

X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

model = tf.keras.Sequential([
    Conv2D(50, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.22),
    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_norm, y_train, epochs=12, batch_size=32, validation_data=(X_test_norm, y_test))
loss, accuracy = model.evaluate(X_test_norm, y_test)

print("Loss sur les données de test:", loss)
print("Accuracy sur les données de test:", accuracy)
# Sauvegarder le modèle entraîné
model_path = "server/model/modelCNN.h5"
model.save(model_path)
