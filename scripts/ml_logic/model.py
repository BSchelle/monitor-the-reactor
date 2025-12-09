import os
import time

import numpy as np
from datetime import datetime
from google.cloud import storage
from sklearn.metrics import f1_score, classification_report
from scripts.params import *

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model_CNN(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(512, kernel_size=1, activation='relu', padding='same'),
        layers.BatchNormalization(),
        # layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(21, activation='softmax')
    ])

    print("✅ Model initialized")

    return model

def initialize_model_RNN(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(20, activation='softmax')  # 20 classes (0-19)
    ])

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        patience=10,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    lr = ReduceLROnPlateau(factor=0.5, patience=5)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es,lr],
        verbose=1
    )
    val_accuracy = np.max(history.history['val_accuracy'])
    print(f"✅ Model trained. Max Val Accuracy: {round(val_accuracy, 4)}")

    return model, history

def make_predictions(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère des prédictions de classes (argmax) et leur confiance,
    en ajustant l'indexation (0-19) vers les labels (1-20).
    """
    print(Fore.BLUE + f"\nPredicting on {len(X)} rows..." + Style.RESET_ALL)

    # 1. Prédiction des probabilités
    y_pred_probs = model.predict(X)

    # 2. Détermination de l'index de la probabilité la plus élevée (0-19)
    y_pred_classes_indices = np.argmax(y_pred_probs, axis=1)

    # 3. CORRECTION : AJOUTER 1 pour passer de l'index 0-19 au label 1-20
    # Le résultat final est la liste des labels prédits (1, 2, ..., 20)
    y_pred_classes = y_pred_classes_indices + 1

    # 4. Calcul de la confiance (la probabilité maximale)
    y_pred_confidence = y_pred_probs[np.arange(len(y_pred_probs)), y_pred_classes_indices]

    print(f"✅ Predictions generated. Shape: {y_pred_classes.shape}")

    # Retourne les labels (1-20) et leur confiance
    return y_pred_classes, y_pred_confidence

def evaluate_and_get_f1(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Fonction utilitaire pour prédire et calculer le Macro F1 Score
    Nécessaire avant de sauver le modèle.
    """
    y_pred = make_predictions(model, X_test)

    # Calcul du F1 Score (Macro pour multiclasse équilibré/déséquilibré)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"✅ Macro F1 Score: {round(f1, 4)}")
    return f1

def save_model(model: Model, f1: float, bucket_name=None) -> None:
    """
    Sauvegarde le modèle avec un timestamp et le score F1.
    Gère la sauvegarde locale et l'upload vers GCS.
    """
    # 1. Génération du nom unique
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # On formatte le F1 score à 4 décimales pour la lisibilité
    filename = f"model_{timestamp}_f1-{f1:.4f}.keras"

    # Création du dossier local 'models' s'il n'existe pas
    local_model_dir = "models"
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    local_path = os.path.join(local_model_dir, filename)

    # 2. Sauvegarde Locale
    print(Fore.BLUE + f"\nSaving model locally to {local_path}..." + Style.RESET_ALL)
    model.save(local_path)
    print("✅ Model saved locally")

    # 3. Upload vers GCS (Optionnel)
    if bucket_name:
        try:
            print(Fore.BLUE + f"Uploading to GCS bucket: {bucket_name}..." + Style.RESET_ALL)

            # Initialisation du client GCS
            # (L'authentification est automatique sur GCC si les scopes sont bons)
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # On garde la même structure 'models/' dans le bucket
            blob = bucket.blob(f"models/{filename}")
            blob.upload_from_filename(local_path)

            print(f"✅ Model uploaded to gs://{bucket_name}/models/{filename}")

        except Exception as e:
            print(f"\n❌ Error uploading to GCS: {e}")
            # On ne raise pas l'erreur pour ne pas crasher le script si juste l'upload échoue,
            # car le modèle est déjà sauvé en local.

def load_model(model_name: str,  bucket_name=None) -> Model:

    if bucket_name:
        try:
            print(Fore.BLUE + f"Loading from GCS bucket: {bucket_name}..." + Style.RESET_ALL)

            model_path = f'gs://{bucket_name}/models/{model_name}'
            instantiated_model = keras.models.load_model(model_path)

            print(f"✅ Model loaded from gs://{bucket_name}/models/{model_name}")

            instantiated_model.summary()

        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
    else:
        instantiated_model = keras.models.load_model("/home/bapt/code/Monitor-the-Reactor/notebooks/Explo_BS/models_keras/models_model_20251209-171751_f1-0.6531.keras")

    return instantiated_model

import numpy as np

def preprocess_and_predict_partial_bulk(
    X_partial_3d: np.ndarray, # Nommer X_partial_3d pour plus de clarté
    scaler,
    model
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique la normalisation et le padding à un ensemble de séquences partielles 3D.
    """
    N, TIMESTEPS_PARTIAL, N_FEATURES = X_partial_3d.shape
    TIMESTEPS_ENTRAINEMENT = 50 # Assurez-vous d'avoir cette constante

    # 1. Reshape de 3D vers 2D pour la normalisation
    X_2d_raw = X_partial_3d.reshape(-1, N_FEATURES)

    # 2. Normalisation (UNIQUEMENT .transform())
    X_scaled_2d = scaler.transform(X_2d_raw)

    # 3. Reshape vers 3D (N, 20, 52)
    X_scaled_3d = X_scaled_2d.reshape(N, TIMESTEPS_PARTIAL, N_FEATURES)

    # 4. Padding Temporel (N, 20, 52) -> (N, 50, 52)
    padding_needed = TIMESTEPS_ENTRAINEMENT - TIMESTEPS_PARTIAL # 50 - 20 = 30

    # Création du padding (N, 30, 52)
    # Note : Le padding doit être ajouté à la dernière dimension (timesteps)
    zero_padding = np.zeros((N, padding_needed, N_FEATURES), dtype=X_partial_3d.dtype)

    # Concaténation le long de l'axe des timesteps (axe 1)
    X_padded_final = np.concatenate((X_scaled_3d, zero_padding), axis=1) # Forme (N, 50, 52)

    # 5. Prédiction
    y_pred_probs = model.predict(X_padded_final)

    # 6. Post-traitement (labels 1-20 et confiance)
    y_pred_classes_indices = np.argmax(y_pred_probs, axis=1)
    y_pred_classes = y_pred_classes_indices + 1
    y_pred_confidence = y_pred_probs[np.arange(N), y_pred_classes_indices]

    return y_pred_classes, y_pred_confidence
