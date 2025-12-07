import os
import time

import numpy as np
from datetime import datetime
from google.cloud import storage
from sklearn.metrics import f1_score, classification_report

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
        batch_size=256,
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

def make_predictions(model: Model, X: np.ndarray) -> np.ndarray:
    """
    Génère des prédictions de classes (argmax) à partir des probabilités.
    """
    print(Fore.BLUE + f"\nPredicting on {len(X)} rows..." + Style.RESET_ALL)

    # model.predict renvoie des probabilités (ex: [0.1, 0.8, 0.1])
    y_pred_probs = model.predict(X)

    # On prend l'index de la probabilité la plus élevée
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print(f"✅ Predictions generated. Shape: {y_pred_classes.shape}")
    return y_pred_classes

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
