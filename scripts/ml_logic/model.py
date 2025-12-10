import os
import time

import numpy as np
from datetime import datetime
from google.cloud import storage
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from scripts.ml_logic.preprocessor import preprocess_and_predict_partial_bulk_masked
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
        layers.Dense(12, activation='softmax')
    ])

    print("✅ Model initialized")

    return model

# def initialize_model_RNN(input_shape: tuple) -> Model:
#     """
#     Initialize the Neural Network with random weights
#     """
#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.LSTM(128, return_sequences=True),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),
#         layers.LSTM(64),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(11, activation='softmax')  # 20 classes (0-19)
#     ])

#     print("✅ Model initialized")

#     return model

def initialize_model_RNN(input_shape: tuple) -> Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.LSTM(48, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(11, activation='softmax')
    ])
    return model



def initialize_model_CNN_pad(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential([
        layers.Masking(mask_value=MASK_VALUE, input_shape=input_shape),
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

def initialize_model_RNN_pad(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = keras.Sequential([
        layers.Masking(mask_value=MASK_VALUE, input_shape=input_shape),
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
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    # Early stopping plus strict
    es = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        min_delta=0.001
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
    y_pred, _ = make_predictions(model, X_test)

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
    model,
    mask_value=MASK_VALUE,
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
    padding_values = np.full(
            (N, padding_needed, N_FEATURES),
            mask_value,
            dtype=X_partial_3d.dtype # Utiliser le même type de données que l'entrée
        )
    # Concaténation le long de l'axe des timesteps (axe 1)
    X_padded_final = np.concatenate((X_scaled_3d, padding_values), axis=1) # Forme (N, 50, 52)

    # 5. Prédiction
    y_pred_probs = model.predict(X_padded_final)

    # 6. Post-traitement (labels 1-20 et confiance)
    y_pred_classes_indices = np.argmax(y_pred_probs, axis=1)
    y_pred_classes = y_pred_classes_indices + 1
    y_pred_confidence = y_pred_probs[np.arange(N), y_pred_classes_indices]

    return y_pred_classes, y_pred_confidence


def evaluate_simulation(
    df_raw,
    model,
    scaler,
    timesteps_available: int
) -> dict:
    """
    Charge les données d'une simulation, applique le preprocessing (normalisation + padding masqué),
    génère les prédictions, et calcule les métriques si les labels sont présents.

    Args:
        df_raw (pd.DataFrame): Données brutes de la simulation.
        model (Model): Le modèle Keras entraîné avec la couche Masking.
        scaler_path (str): Chemin vers l'objet scaler entraîné (e.g., 'scaler_final.pkl').
        timesteps_available (int): Nombre de timesteps réels à utiliser pour la prédiction (ex: 20 ou 50).

    Returns:
        dict: Résultats des prédictions et métriques.
    """
    results = {}

    # try:
    #     # 1. Chargement du Scaler
    #     with open(scaler_path, "rb") as f:
    #         scaler = pickle.load(f)
    #     print(f"✅ Scaler chargé depuis {scaler_path}")

    # except FileNotFoundError:
    #     print(Fore.RED + f"❌ Erreur: Scaler non trouvé à {scaler_path}" + Style.RESET_ALL)
    #     return {"error": "Scaler file not found"}

    # --- Préparation des données brutes ---

    # Séparation X et Y (y est 'faultNumber')
    # Assurez-vous que df_raw a les mêmes colonnes que l'entraînement après chargement/renommage
    if 'faultNumber' in df_raw.columns:
        X_df = df_raw.drop(columns=['faultNumber', 'simulationRun', 'sample'])
        y_true_df = df_raw['faultNumber']
        labels_present = True
    else:
        # Cas d'une simulation en temps réel sans label
        X_df = df_raw.drop(columns=['simulationRun', 'sample'])
        y_true_df = None
        labels_present = False

    X_np = X_df.values

    # 2. Reshape des données en séquences complètes (N, T, F)
    # Calculez la longueur tronquée pour assurer la division entière
    num_samples_to_keep = len(X_np) - (len(X_np) % 50)
    X_np = X_np[:num_samples_to_keep]

    num_sequences = len(X_np) // 50

    # Reshape 2D -> 3D pour la gestion des séquences
    X_3d_full = X_np.reshape(num_sequences, 50, 52)

    # 3. Troncation si nécessaire (Ex: réduire 50 timesteps à 20 pour le test d'alerte précoce)
    # Cette étape est cruciale si timesteps_available < TIMESTEPS_ENTRAINEMENT
    if timesteps_available < 50:
        print(f"✂️ Troncation des séquences de {50} à {timesteps_available} timesteps.")
        X_to_predict = X_3d_full[:, :timesteps_available, :]
    else:
        X_to_predict = X_3d_full # Utilise 50 timesteps

    # --- Prédiction (Utilisation de la fonction masquée) ---

    # La fonction preprocess_and_predict_partial_bulk_masked doit exister et utiliser MASK_VALUE = -5.0
    y_pred, y_conf = preprocess_and_predict_partial_bulk_masked(
        X_to_predict,
        scaler,
        model
    )

    results['predictions'] = y_pred
    results['confidence'] = y_conf
    results['num_sequences'] = num_sequences

    # --- 4. Évaluation ---

    if labels_present:
        # Extrait le vrai label par séquence (doit être fait avant le trimming initial)
        y_true_seq = y_true_df.iloc[::50].values[:num_sequences]

        # Vérification: y_true_seq contient des labels de 1 à 20. Les prédictions y_pred aussi.

        results['true_labels'] = y_true_seq

        print(Fore.GREEN + "\n📊 Résultats d'Évaluation :" + Style.RESET_ALL)

        # A. Précision
        acc = accuracy_score(y_true_seq, y_pred)
        results['accuracy'] = acc
        print(f"   Accuracy: {acc:.4f}")

        # B. Rapport de Classification
        target_names = [f'Faute {i}' for i in range(1, 21)]
        report = classification_report(y_true_seq, y_pred, target_names=target_names, zero_division=0)
        results['classification_report'] = report
        print("\n   Rapport de Classification :\n", report)

        # C. Matrice de Confusion
        cm = confusion_matrix(y_true_seq, y_pred)
        results['confusion_matrix'] = cm
        #

    return results

import matplotlib.pyplot as plt

def plot_learning_curves(history):
    """Visualise les courbes d'apprentissage"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Evolution')

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Evolution')

    plt.tight_layout()
    plt.show()
