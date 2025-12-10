"""
Deep Learning models for TEP fault detection
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from colorama import Fore, Style
from typing import Tuple
import pickle
import os

# ============================================================================
# INITIALISATION DES MODÈLES
# ============================================================================

def initialize_model_RNN(input_shape: tuple, num_classes: int = 11) -> Model:
    """
    Modèle RNN (LSTM) optimisé contre l'overfitting.

    Args:
        input_shape: (timesteps, features)
        num_classes: Nombre de classes à prédire

    Returns:
        Modèle Keras non compilé
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Première couche LSTM avec régularisation
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.01),
            recurrent_regularizer=regularizers.l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Deuxième couche LSTM
        layers.LSTM(
            32,
            kernel_regularizer=regularizers.l2(0.01),
            recurrent_regularizer=regularizers.l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Couches Dense
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),

        # Sortie
        layers.Dense(num_classes, activation='softmax')
    ])

    print(f"✅ RNN Model initialized")
    print(f"   Architecture: LSTM(64) -> LSTM(32) -> Dense(64) -> Dense({num_classes})")
    print(f"   Total params: {model.count_params():,}")

    return model


def initialize_model_RNN_simple(input_shape: tuple, num_classes: int = 11) -> Model:
    """
    Modèle RNN ultra-simple pour debug/baseline.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),  # Réduit de 128
        layers.Dropout(0.3),
        layers.LSTM(32),  # Réduit de 64
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),  # Réduit de 128
        layers.Dropout(0.3),
        layers.Dense(11, activation='softmax')
    ])
    print(f"✅ Simple RNN Model initialized")
    print(f"   Total params: {model.count_params():,}")

    return model


def initialize_model_CNN(input_shape: tuple, num_classes: int = 11) -> Model:
    """
    Modèle CNN 1D pour séries temporelles.

    Args:
        input_shape: (timesteps, features)
        num_classes: Nombre de classes

    Returns:
        Modèle Keras non compilé
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Première couche Conv1D
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Deuxième couche Conv1D
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Troisième couche Conv1D
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),

        # Couches Dense
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    print(f"✅ CNN Model initialized")
    print(f"   Total params: {model.count_params():,}")

    return model


# ============================================================================
# COMPILATION
# ============================================================================

def compile_model(model: Model, learning_rate: float = 0.001) -> Model:
    """
    Compile le modèle avec optimizer et loss.

    Args:
        model: Modèle Keras non compilé
        learning_rate: Taux d'apprentissage initial

    Returns:
        Modèle compilé
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model compiled (lr={learning_rate})")

    return model


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

def train_model(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    patience: int = 5,
    validation_data: tuple = None,
    validation_split: float = 0.2,
    verbose: int = 1
) -> Tuple[Model, dict]:
    """
    Entraîne le modèle avec callbacks et diagnostics.

    Args:
        model: Modèle compilé
        X: Données d'entraînement (3D pour RNN/CNN)
        y: Labels d'entraînement
        batch_size: Taille des batchs
        patience: Patience pour Early Stopping
        validation_data: (X_val, y_val) optionnel
        validation_split: Si validation_data=None, fraction pour validation
        verbose: Niveau de verbosité

    Returns:
        (model entraîné, history)
    """
    print(f"\n{Fore.BLUE}{'='*70}")
    print("TRAINING MODEL")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    # Diagnostics pré-entraînement
    print(f"📊 Dataset info:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y unique: {np.unique(y)}")
    print(f"   y distribution: {np.bincount(y.astype(int))}")

    if validation_data:
        X_val, y_val = validation_data
        print(f"\n   X_val shape: {X_val.shape}")
        print(f"   y_val unique: {np.unique(y_val)}")
        print(f"   y_val distribution: {np.bincount(y_val.astype(int))}")

        # Vérifier classes manquantes
        train_classes = set(np.unique(y))
        val_classes = set(np.unique(y_val))
        missing = train_classes - val_classes
        if missing:
            print(f"\n   ⚠️  Classes manquantes dans validation: {missing}")
            print(f"   ⚠️  Le modèle ne pourra pas apprendre ces classes!")

    # Callbacks
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(2, patience // 2),
        verbose=1,
        min_lr=1e-6
    )

    print(f"\n🏋️‍♂️ Starting training...")
    print(f"   Batch size: {batch_size}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Max epochs: 100\n")

    # Entraînement
    history = model.fit(
        X, y,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es, lr_scheduler],
        verbose=verbose
    )

    # Diagnostics post-entraînement
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    gap = train_acc - val_acc

    print(f"\n{Fore.GREEN}{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}{Style.RESET_ALL}")
    print(f"📈 Final metrics:")
    print(f"   Train Acc:  {train_acc:.4f}")
    print(f"   Val Acc:    {val_acc:.4f}")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")
    print(f"   Gap:        {gap:.4f}")

    if gap > 0.3:
        print(f"\n{Fore.RED}⚠️  OVERFITTING DÉTECTÉ (gap={gap:.4f}){Style.RESET_ALL}")
        print("   Suggestions:")
        print("   - Augmenter le dropout")
        print("   - Ajouter plus de régularisation L2")
        print("   - Réduire la complexité du modèle")
        print("   - Augmenter les données d'entraînement")
    elif val_acc < 0.5:
        print(f"\n{Fore.RED}⚠️  UNDERFITTING DÉTECTÉ (val_acc={val_acc:.4f}){Style.RESET_ALL}")
        print("   Suggestions:")
        print("   - Augmenter la complexité du modèle")
        print("   - Réduire le dropout")
        print("   - Augmenter le learning rate")
        print("   - Entraîner plus longtemps")
    else:
        print(f"\n{Fore.GREEN}✅ Bon équilibre biais/variance{Style.RESET_ALL}")

    return model, history


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_and_get_f1(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Évalue le modèle et calcule le F1-score macro.

    Returns:
        F1-score macro
    """
    print(f"\n{Fore.CYAN}{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    # Prédictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"✅ Predictions generated. Shape: {y_pred.shape}")

    # Métriques
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n📊 Metrics:")
    print(f"   F1 Macro:    {f1_macro:.4f}")
    print(f"   F1 Weighted: {f1_weighted:.4f}")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Diagnostic par classe
    print(f"\n🎯 Per-class analysis:")
    for cls in np.unique(y_test):
        mask = y_test == cls
        if mask.sum() > 0:
            acc = (y_pred[mask] == cls).mean()
            print(f"   Class {cls}: {acc:.2%} accuracy ({mask.sum()} samples)")

    return f1_macro


# ============================================================================
# PRÉDICTIONS
# ============================================================================

def make_predictions(model: Model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère des prédictions.

    Returns:
        (classes prédites, probabilités)
    """
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ============================================================================
# SAUVEGARDE
# ============================================================================

def save_model(model: Model, f1_score: float, bucket_name: str = None, local_path: str = "models"):
    """
    Sauvegarde le modèle localement et optionnellement sur GCS.

    Args:
        model: Modèle entraîné
        f1_score: Score F1 pour le nommage
        bucket_name: Nom du bucket GCS (optionnel)
        local_path: Chemin local pour sauvegarde
    """
    # Créer le dossier local
    os.makedirs(local_path, exist_ok=True)

    # Nom du fichier
    model_name = f"model_f1_{f1_score:.4f}.keras"
    local_file = os.path.join(local_path, model_name)

    # Sauvegarde locale
    model.save(local_file)
    print(f"\n💾 Modèle sauvegardé localement: {local_file}")

    # Sauvegarde GCS (si bucket fourni)
    if bucket_name:
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"models/{model_name}")
            blob.upload_from_filename(local_file)

            print(f"☁️  Modèle uploadé sur GCS: gs://{bucket_name}/models/{model_name}")
        except Exception as e:
            print(f"⚠️  Échec upload GCS: {e}")

    return local_file


def load_model(model_path: str) -> Model:
    """Charge un modèle sauvegardé."""
    model = keras.models.load_model(model_path)
    print(f"✅ Modèle chargé: {model_path}")
    return model


# ============================================================================
# VISUALISATION (OPTIONNEL)
# ============================================================================

def plot_history(history):
    """
    Visualise les courbes d'apprentissage.
    Nécessite matplotlib.
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_title('Loss Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.set_title('Accuracy Evolution', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("📊 Graphique sauvegardé: training_history.png")
        plt.show()

    except ImportError:
        print("⚠️  matplotlib non installé, graphique non généré")
