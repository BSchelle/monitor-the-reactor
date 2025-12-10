# scripts/ml_logic/evaluator.py

import numpy as np
from keras.models import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from colorama import Fore, Style
from interface.scripts.params import *
from interface.scripts.ml_logic.model import preprocess_and_predict_partial_bulk

def evaluate_prepared_data(
    X_test_3d: np.ndarray,
    y_true_seq: np.ndarray, # Labels 0-19
    model: Model,
    timesteps_available: int
) -> dict:
    """
    Évalue le modèle sur X_test, en tronquant les séquences à 'timesteps_available'
    et en appliquant le padding masqué pour la prédiction.
    """

    # 1. Troncation des séquences pour la détection précoce
    if timesteps_available > X_test_3d.shape[1]:
        raise ValueError("timesteps_available ne peut pas être supérieur à la longueur de la séquence.")

    X_to_predict = X_test_3d[:, :timesteps_available, :]

    print(f"✂️ Évaluation sur {X_to_predict.shape[0]} séquences tronquées à {timesteps_available} timesteps.")

    # 2. Prédiction (y_pred et y_conf sont en labels 1-20)
    # On passe le scaler en None car X_to_predict est déjà normalisé.
    y_pred, y_conf = preprocess_and_predict_partial_bulk(
        X_to_predict,
        scaler=None, # Non utilisé ici car X_test est déjà scalé
        model=model
    )

    # 3. Évaluation
    # Conversion de y_true_seq (0-19) à 1-20 pour l'aligner avec y_pred
    y_true_aligned = y_true_seq + 1

    results = {}

    print(Fore.GREEN + "\n📊 Résultats d'Évaluation :" + Style.RESET_ALL)
    acc = accuracy_score(y_true_aligned, y_pred)

    results['accuracy'] = acc
    print(f"   Accuracy sur {timesteps_available} timesteps: {acc:.4f}")

    target_names = [f'Faute {i}' for i in range(1, N_CLASSES + 1)]
    report = classification_report(y_true_aligned, y_pred, target_names=target_names, zero_division=0)
    results['classification_report'] = report
    print("\n   Rapport de Classification :\n", report)

    return results
