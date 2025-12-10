"""
Main pipeline for training the TEP fault detection model
"""
import logging
import numpy as np
import pandas as pd
from colorama import Fore, Style
from params import *

from ml_logic.preprocessor import preprocess_and_split
from ml_logic.model import (
    initialize_model_RNN,
    initialize_model_CNN,
    compile_model,
    train_model,
    evaluate_and_get_f1,
    save_model,
    make_predictions
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# CONFIGURATION - À adapter selon votre environnement
# ============================================================================
# GCP
GCP_PROJECT_NAME = "monitor-the-reactor"
BQ_DATASET = "TEP_Faulty_and_Free_Training"
BUCKET_NAME = "modele-monitor-the-reactor"

# Data
DATA_PATH = '/home/bapt/code/Monitor-the-Reactor/data/processed_data/TEP_Faulty_and_Free_Training_2.csv'
SAMPLE_DIVISION = 10  # Downsampling factor

# Model
MODEL_ARCHITECTURE = 'RNN'  # 'RNN' ou 'CNN'

# Training
PATIENCE = 5
BATCH_SIZE = 32

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale d'orchestration du pipeline ML avec validation croisée.
    """
    logging.info("🚀 Démarrage du pipeline d'entraînement...")
    logging.info(f"Architecture: {MODEL_ARCHITECTURE}")

    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis {DATA_PATH}")

        raw_df = pd.read_csv(DATA_PATH)
        raw_df = raw_df.iloc[::SAMPLE_DIVISION]

        if raw_df.empty:
            raise ValueError("Le dataset téléchargé est vide !")

        logging.info(f"✅ Données chargées. Shape: {raw_df.shape}")
        logging.info(f"Colonnes: {raw_df.columns.tolist()[:5]}...")

        # ---------------------------------------------------------
        # ÉTAPE 2 : Prétraitement et création des folds
        # ---------------------------------------------------------
        logging.info("⚙️ Préparation des folds de validation croisée...")
        all_folds = preprocess_and_split(raw_df)

        logging.info(f"✅ {len(all_folds)} folds créés")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Entraînement sur chaque fold
        # ---------------------------------------------------------
        fold_scores = []
        fold_models = []
        fold_histories = []

        for fold_data in all_folds:
            fold_num = fold_data['fold']

            logging.info(f"\n{'='*70}")
            logging.info(f"🏋️‍♂️  FOLD {fold_num}/{len(all_folds)}")
            logging.info(f"{'='*70}")

            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']

            logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            logging.info(f"Classes train: {np.unique(y_train)}")
            logging.info(f"Classes test: {np.unique(y_test)}")

            # A. Initialisation du modèle
            input_shape = X_train.shape[1:]

            if MODEL_ARCHITECTURE == 'CNN':
                model = initialize_model_CNN(input_shape)
            elif MODEL_ARCHITECTURE == 'RNN':
                model = initialize_model_RNN(input_shape)
            else:
                raise ValueError(f"Architecture {MODEL_ARCHITECTURE} non reconnue!")

            # B. Compilation
            model = compile_model(model, learning_rate=0.001)

            # C. Entraînement
            model, history = train_model(
                model,
                X_train,
                y_train,
                batch_size=BATCH_SIZE,
                patience=PATIENCE,
                validation_data=(X_test, y_test)
            )

            # D. Évaluation
            f1_score = evaluate_and_get_f1(model, X_test, y_test)

            fold_scores.append(f1_score)
            fold_models.append((model, f1_score, fold_num))
            fold_histories.append(history)

            logging.info(f"📊 Fold {fold_num} - F1 Score: {f1_score:.4f}")

            # E. Diagnostic overfitting
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            gap = train_acc - val_acc

            logging.info(f"   Train Acc: {train_acc:.4f}")
            logging.info(f"   Val Acc: {val_acc:.4f}")
            logging.info(f"   Gap (overfitting): {gap:.4f}")

            if gap > 0.3:
                logging.warning(f"⚠️  Overfitting détecté! Gap = {gap:.4f}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Résultats globaux
        # ---------------------------------------------------------
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        min_f1 = np.min(fold_scores)
        max_f1 = np.max(fold_scores)

        logging.info(f"\n{'='*70}")
        logging.info(f"📊 RÉSULTATS VALIDATION CROISÉE")
        logging.info(f"{'='*70}")
        logging.info(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"Min F1:  {min_f1:.4f}")
        logging.info(f"Max F1:  {max_f1:.4f}")
        logging.info(f"Scores: {[f'{s:.4f}' for s in fold_scores]}")

        if std_f1 > 0.1:
            logging.warning(f"⚠️  Variance élevée ({std_f1:.4f}) - possible overfitting")

        # ---------------------------------------------------------
        # ÉTAPE 5 : Sauvegarde du meilleur modèle
        # ---------------------------------------------------------
        best_model, best_f1, best_fold = max(fold_models, key=lambda x: x[1])

        logging.info(f"\n💾 Sauvegarde du meilleur modèle...")
        logging.info(f"   Fold: {best_fold}")
        logging.info(f"   F1 Score: {best_f1:.4f}")

        save_model(best_model, best_f1, bucket_name=BUCKET_NAME)

        # ---------------------------------------------------------
        # ÉTAPE 6 : Prédictions exemple (sur le meilleur fold)
        # ---------------------------------------------------------
        best_fold_data = all_folds[best_fold - 1]
        X_test_best = best_fold_data['X_test']

        logging.info("\n🔮 Génération de prédictions exemples...")
        random_indices = np.random.choice(X_test_best.shape[0], size=min(10, X_test_best.shape[0]), replace=False)
        sample_input = X_test_best[random_indices]

        preds, probs = make_predictions(best_model, sample_input)
        logging.info(f"✅ Prédictions: {preds}")

        # ---------------------------------------------------------
        # ÉTAPE 7 : Résumé final
        # ---------------------------------------------------------
        logging.info(f"\n{'='*70}")
        logging.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        logging.info(f"{'='*70}")
        logging.info(f"Meilleur modèle: Fold {best_fold} (F1={best_f1:.4f})")
        logging.info(f"Performance moyenne: {mean_f1:.4f} ± {std_f1:.4f}")

    except Exception as e:
        logging.error(f"❌ Erreur critique dans le pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
