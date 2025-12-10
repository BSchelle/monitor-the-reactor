import numpy as np
import pandas as pd

from pathlib import Path
from google.cloud import bigquery
from colorama import Fore, Style

from scripts.get_data import get_data
from scripts.ml_logic.preprocessor import preprocess_and_split, \
    load_preprocessor, preprocess_simulation, prepare_training_data_masked

from scripts.ml_logic.model import initialize_model_CNN, initialize_model_RNN, \
    compile_model, train_model, make_predictions, evaluate_and_get_f1, \
        save_model, load_model, preprocess_and_predict_partial_bulk,\
            initialize_model_CNN_pad, initialize_model_RNN_pad, evaluate_simulation, plot_learning_curves

from scripts.params import *

import logging
import sys
import os

# Configuration du logging (Indispensable pour le débogage sur le Cloud)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Configuration de la seed pour le sampling du test set
np.random.seed(42)

def main():
    logging.info("🚀 Démarrage du pipeline d'entraînement...")

    try:
        # Chargement données
        raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/TEP_Faulty_and_Free_Training_2.csv')
        raw_df = raw_df.iloc[::SAMPLE_DIVISION]

        # Prétraitement
        logging.info("⚙️ Préparation des folds de validation croisée...")
        all_folds = preprocess_and_split(raw_df)

        # 🎯 Entraînement sur chaque fold
        fold_scores = []
        fold_models = []

        for fold_data in all_folds:
            fold_num = fold_data['fold']
            logging.info(f"\n{'='*50}")
            logging.info(f"🏋️‍♂️ Entraînement Fold {fold_num}/5")
            logging.info(f"{'='*50}")

            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']

            # Initialisation modèle
            input_shape = X_train.shape[1:]
            if MODEL_ARCHITECTURE == 'CNN':
                model = initialize_model_CNN(input_shape)
            elif MODEL_ARCHITECTURE == 'RNN':
                model = initialize_model_RNN(input_shape)
            else:
                raise ValueError("Architecture non reconnue!")

            model = compile_model(model)

            # Entraînement
            model, history = train_model(
                model, X_train, y_train,
                validation_data=(X_test, y_test),
                patience=5
            )

            # Évaluation
            f1 = evaluate_and_get_f1(model, X_test, y_test)
            fold_scores.append(f1)
            fold_models.append((model, f1, fold_num))

            logging.info(f"📊 Fold {fold_num} - F1 Score: {f1:.4f}")

        # 📈 Résultats globaux
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        logging.info(f"\n{'='*50}")
        logging.info(f"📊 RÉSULTATS VALIDATION CROISÉE")
        logging.info(f"{'='*50}")
        logging.info(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"Scores par fold: {[f'{s:.4f}' for s in fold_scores]}")

        # Sauvegarder le meilleur modèle
        best_model, best_f1, best_fold = max(fold_models, key=lambda x: x[1])
        logging.info(f"💾 Sauvegarde du meilleur modèle (Fold {best_fold}, F1={best_f1:.4f})")
        save_model(best_model, best_f1, bucket_name=BUCKET_NAME)

    except Exception as e:
        logging.error(f"❌ Erreur: {e}")
        raise

def main_1():
    """
    Fonction principale d'orchestration du pipeline ML.
    """
    logging.info("🚀 Démarrage du pipeline d'entraînement...")

    # 1. Configuration et Paramètres
    # Idéalement, utilisez des variables d'environnement sur le Cloud
    MODEL_PATH = "model.joblib" # Ou un chemin GCS: gs://mon-bucket/model.joblib

    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données (BigQuery)
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis BigQuery : {GCP_PROJECT_NAME}.{BQ_DATASET}")

        # raw_df = get_data(project_id=GCP_PROJECT_NAME,
        #                   dataset=BQ_FAULTY_TRAIN,
        #                   col_to_keep=COLUMN_NAMES,
        #                   sample_division=SAMPLE_DIVISION,
        #                 #   fault=tuple(i for i in range(21) if i not in [9,10,15,16])
                        # )

        raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/TEP_Faulty_and_Free_Training_2.csv')
        raw_df = raw_df.iloc[::SAMPLE_DIVISION]
        # raw_df.sort_values(["simulationRun", "faultNumber", "sample"]).

        if raw_df.empty:
            raise ValueError("Le dataset téléchargé est vide !")
        logging.info(f"✅ Données chargées. Shape: {raw_df.shape}")

        # ---------------------------------------------------------
        # ÉTAPE 2 : Prétraitement (Cleaning / Feature Engineering)
        # ---------------------------------------------------------
        logging.info("⚙️ Traitement des données et split Train/Test...")
        # On suppose que process_data renvoie les sets divisés
        X_train, X_test, y_train, y_test = preprocess_and_split(raw_df)
        logging.info(f"✅ Données traitées. X_train shape: {X_train.shape}")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Entraînement (Sur la VM GCC)
        # ---------------------------------------------------------
        logging.info(f"🏋️‍♂️ Initialisation du modèle {MODEL_ARCHITECTURE} en cours...")
        input_shape = X_train.shape[1:]
        if MODEL_ARCHITECTURE == 'CNN':
            model = initialize_model_CNN(input_shape)
        elif MODEL_ARCHITECTURE == 'RNN':
            model = initialize_model_RNN(input_shape)
        else :
            raise ValueError("Aucune architecture de DL initialisée !")

        logging.info("🏋️‍♂️ Compilation du modèle en cours...")
        model = compile_model(model)
        logging.info("🏋️‍♂️ Entraînement du modèle en cours...")
        model, history = train_model(model, X_train, y_train, patience=3)
        logging.info("✅ Modèle entraîné avec succès.")

        # Dans votre main():
        plot_learning_curves(history)

        # ---------------------------------------------------------
        # ÉTAPE 5 : Test et Métriques
        # ---------------------------------------------------------
        logging.info("📊 Évaluation des performances...")
        f1 = evaluate_and_get_f1(model, X_test, y_test)
        logging.info(f"📈 Résultats du test : {f1}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Sauvegarde du modèle
        # ---------------------------------------------------------
        logging.info(f"💾 Sauvegarde du modèle vers {MODEL_PATH}...")
        save_model(model, f1, bucket_name=BUCKET_NAME)

        # ---------------------------------------------------------
        # ÉTAPE 6 : Prédictions (Optionnel ou sur un set de validation)
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        random_indices = np.random.choice(X_test.shape[0], size=10, replace=False)
        sample_input = X_test[random_indices]
        preds, _ = make_predictions(model, sample_input)
        logging.info(f"✅ Prédictions terminées : {preds}")

    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")


def load_eval():
    """
    Fonction d'orchestration du chargement et de l'évaluation d'un modèle ML.
    """
    logging.info("🚀 Démarrage du pipeline d'évaluation...")

    # 1. Configuration et Paramètres
    # Idéalement, utilisez des variables d'environnement sur le Cloud
    MODEL_PATH = "model.joblib" # Ou un chemin GCS: gs://mon-bucket/model.joblib


    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données (BigQuery)
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis BigQuery : {GCP_PROJECT_NAME}.{BQ_DATASET}")

        # raw_df = get_data(project_id=GCP_PROJECT_NAME,
        #                   dataset=BQ_FAULTY_TRAIN,
        #                   col_to_keep=COLUMN_NAMES,
        #                   sample_division=10,
        #                   number_simulations=5,
        #                   fault=None)

        raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/faulty_train_fault1_sim1.csv')
        raw_df = raw_df[::10]
        # ---------------------------------------------------------
        # ÉTAPE 2 : Chargement du scaler
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement du scaler")
        scaler = load_preprocessor('scripts/ml_logic/scaler/scaler.pkl')
        logging.info(f"✅ Scaler chargé ")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Prétraitement du dataset
        # ---------------------------------------------------------
        logging.info("⚙️ Traitement des données et split X_eval, y_eval")
        # On suppose que process_data renvoie les sets divisés
        X_eval, y_eval = preprocess_simulation(raw_df, scaler,
                                     timesteps_per_sequence= 50)
        logging.info(f"✅ Données traitées. X_eval shape: {X_eval.shape}, y_eval shape : {y_eval.shape}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Chargement du modèle entrainé
        # ---------------------------------------------------------
        logging.info("⚙️ Chargement du modèle : {MODEL_NAME}")
        model = load_model(model_name = MODEL_NAME)
        logging.info("✅ Modèle : {MODEL_NAME} chargé")

        # ---------------------------------------------------------
        # ÉTAPE 5 : Prediction de X_eval
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        preds, confidence = make_predictions(model, X_eval)
        print(preds)
        print(confidence)
        logging.info(f"✅ Prédictions terminées : {preds}")


    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")

def load_eval_partial():
    """
    Fonction d'orchestration du chargement et de l'évaluation d'un modèle ML.
    """
    logging.info("🚀 Démarrage du pipeline d'évaluation...")

    # 1. Configuration et Paramètres
    # Idéalement, utilisez des variables d'environnement sur le Cloud
    MODEL_PATH = "model.joblib" # Ou un chemin GCS: gs://mon-bucket/model.joblib


    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données (BigQuery)
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis BigQuery : {GCP_PROJECT_NAME}.{BQ_DATASET}")

        # raw_df = get_data(project_id=GCP_PROJECT_NAME,
        #                   dataset=BQ_FAULTY_TRAIN,
        #                   col_to_keep=COLUMN_NAMES,
        #                   sample_division=10,
        #                   number_simulations=1,
        #                   fault=None)

        raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/faulty_train_fault1_sim1.csv')
        raw_df = raw_df[:300:10]
        # ---------------------------------------------------------
        # ÉTAPE 2 : Chargement du scaler
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement du scaler")
        scaler = load_preprocessor('scripts/ml_logic/scaler/scaler.pkl')
        logging.info(f"✅ Scaler chargé ")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Prétraitement du dataset
        # ---------------------------------------------------------
        logging.info("⚙️ Traitement des données et split X_eval, y_eval")
        # On suppose que process_data renvoie les sets divisés
        X_eval, y_eval = preprocess_simulation(raw_df, scaler,
                                     timesteps_per_sequence= 50)
        logging.info(f"✅ Données traitées. X_eval shape: {X_eval.shape}, y_eval shape : {y_eval.shape}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Chargement du modèle entrainé
        # ---------------------------------------------------------
        logging.info("⚙️ Chargement du modèle : {MODEL_NAME}")
        model = load_model(model_name = MODEL_NAME, bucket_name=BUCKET_NAME)
        logging.info("✅ Modèle : {MODEL_NAME} chargé")

        # ---------------------------------------------------------
        # ÉTAPE 5 : Prediction de X_eval
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        preds, confidence = preprocess_and_predict_partial_bulk(
            X_eval[:,:,:],
            scaler,
            model
        )
        print(preds)
        print(confidence)
        logging.info(f"✅ Prédictions terminées : {preds}")


    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")

def train_padded():
    """
    Fonction principale d'orchestration du pipeline ML.
    """
    logging.info("🚀 Démarrage du pipeline d'entraînement...")

    # 1. Configuration et Paramètres
    # Idéalement, utilisez des variables d'environnement sur le Cloud
    MODEL_PATH = "model.joblib" # Ou un chemin GCS: gs://mon-bucket/model.joblib

    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données (BigQuery)
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis BigQuery : {GCP_PROJECT_NAME}.{BQ_DATASET}")

        raw_df = get_data(project_id=GCP_PROJECT_NAME,
                dataset=BQ_FAULTY_TRAIN,
                col_to_keep=COLUMN_NAMES,
                sample_division=10,
                number_simulations=500)

        # raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/ftrain_500sim_selectd_flt_sample10.csv')

        if raw_df.empty:
            raise ValueError("Le dataset téléchargé est vide !")
        logging.info(f"✅ Données chargées. Shape: {raw_df.shape}")


        # ---------------------------------------------------------
        # ÉTAPE 2 : Prétraitement (Cleaning / Feature Engineering)
        # ---------------------------------------------------------
        logging.info("⚙️ Traitement des données et split Train/Test...")
        # On suppose que process_data renvoie les sets divisés
        X_train, X_test, y_train, y_test, scaler = prepare_training_data_masked(raw_df)
        logging.info(f"✅ Données traitées. X_train shape: {X_train.shape}")


        # ---------------------------------------------------------
        # ÉTAPE 3 : Entraînement (Sur la VM GCC)
        # ---------------------------------------------------------
        logging.info(f"🏋️‍♂️ Initialisation du modèle {MODEL_ARCHITECTURE} en cours...")
        input_shape = X_train.shape[1:]
        if MODEL_ARCHITECTURE == 'CNN':
            model = initialize_model_CNN_pad(input_shape)
        elif MODEL_ARCHITECTURE == 'RNN':
            model = initialize_model_RNN_pad(input_shape)
        else :
            raise ValueError("Aucune architecture de DL initialisée !")

        logging.info("🏋️‍♂️ Compilation du modèle en cours...")
        model = compile_model(model)
        logging.info("🏋️‍♂️ Entraînement du modèle en cours...")
        model, history = train_model(model, X_train, y_train, patience=5)
        logging.info("✅ Modèle entraîné avec succès.")

        # ---------------------------------------------------------
        # ÉTAPE 5 : Test et Métriques
        # ---------------------------------------------------------
        logging.info("📊 Évaluation des performances...")
        test_set = pd.concat((y_test, X_test), axis=1)
        evaluate_simulation(
            test_set,
            model,
            scaler,
            timesteps_available=20
        )
        # logging.info(f"📈 Résultats du test : {f1}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Sauvegarde du modèle
        # ---------------------------------------------------------
        logging.info(f"💾 Sauvegarde du modèle vers {MODEL_PATH}...")
        save_model(model, f1, bucket_name=BUCKET_NAME)

        # ---------------------------------------------------------
        # ÉTAPE 6 : Prédictions (Optionnel ou sur un set de validation)
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        random_indices = np.random.choice(X_test.shape[0], size=10, replace=False)
        sample_input = X_test[random_indices]
        preds, _ = make_predictions(model, sample_input)
        logging.info(f"✅ Prédictions terminées : {preds}")

    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")



def load_eval_partial():
    """
    Fonction d'orchestration du chargement et de l'évaluation d'un modèle ML.
    """
    logging.info("🚀 Démarrage du pipeline d'évaluation...")

    # 1. Configuration et Paramètres
    # Idéalement, utilisez des variables d'environnement sur le Cloud
    MODEL_PATH = "model.joblib" # Ou un chemin GCS: gs://mon-bucket/model.joblib


    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Chargement des données (BigQuery)
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement des données depuis BigQuery : {GCP_PROJECT_NAME}.{BQ_DATASET}")

        # raw_df = get_data(project_id=GCP_PROJECT_NAME,
        #                   dataset=BQ_FAULTY_TRAIN,
        #                   col_to_keep=COLUMN_NAMES,
        #                   sample_division=10,
        #                   number_simulations=1,
        #                   fault=None)

        raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/faulty_train_fault1_sim1.csv')
        raw_df = raw_df[:300:10]
        # ---------------------------------------------------------
        # ÉTAPE 2 : Chargement du scaler
        # ---------------------------------------------------------
        logging.info(f"📥 Chargement du scaler")
        scaler = load_preprocessor('scripts/ml_logic/scaler/scaler.pkl')
        logging.info(f"✅ Scaler chargé ")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Prétraitement du dataset
        # ---------------------------------------------------------
        logging.info("⚙️ Traitement des données et split X_eval, y_eval")
        # On suppose que process_data renvoie les sets divisés
        X_eval, y_eval = preprocess_simulation(raw_df, scaler,
                                     timesteps_per_sequence= 50)
        logging.info(f"✅ Données traitées. X_eval shape: {X_eval.shape}, y_eval shape : {y_eval.shape}")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Chargement du modèle entrainé
        # ---------------------------------------------------------
        logging.info("⚙️ Chargement du modèle : {MODEL_NAME}")
        model = load_model(model_name = MODEL_NAME, bucket_name=BUCKET_NAME)
        logging.info("✅ Modèle : {MODEL_NAME} chargé")

        # ---------------------------------------------------------
        # ÉTAPE 5 : Prediction de X_eval
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        # preds, confidence = evaluate_simulation(
        #         df_raw,
        #         model,
        #         scaler,
        #         timesteps_available: int
        #     )
        # print(preds)
        # print(confidence)
        # logging.info(f"✅ Prédictions terminées : {preds}")


    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")





if __name__ == '__main__':
    main()
