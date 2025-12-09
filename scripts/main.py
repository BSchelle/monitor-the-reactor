import numpy as np
import pandas as pd

from pathlib import Path
from google.cloud import bigquery
from colorama import Fore, Style

from scripts.get_data import get_data
from scripts.ml_logic.preprocessor import preprocess_and_split, \
    load_preprocessor, preprocess_simulation

from scripts.ml_logic.model import initialize_model_CNN, initialize_model_RNN, \
    compile_model, train_model, make_predictions, evaluate_and_get_f1, \
        save_model, load_model

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
                          sample_division=SAMPLE_DIVISION,
                        #   fault=tuple(i for i in range(21) if i not in [9,10,15,16])
                        )

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
        save_model(model, f1, BUCKET_NAME)

        # ---------------------------------------------------------
        # ÉTAPE 6 : Prédictions (Optionnel ou sur un set de validation)
        # ---------------------------------------------------------
        logging.info("🔮 Génération de prédictions exemples...")
        random_indices = np.random.choice(X_test.shape[0], size=10, replace=False)
        sample_input = X_test[random_indices]
        preds = make_predictions(model, sample_input)
        logging.info(f"✅ Prédictions terminées : {preds}")

    except Exception as e:
        logging.error(f"❌ Une erreur critique est survenue dans le pipeline : {e}")
        # Re-raise l'erreur pour que le job Cloud soit marqué comme 'Failed'
        raise e

    logging.info("🏁 Fin du pipeline avec succès.")


def main():
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
                          sample_division=SAMPLE_DIVISION,
                        #   fault=tuple(i for i in range(21) if i not in [9,10,15,16])
                        )

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
        #                   number_simulations=1,
        #                   fault=(1))

        # raw_df = pd.read_csv('/home/bapt/code/Monitor-the-Reactor/data/processed_data/faulty_train_fault1_sim1.csv')
        # raw_df = raw_df[::10]
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


if __name__ == '__main__':
    main()
