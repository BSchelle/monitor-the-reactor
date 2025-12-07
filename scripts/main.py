import numpy as np
import pandas as pd

from pathlib import Path
from google.cloud import bigquery
from colorama import Fore, Style

from scripts.get_data import get_data
from scripts.ml_logic.preprocessor import preprocess_and_split
from scripts.ml_logic.model import initialize_model_CNN, initialize_model_RNN, \
    compile_model, train_model, make_predictions, evaluate_and_get_f1, \
        save_model

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
                          sample_division=SAMPLE_DIVISION)

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
        model, history = train_model(model, X_train, y_train)
        logging.info("✅ Modèle entraîné avec succès.")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Sauvegarde du modèle
        # ---------------------------------------------------------
        logging.info(f"💾 Sauvegarde du modèle vers {MODEL_PATH}...")
        # save_model(model, MODEL_PATH)
        # Note : Si vous êtes sur GCC, pensez à uploader ce fichier sur GCS (Google Cloud Storage)
        # pour ne pas le perdre si la VM s'éteint.

        # ---------------------------------------------------------
        # ÉTAPE 5 : Test et Métriques
        # ---------------------------------------------------------
        logging.info("📊 Évaluation des performances...")
        metrics = evaluate_and_get_f1(model, X_test, y_test)
        logging.info(f"📈 Résultats du test : {metrics}")

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

if __name__ == '__main__':
    main()
