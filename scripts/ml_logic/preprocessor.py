import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from scripts.params import *

def preprocess_and_split(df) -> tuple:

    timesteps_per_sequence = 500 / SAMPLE_DIVISION

    int_cols = COLUMN_NAMES[0:3]   # ['faultNumber', 'simulationRun', 'sample']
    float_cols = COLUMN_NAMES[3: 3+N_TH_FIRST_FEATURES]

    dtype_map = {col: 'int16' for col in int_cols}
    dtype_map.update({col: 'float32' for col in float_cols})

    df = df.astype(dtype_map)

    # 1. Simulation des données
    # Drop simulationRun and sample if they are not features
    X = df.drop(columns=['faultNumber', 'simulationRun', 'sample'])
    # 20 classes
    y = df['faultNumber']

    # 2. Préparation de la validation croisée
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Shape globale : {X.shape}")

    # 3. Boucle d'entraînement
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold+1} ---")

        # A. Découpage (Pandas)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        num_features = X_train.shape[1]

        # --- CORRECTION 1 : Forcer le type INT dès le début ---
        # Si SAMPLE_DIVISION vaut 10, cela donne 50 (int) et pas 50.0 (float)
        timesteps_per_sequence = int(500 / SAMPLE_DIVISION)

        # --- CORRECTION 2 : Logique de Trimming unifiée et propre ---
        # On s'assure que X et y sont coupés ensemble

        # Traitement du Train
        remainder_train = len(X_train) % timesteps_per_sequence
        if remainder_train > 0:
            print(f"Warning: Trimming {remainder_train} rows from training set.")
            X_train = X_train.iloc[:-remainder_train]
            y_train = y_train.iloc[:-remainder_train] # Important : Couper y aussi !

        # Traitement du Test
        remainder_test = len(X_test) % timesteps_per_sequence
        if remainder_test > 0:
            print(f"Warning: Trimming {remainder_test} rows from test set.")
            X_test = X_test.iloc[:-remainder_test]
            y_test = y_test.iloc[:-remainder_test]    # Important : Couper y aussi !

        # Calcul sécurisé du nombre de séquences (Division entière //)
        num_sequences_train_fold = len(X_train) // timesteps_per_sequence
        num_sequences_test_fold = len(X_test) // timesteps_per_sequence

        # B. Normalisation
        X_train_np = X_train.values
        X_test_np = X_test.values

        # --- CORRECTION 3 : Sécurisation du Scaler ---
        if SCALER == 'standard':
            scaler = StandardScaler()
        elif SCALER == 'robust':
            scaler = RobustScaler()
        elif SCALER == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"⚠️ Scaler '{SCALER}' not recognized. Defaulting to StandardScaler.")
            scaler = StandardScaler()

        # Fit sur Train, Transform sur Test
        X_train_scaled_2d = scaler.fit_transform(X_train_np)
        X_test_scaled_2d = scaler.transform(X_test_np)

        # C. Reshape (2D -> 3D)
        # Maintenant, toutes les variables de dimension sont garanties d'être des INT
        X_train_final = X_train_scaled_2d.reshape(
            num_sequences_train_fold,
            timesteps_per_sequence,
            num_features
        )

        X_test_final = X_test_scaled_2d.reshape(
            num_sequences_test_fold,
            timesteps_per_sequence,
            num_features
        )

        print(f"✅ Fold {fold+1} Ready. X_train shape: {X_train_final.shape}")

        print(f"Train shape: {X_train_final.shape}, Train type : {type(X_train_final)}")
        print(f"Test shape:  {X_test_final.shape}, Test type : {type(X_test_final)}")

    with open(f"scaler/scaler_fold_{fold+1}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Returning a unique (0-19) label for each simulation
    y_train_seq = y_train.iloc[::timesteps_per_sequence] - 1
    y_test_seq = y_test.iloc[::timesteps_per_sequence] - 1

    return X_train_final, X_test_final, y_train_seq, y_test_seq

def load_preprocessor(scaler_path: str):
    """Charge l'objet Scaler (StandardScaler, RobustScaler, etc.) sauvegardé."""
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler chargé depuis {scaler_path}")
        return scaler
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier scaler non trouvé à {scaler_path}")
        return None

def preprocess_simulation(df: pd.DataFrame, scaler, timesteps_per_sequence: int, n_features=52) -> tuple:
    """
    Applique le même preprocessing (sauf le fit du scaler) à un jeu de données de simulation.
    """

    # 1. Sélection des données (Identique à l'entraînement)
    # On suppose que 'faultNumber' n'est pas présent ou n'est pas utile ici.
    # Pour l'évaluation, nous n'avons besoin que de X (les features).

    # Assurez-vous d'avoir les mêmes colonnes et dtypes
    int_cols = COLUMN_NAMES[0:3]
    float_cols = COLUMN_NAMES[3: 3+N_TH_FIRST_FEATURES]
    dtype_map = {col: 'int16' for col in int_cols}
    dtype_map.update({col: 'float32' for col in float_cols})
    df = df.astype(dtype_map)

    # Séparer les features (X) des labels (y) si les labels sont présents
    # Pour l'évaluation, souvent y (faultNumber) est le label à prédire.
    if 'faultNumber' in df.columns:
        X = df.drop(columns=['faultNumber', 'simulationRun', 'sample'])
        y = df['faultNumber']
    else:
        # Cas où seuls les features sont fournis pour la prédiction
        X = df.drop(columns=['simulationRun', 'sample'])
        y = None # Pas de labels pour l'évaluation si non fournis

    X_np = X.values

    # 2. Trimming (Identique à l'entraînement)
    remainder = len(X_np) % timesteps_per_sequence
    if remainder > 0:
        print(f"Warning: Trimming {remainder} rows from simulation data.")
        X_trimmed = X_np[:-remainder]
        if y is not None:
            y_trimmed = y.iloc[:-remainder]
        else:
            y_trimmed = None
    else:
        X_trimmed = X_np
        y_trimmed = y

    num_sequences = len(X_trimmed) // timesteps_per_sequence

    # 3. Normalisation (UNIQUEMENT .transform())
    if scaler is None:
        raise ValueError("Le scaler n'a pas été chargé correctement. Impossible de normaliser.")

    # Transformation des données brutes avec le scaler entraîné
    X_scaled_2d = scaler.transform(X_trimmed)

    # 4. Reshape (2D -> 3D)
    X_final = X_scaled_2d.reshape(
        num_sequences,
        timesteps_per_sequence,
        n_features
    )

    print(f"✅ Simulation Preprocessed. X_final shape: {X_final.shape}")

    # Préparation du label par séquence si y est disponible
    if y_trimmed is not None:
        y_seq = y_trimmed.iloc[::timesteps_per_sequence] - 1
        return X_final, y_seq
    else:
        return X_final, None
