import numpy as np
import pandas as pd

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

    # Returning a unique (0-19) label for each simulation
    y_train_seq = y_train.iloc[::timesteps_per_sequence] - 1
    y_test_seq = y_test.iloc[::timesteps_per_sequence] - 1

    return X_train_final, X_test_final, y_train_seq, y_test_seq
