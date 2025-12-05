import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from scripts.params import *

def preprocess_and_split(df) -> tuple:

    timesteps_per_sequence = 500 / SAMPLE_DIVISION

    int_cols = COLUMN_NAMES_RAW[0:3]   # ['faultNumber', 'simulationRun', 'sample']
    float_cols = COLUMN_NAMES_RAW[3:]

    dtype_map = {col: 'int16' for col in int_cols}
    dtype_map.update({col: 'float32' for col in float_cols})

    df = df.astype(dtype_map)

    # 1. Simulation des données
    # (500 séquences, 50 pas de temps, 52 features)
    X = df.drop(columns=['faultNumber', 'simulationRun', 'sample']) # Drop simulationRun and sample if they are not features

    # 20 classes
    y = df['faultNumber']

    # 2. Préparation de la validation croisée
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Shape globale : {X.shape}")

    # 3. Boucle d'entraînement
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold+1} ---")

        # A. Découpage
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # B. Normalisation (L'étape critique avec 52 features)
        # On doit aplatir les données pour le scaler : (N_seq * Time, Features)
        # Cela permet de normaliser chaque feature indépendamment du temps

        num_features = X_train.shape[1]        # Number of features
        timesteps_per_sequence = 500 / SAMPLE_DIVISION # Assuming 500 timesteps per sequence. PLEASE VERIFY.

        # Ensure training set has complete sequences by trimming excess rows
        num_rows_train_fold = X_train.shape[0]
        remainder_train = num_rows_train_fold % timesteps_per_sequence
        if remainder_train != 0:
            print(f"Warning: Trimming {remainder_train} rows from training set to ensure complete sequences.")
            X_train = X_train.iloc[:-remainder_train]
            y_train = y_train.iloc[:-remainder_train]
            num_rows_train_fold = X_train.shape[0]
        num_sequences_train_fold = num_rows_train_fold // timesteps_per_sequence

        # Ensure test set has complete sequences by trimming excess rows
        num_rows_test_fold = X_test.shape[0]
        remainder_test = num_rows_test_fold % timesteps_per_sequence
        if remainder_test != 0:
            print(f"Warning: Trimming {remainder_test} rows from test set to ensure complete sequences.")
            X_test = X_test.iloc[:-remainder_test]
            y_test = y_test.iloc[:-remainder_test]
            num_rows_test_fold = X_test.shape[0]
        num_sequences_test_fold = num_rows_test_fold // timesteps_per_sequence

        # Convert DataFrames to NumPy arrays before scaling and reshaping
        X_train_np = X_train.values
        X_test_np = X_test.values

        if SCALER == 'standard':
            scaler = StandardScaler()
        elif SCALER == 'robust':
            scaler = RobustScaler()
        elif SCALER == 'minmax':
            scaler = MinMaxScaler()
        else:
            print('Scaler not defined!')

        # FIT only on TRAIN! The data is already in 2D (num_rows, num_features) for scaling.
        X_train_scaled_2d = scaler.fit_transform(X_train_np)
        # TRANSFORM on TEST (using stats from train)
        X_test_scaled_2d = scaler.transform(X_test_np)

        # Return to 3D for the RNN: (num_sequences, timesteps_per_sequence, num_features)
        X_train_final = X_train_scaled_2d.reshape(num_sequences_train_fold, timesteps_per_sequence, num_features)
        X_test_final = X_test_scaled_2d.reshape(num_sequences_test_fold, timesteps_per_sequence, num_features)

        print(f"Train shape: {X_train_final.shape}, Train type : {type(X_train_final)}")
        print(f"Test shape:  {X_test_final.shape}, Test type : {type(X_test_final)}")

    return (X_train_final, X_test_final)
