# scripts/ml_logic/preprocessor.py
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from params import *

# --- Fonction d'Augmentation de Données ---
def preprocess_and_split(df) -> tuple:
    timesteps_per_sequence = int(500 / SAMPLE_DIVISION)

    df.columns = COLUMN_NAMES
    df['faultNumber'] = df['faultNumber'].map(FAULT_TABLE)

    X = df.drop(columns=['faultNumber', 'simulationRun', 'sample'])
    y = df['faultNumber']

    tscv = TimeSeriesSplit(n_splits=5)
    all_folds = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n{'='*60}")
        print(f"--- Fold {fold+1} ---")

        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        # 🔍 DÉBOGAGE : Vérifier la distribution des labels
        print(f"Distribution y_train: {y_train.value_counts().sort_index()}")
        print(f"Distribution y_test: {y_test.value_counts().sort_index()}")

        num_features = X_train.shape[1]

        # Trimming
        remainder_train = len(X_train) % timesteps_per_sequence
        if remainder_train > 0:
            X_train = X_train.iloc[:-remainder_train]
            y_train = y_train.iloc[:-remainder_train]

        remainder_test = len(X_test) % timesteps_per_sequence
        if remainder_test > 0:
            X_test = X_test.iloc[:-remainder_test]
            y_test = y_test.iloc[:-remainder_test]

        num_sequences_train = len(X_train) // timesteps_per_sequence
        num_sequences_test = len(X_test) // timesteps_per_sequence

        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)

        # Reshape 3D
        X_train_final = X_train_scaled.reshape(
            num_sequences_train, timesteps_per_sequence, num_features
        )
        X_test_final = X_test_scaled.reshape(
            num_sequences_test, timesteps_per_sequence, num_features
        )

        # 🎯 CORRECTION LABELS : Prendre le label majoritaire de chaque séquence
        y_train_seq = []
        for i in range(num_sequences_train):
            start_idx = i * timesteps_per_sequence
            end_idx = start_idx + timesteps_per_sequence
            # Prendre le label le plus fréquent dans la séquence
            seq_labels = y_train.iloc[start_idx:end_idx]
            most_common_label = seq_labels.mode()[0]
            y_train_seq.append(most_common_label)

            # 🚨 ALERTE si la séquence a plusieurs labels
            if seq_labels.nunique() > 1:
                print(f"⚠️ Séquence {i} a {seq_labels.nunique()} labels différents!")

        y_test_seq = []
        for i in range(num_sequences_test):
            start_idx = i * timesteps_per_sequence
            end_idx = start_idx + timesteps_per_sequence
            seq_labels = y_test.iloc[start_idx:end_idx]
            most_common_label = seq_labels.mode()[0]
            y_test_seq.append(most_common_label)

            if seq_labels.nunique() > 1:
                print(f"⚠️ Séquence test {i} a {seq_labels.nunique()} labels différents!")

        y_train_seq = np.array(y_train_seq)
        y_test_seq = np.array(y_test_seq)

        print(f"✅ Fold {fold+1}: Train {X_train_final.shape}, Test {X_test_final.shape}")
        print(f"   y_train_seq shape: {y_train_seq.shape}, unique labels: {np.unique(y_train_seq)}")
        print(f"   y_test_seq shape: {y_test_seq.shape}, unique labels: {np.unique(y_test_seq)}")

        all_folds.append({
            'fold': fold + 1,
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'scaler': scaler
        })

    return all_folds
