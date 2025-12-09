import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from scripts.params import * # Assuming this contains COLUMN_NAMES, N_TH_FIRST_FEATURES, SAMPLE_DIVISION

def preprocess_and_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    scaler_type: StandardScaler = StandardScaler
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prépare et sépare les données de séries temporelles pour un modèle RNN.

    Args:
        df (pd.DataFrame): DataFrame contenant les features et les labels ('faultNumber').
        n_splits (int): Nombre de splits pour TimeSeriesSplit. Le dernier split est retourné.
        scaler_type (StandardScaler, RobustScaler, etc.): Type de scaler à utiliser.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) en format NumPy 3D/1D.
    """

    print("--- Préparation des données de Séries Temporelles ---")

    # Séparation des features (X) et de la cible (Y)
    # NOTE: Assurez-vous que df contient les colonnes nécessaires (ex: 'faultNumber')
    feature_cols = [col for col in df.columns if col not in ['faultNumber', 'simulationRun', 'sample']]

    X = df[feature_cols]
    y = df['faultNumber']

    if X.shape[1] != NUM_FEATURES:
        raise ValueError(f"Le nombre de features attendu ({NUM_FEATURES}) ne correspond pas au DataFrame ({X.shape[1]}).")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Nous itérons pour obtenir les index du DERNIER fold (le plus grand jeu de test chronologique)
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):

        # A. Découpage chronologique du Fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # B. Trimming des données pour assurer des séquences complètes

        # --- Train Set Trimming ---
        num_rows_train_fold = X_train.shape[0]
        remainder_train = num_rows_train_fold % TIMESTEPS_PER_SEQUENCE
        if remainder_train != 0:
            X_train = X_train.iloc[:-remainder_train]
            y_train_trimmed = y_train.iloc[:-remainder_train]
        else:
            y_train_trimmed = y_train

        # --- Test Set Trimming ---
        num_rows_test_fold = X_test.shape[0]
        remainder_test = num_rows_test_fold % TIMESTEPS_PER_SEQUENCE
        if remainder_test != 0:
            X_test = X_test.iloc[:-remainder_test]
            y_test_trimmed = y_test.iloc[:-remainder_test]
        else:
            y_test_trimmed = y_test

    # --- C. Normalisation et Reshape (Utilisation des données du dernier fold) ---

    num_sequences_train = X_train.shape[0] // TIMESTEPS_PER_SEQUENCE
    num_sequences_test = X_test.shape[0] // TIMESTEPS_PER_SEQUENCE

    # 1. Scaling (Ajustement UNIQUEMENT sur X_train)
    scaler = scaler_type()
    X_train_scaled_2d = scaler.fit_transform(X_train.values)
    X_test_scaled_2d = scaler.transform(X_test.values)

    # 2. Reshape X pour le RNN (N_sequences, Timesteps, Features)
    X_train_final = X_train_scaled_2d.reshape(
        num_sequences_train, TIMESTEPS_PER_SEQUENCE, NUM_FEATURES
    )
    X_test_final = X_test_scaled_2d.reshape(
        num_sequences_test, TIMESTEPS_PER_SEQUENCE, NUM_FEATURES
    )

    # 3. Reshape Y (Subsample to one label per sequence)
    # y_final doit avoir la taille (N_sequences, )
    y_train_final = y_train_trimmed.iloc[::TIMESTEPS_PER_SEQUENCE].values - 1 # -1 pour indexer de 0
    y_test_final = y_test_trimmed.iloc[::TIMESTEPS_PER_SEQUENCE].values - 1 # -1 pour indexer de 0

    # 4. Vérification finale (Pour attraper l'erreur 8333 vs 8334)
    if X_train_final.shape[0] != y_train_final.shape[0] or X_test_final.shape[0] != y_test_final.shape[0]:
        raise ValueError("Erreur critique de cardinalité après reshape/subsample. Les formes X et Y ne correspondent pas.")

    print(f"\n✅ Data Split Terminé (Fold {fold+1}):")
    print(f"   X_train final: {X_train_final.shape}, Y_train final: {y_train_final.shape}")
    print(f"   X_test final:  {X_test_final.shape}, Y_test final:  {y_test_final.shape}")

    return X_train_final, X_test_final, y_train_final, y_test_final
