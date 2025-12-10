import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from scripts.params import *

def preprocess_and_split_fixed(df) -> tuple:
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

# def preprocess_and_split(df) -> tuple:

    timesteps_per_sequence = 500 / SAMPLE_DIVISION
    df.columns = COLUMN_NAMES
    int_cols = COLUMN_NAMES[0:3]   # ['faultNumber', 'simulationRun', 'sample']
    float_cols = COLUMN_NAMES[3: 3+N_TH_FIRST_FEATURES]

    df['faultNumber'] = df['faultNumber'].map(FAULT_TABLE)

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

    # with open(f"scaler/scaler_fold_{fold+1}.pkl", "wb") as f:
    #     pickle.dump(scaler, f)

    # Returning a unique (0-19) label for each simulation
    y_train_seq = y_train.iloc[::timesteps_per_sequence]
    y_test_seq = y_test.iloc[::timesteps_per_sequence]

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


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import math

# --- Constantes (À adapter à votre script de paramètres) ---
# Longueur de la séquence complète (T)
TIMESTEPS_ENTRAINEMENT = int(500 / SAMPLE_DIVISION)
# Valeur de masquage
MASK_VALUE = -5.0
# Nombre de features (colonnes de données)
N_FEATURES = 52 # Assurez-vous que cette valeur est correcte
COLUMN_NAMES = ['faultNumber', 'simulationRun', 'sample'] + [f'feature_{i}' for i in range(N_FEATURES)]
SCALER = 'standard' # Doit être défini
# --- Fin Constantes ---

def generate_augmented_data(X_original: np.ndarray, y_original: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences tronquées (augmentées) avec MASK_VALUE et duplique les labels.
    """
    N = X_original.shape[0]

    # Nous allons tronquer 50% des séquences originales
    N_to_truncate = N // 2

    indices_to_truncate = np.random.choice(N, size=N_to_truncate, replace=False)
    X_truncated = np.zeros((N_to_truncate, TIMESTEPS_ENTRAINEMENT, N_FEATURES), dtype=X_original.dtype)

    # Dupliquer les labels pour les séquences tronquées
    y_truncated = y_original[indices_to_truncate].copy()

    for i, seq_idx in enumerate(indices_to_truncate):
        # 1. Copier la séquence (N, 50, 52)
        sequence = X_original[seq_idx].copy()

        # 2. Choisir une longueur de troncature aléatoire (entre 1 et 49)
        truncation_length = np.random.randint(1, TIMESTEPS_ENTRAINEMENT)

        # 3. Application du Masking (-5.0) à la fin
        start_mask_index = truncation_length
        sequence[start_mask_index:, :] = MASK_VALUE

        X_truncated[i] = sequence

    # Concaténation des données originales et des nouvelles données tronquées
    X_final = np.concatenate((X_original, X_truncated), axis=0)
    y_final = np.concatenate((y_original, y_truncated), axis=0)

    return X_final, y_final


def prepare_training_data_masked(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prépare les données pour l'entraînement d'un modèle masqué (Masked LSTM),
    en effectuant le découpage, la normalisation, le reshape 3D, et l'augmentation (padding).
    """
# Noms des colonnes que vous voulez traiter comme INTs (les 3 premières)
    int_cols_df = df.columns[0:3]
    # Noms des colonnes que vous voulez traiter comme FLOATs (les 52 suivantes)
    float_cols_df = df.columns[3: 3+N_FEATURES]

    # Reconstruction de dtype_map basée sur les noms réels
    dtype_map = {col: 'int16' for col in int_cols_df}
    dtype_map.update({col: 'float32' for col in float_cols_df})

    # 1. Sélection des données
    X = df.drop(columns=['faultNumber', 'simulationRun', 'sample'])
    y = df['faultNumber']

    # 2. Préparation de la validation croisée
    tscv = TimeSeriesSplit(n_splits=5)

    # Nous allons stocker les résultats du dernier fold
    X_train_final, X_test_final, y_train_seq, y_test_seq = None, None, None, None
    scaler = None

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Processing Fold {fold+1} ---")

        # A. Découpage (Pandas)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # B. Trimming (s'assurer que les longueurs sont multiples de TIMESTEPS_ENTRAINEMENT)
        remainder_train = len(X_train) % TIMESTEPS_ENTRAINEMENT
        if remainder_train > 0:
            X_train = X_train.iloc[:-remainder_train]
            y_train = y_train.iloc[:-remainder_train]

        remainder_test = len(X_test) % TIMESTEPS_ENTRAINEMENT
        if remainder_test > 0:
            X_test = X_test.iloc[:-remainder_test]
            y_test = y_test.iloc[:-remainder_test]

        num_sequences_train = len(X_train) // TIMESTEPS_ENTRAINEMENT
        num_sequences_test = len(X_test) // TIMESTEPS_ENTRAINEMENT

        # C. Normalisation (choix du scaler)
        if SCALER == 'standard':
            scaler = StandardScaler()
        # ... ajoutez les autres scalers (RobustScaler, MinMaxScaler) si nécessaire ...
        else:
            scaler = StandardScaler()

        X_train_np = X_train.values
        X_test_np = X_test.values

        scaler.fit(X_train_np) # Fit UNIQUEMENT sur l'ensemble d'entraînement
        X_train_scaled_2d = scaler.transform(X_train_np)
        X_test_scaled_2d = scaler.transform(X_test_np)

        # D. Reshape 2D -> 3D (Séquences complètes)
        X_train_3d_full = X_train_scaled_2d.reshape(num_sequences_train, TIMESTEPS_ENTRAINEMENT, N_FEATURES)
        X_test_3d_full = X_test_scaled_2d.reshape(num_sequences_test, TIMESTEPS_ENTRAINEMENT, N_FEATURES)

        # Labels pour l'entraînement (un label par séquence)
        y_train_seq_full = y_train.iloc[::TIMESTEPS_ENTRAINEMENT].values - 1 # 0-19
        y_test_seq_full = y_test.iloc[::TIMESTEPS_ENTRAINEMENT].values - 1  # 0-19

        # E. Augmentation des données (Padding avec MASK_VALUE) - UNIQUEMENT SUR TRAIN
        X_train_final, y_train_final = generate_augmented_data(X_train_3d_full, y_train_seq_full)

        # Stockage du résultat du dernier fold
        X_test_final = X_test_3d_full # Le test reste non-augmenté
        y_test_final = y_test_seq_full

        print(f"   ✅ Fold {fold+1} Ready. Train 3D: {X_train_final.shape}, Test 3D: {X_test_final.shape}")

    # Sauvegarder le dernier scaler pour l'évaluation si nécessaire
    # with open("scaler_final.pkl", "wb") as f:
    #     pickle.dump(scaler, f)

    return X_train_final, X_test_final, y_train_final, y_test_final, scaler


def preprocess_and_predict_partial_bulk_masked(
    X_partial_3d: np.ndarray, # Entrée: (N, TIMESTEPS_PARTIAL, N_FEATURES)
    scaler,
    model
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prétraite un lot de séquences (normalisation, padding avec MASK_VALUE)
    et génère les prédictions (labels 1-20 et confiance).

    Args:
        X_partial_3d (np.ndarray): Séquences 3D tronquées (ex: (N, 20, 52)).
        scaler: L'objet scaler FIT pendant l'entraînement.
        model (Model): Le modèle Keras entraîné avec la couche Masking(-5.0).

    Returns:
        tuple[np.ndarray, np.ndarray]: Classes prédites (1-20) et confiance associée.
    """

    # --- CONSTANTES ---
    # Longueur de la séquence complète attendue par le modèle
    TIMESTEPS_ENTRAINEMENT = 50
    # Nombre de features
    N_FEATURES = 52
    # Valeur de masquage - DOIT correspondre à la valeur dans la couche Masking du modèle
    MASK_VALUE = -5.0
    # --- FIN CONSTANTES ---

    # 1. Lecture des formes
    N, TIMESTEPS_PARTIAL, N_FEATURES_input = X_partial_3d.shape

    # Vérification de sécurité des features
    if N_FEATURES_input != N_FEATURES:
        raise ValueError(f"Le nombre de features est incorrect. Attendu: {N_FEATURES}, Reçu: {N_FEATURES_input}")

    # 2. Reshape de 3D vers 2D pour la normalisation
    X_2d_raw = X_partial_3d.reshape(-1, N_FEATURES) # Forme: (N * TIMESTEPS_PARTIAL, 52)

    # 3. Normalisation (UNIQUEMENT .transform())
    # Applique la transformation apprise sur le jeu d'entraînement.
    X_scaled_2d = scaler.transform(X_2d_raw)

    # 4. Reshape vers 3D (N, TIMESTEPS_PARTIAL, 52)
    X_scaled_3d = X_scaled_2d.reshape(N, TIMESTEPS_PARTIAL, N_FEATURES)

    # 5. Padding Temporel (N, T_PARTIAL, 52) -> (N, 50, 52)
    padding_needed = TIMESTEPS_ENTRAINEMENT - TIMESTEPS_PARTIAL

    if padding_needed > 0:
        # Création du padding en utilisant MASK_VALUE (-5.0)
        padding_values = np.full(
            (N, padding_needed, N_FEATURES),
            MASK_VALUE,
            dtype=X_partial_3d.dtype
        )

        # Concaténation le long de l'axe des timesteps (axe 1)
        X_padded_final = np.concatenate((X_scaled_3d, padding_values), axis=1) # Forme (N, 50, 52)
    else:
        # Si TIMESTEPS_PARTIAL >= TIMESTEPS_ENTRAINEMENT (pas de padding nécessaire ou trop long)
        X_padded_final = X_scaled_3d

    # 6. Prédiction
    # Le modèle (avec la couche Masking) ignorera les valeurs -5.0
    y_pred_probs = model.predict(X_padded_final)

    # 7. Post-traitement (labels 1-20 et confiance)
    N_sequences = len(y_pred_probs)
    # Index de la classe prédite (0-19)
    y_pred_classes_indices = np.argmax(y_pred_probs, axis=1)

    # Conversion de l'index 0-19 au label 1-20
    y_pred_classes = y_pred_classes_indices + 1

    # Extraction de la probabilité maximale (confiance)
    y_pred_confidence = y_pred_probs[np.arange(N_sequences), y_pred_classes_indices]

    return y_pred_classes, y_pred_confidence
