# scripts/params.py

import os

# --- CONTEXTE DE DONNÉES ---
SAMPLE_DIVISION = 1 # Si 500 / 10 = 50 timesteps par séquence
TIMESTEPS_ENTRAINEMENT = int(500 / SAMPLE_DIVISION) # 50
N_FEATURES = 52 # 52 features de simulation
COLUMN_NAMES = ['faultNumber', 'simulationRun', 'sample'] + [f'feature_{i}' for i in range(N_FEATURES)]

# --- HYPERPARAMÈTRES ET ARCHITECTURE ---
SCALER = 'standard'
MODEL_ARCHITECTURE = 'CNN' # RNN ou CNN

# --- MASKING (CRUCIAL POUR LA DÉTECTION PRÉCOCE) ---
MASK_VALUE = -5.0 # Valeur utilisée pour le padding et le masquage Keras
N_CLASSES = 11 # Classes de faute (labels 1 à 20)

FAULT_TABLE = {0:0,
               1:1,
               2:2,
                4:3,
               6:4,
               7:5,
               12:6,
               13:7,
               14:8,
               17:9,
               18:10
                }
