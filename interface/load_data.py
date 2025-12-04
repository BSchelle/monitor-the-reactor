import os
import pandas as pd

# Chargement de l'entiereté du dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data_fault_free_test():
    path = os.path.join(BASE_DIR, '..', 'raw_data', 'TEP_FaultFree_Testing.csv')
    return pd.read_csv(path)

def load_data_fault_free_train():
    path = os.path.join(BASE_DIR, '..', 'raw_data', 'TEP_FaultFree_Training.csv')
    return pd.read_csv(path)

def load_data_faulty_test():
    path = os.path.join(BASE_DIR, '..', 'raw_data', 'TEP_Faulty_Testing.csv')
    return pd.read_csv(path)

def load_data_faulty_train():
    path = os.path.join(BASE_DIR, '..', 'raw_data', 'TEP_Faulty_Training.csv')
    return pd.read_csv(path)


# Selectionne un nombre definis de features classées par ordre d'importance
# A UTILISER UNIQUEMENT AVEC FAUTES 6, 18, 7, 13 !!!

def load_data_fault_free_test_select(features=4):
    df = load_data_fault_free_test()
    X = df.drop(columns=['faultNumber', 'simulationRun', 'sample'])

    X_6 = X.iloc[2500:3000]
    X_18 = X.iloc[8500:9000]
    X_7 = X.iloc[3000:3500]
    X_13 = X.iloc[6000:6500]

    corr_6 = X_6.corr()
    corr_18 = X_18.corr()
    corr_7 = X_7.corr()
    corr_13 = X_13.corr()

    mask_6 = (corr_6 >= 0.3) & (corr_6 <= 0.8)
    mask_18 = (corr_18 >= 0.3) & (corr_18 <= 0.8)
    mask_7 = (corr_7 >= 0.3) & (corr_7 <= 0.8)
    mask_13 = (corr_13 >= 0.3) & (corr_13 <= 0.8)

    feature_select_name = ((mask_6 == mask_18) == (mask_7 == mask_13)).sum()\
        .sort_values(ascending=False).index

    df_feat_selected = df[feature_select_name[:features]]

    return df_feat_selected
