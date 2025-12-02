import os
import pandas as pd

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
