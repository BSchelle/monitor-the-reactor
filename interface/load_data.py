import os
import pandas as pd

def load_data_fault_free_test():
    path = os.path.join('raw_data/TEP_FaultFree_Testing.csv')
    return pd.read_csv(path)

def load_data_fault_free_train():
    path = os.path.join('raw_data/TEP_FaultFree_Training.csv')
    return pd.read_csv(path)

def load_data_faulty_test():
    path = os.path.join('raw_data/TEP_Faulty_Testing.csv')
    return pd.read_csv(path)

def load_data_faulty_train():
    path = os.path.join('raw_data/TEP_Faulty_Trainig.csv')
    return pd.read_csv(path)
