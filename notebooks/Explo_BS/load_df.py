import pandas as pd

sampled_datasets = ['/home/bapt/code/Monitor-the-Reactor/raw_data/sampled_data_1_10/TEP_FaultFree_Testing_sampled.csv',
'/home/bapt/code/Monitor-the-Reactor/raw_data/sampled_data_1_10/TEP_FaultFree_Training_sampled.csv',
'/home/bapt/code/Monitor-the-Reactor/raw_data/sampled_data_1_10/TEP_Faulty_Testing_sampled.csv',
'/home/bapt/code/Monitor-the-Reactor/raw_data/sampled_data_1_10/TEP_Faulty_Training_sampled.csv']

two_h_datasets = ['/home/bapt/code/Monitor-the-Reactor/raw_data/data_2h/F_test_2h.csv'
    ,'/home/bapt/code/Monitor-the-Reactor/raw_data/data_2h/F_train_2h.csv'
    ,'/home/bapt/code/Monitor-the-Reactor/raw_data/data_2h/FF_test_2h.csv'
    ,'/home/bapt/code/Monitor-the-Reactor/raw_data/data_2h/FF_train_2h.csv'
]

def get_sampled_df_from_csv():

    fault_free_test = pd.read_csv(sampled_datasets[0], index_col=0)
    fault_free_train = pd.read_csv(sampled_datasets[1], index_col=0)
    faulty_test = pd.read_csv(sampled_datasets[2], index_col=0)
    faulty_train = pd.read_csv(sampled_datasets[3], index_col=0)

    return fault_free_test, fault_free_train, faulty_test, faulty_train

def get_2h_df_from_csv():

    faulty_test_two = pd.read_csv(two_h_datasets[0], index_col=0)
    faulty_train_two = pd.read_csv(two_h_datasets[1], index_col=0)
    fault_free_test_two = pd.read_csv(two_h_datasets[2], index_col=0)
    fault_free_train_two = pd.read_csv(two_h_datasets[3], index_col=0)

    return fault_free_test_two, fault_free_train_two, faulty_test_two, faulty_train_two
