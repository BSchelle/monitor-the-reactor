import pandas as pd

def convert_csv_types(input_csv_path):
    """
    Lit un CSV, convertit les colonnes selon les préfixes spécifiés.

    Args:
        input_csv_path (str): Chemin vers le fichier CSV d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame avec les types convertis.
    """
    # Lire le CSV
    data = pd.read_csv(input_csv_path)

    # Conversion des colonnes commençant par 'fault', 'simul', ou 'samp' en int16
    cols_to_convert = [col for col in data.columns if col.startswith(('fault', 'simul', 'samp'))]
    data[cols_to_convert] = data[cols_to_convert].astype('int16')

    # Conversion des colonnes commençant par 'xmeas' ou 'xmv' en float32
    cols_to_convert = [col for col in data.columns if col.startswith(('xmeas', 'xmv'))]
    data[cols_to_convert] = data[cols_to_convert].astype('float32')

    return data
