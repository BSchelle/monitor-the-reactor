import numpy as np
import pandas as pd

from pathlib import Path
from google.cloud import bigquery
from colorama import Fore, Style

from scripts.params import *
from scripts.ml_logic import preprocessor, load_data

def preprocess(sample_size=500, simulation_nb=500, ):
